import time
import logging
import sys
import os
import numpy as np
import traceback
import abc
from dataclasses import dataclass
from numpy.typing import NDArray

from xframe import settings
from xframe import database
from xframe import Multiprocessing
from xframe.library.pythonLibrary import (DictNamespace,
                                          xprint)
from .projectLibrary.misk import _get_reciprocity_coefficient
from xframe.interfaces import ProjectWorkerInterface
from xframe.library.math_transforms import (HankelTransformWeights,
                                            HankelWeightStruct,
                                            SphericalFourierTransform,
                                            SphericalFourierTransformStruct)
from xframe.library.mathLibrary import (PolarIntegrator,
                                       SphericalIntegrator,
                                       SampleShapeFunctions,
                                       get_test_function)
from .projectLibrary.fxs_Projections import RealProjectionSNR,ReciprocalProjection
from .projectLibrary.fxs_IO_methods import generate_error_routines,generate_main_error_routine
log=logging.getLogger('root')

opt = None
db = None
comm_module = None
def set_globals():
    global opt
    opt = settings.project
    global db
    db = database.project
    global comm_module
    comm_module = Multiprocessing.comm_module



class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        #log.info('analysis worker init')
        self.opt = settings.project
        MTIP.preinit()
        self.mtip = MTIP
        if settings.project.GPU.use:
            n_gpus = Multiprocessing.get_number_of_gpus()
            if n_gpus == 0:
                log.warning('Problem with GPU initialization. There maybe no GPU or pyOpenCL is missing. Defaulting to CPU only computations.')
                settings.project.GPU.use = False
                settings.general.n_control_workers = 0
                xprint('Starting Reconstruction in CPU only mode. This is gonna be slow!')
            else:
                settings.general.n_control_workers = settings.project.GPU.get("n_gpu_workers",6)
            Multiprocessing.comm_module.restart_control_worker()
        else:
            xprint('Starting Reconstruction in CPU only mode. This is gonna be slow!')

        self.results={}
        self.results['stats']={}
    
    def setup_phasing_loop(self):
        self.mtip = MTIP(self.process_factory)
        self.mtip.generate_phasing_loop()
    def start_phasing(self):
        run_profiling=settings.project.get('profiling',DictNamespace(enable=False)).enable
        m = self.mtip
        #log.info("run profiling = {}".format(run_profiling))
        if run_profiling:
            process_id = settings.project.profiling.reconstruction_process_id
            if not settings.project.multi_process.use:
                process_id=0
            log.info("Profiling: id = {} profile_id = {}".format(Multiprocessing.get_process_name(),process_id))
            if Multiprocessing.get_process_name() == np.abs(process_id):
                path = db.get_reconstruction_path()
                db.create_path_if_nonexistent(path)
                path += "reconstruction_{}.stats".format(process_id)
                log.info('profile path = {}'.format(path))
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
                result = m.phasing_loop()
                profiler.disable()
                profiler.dump_stats(path)                    
            else:
                result = m.phasing_loop()
        else:
            result = m.phasing_loop()
        return result
        
    def start_parallel_phasing(self):
        def generate_phasing_run_phasing(*opt,**kwargs):
            self.setup_phasing_loop()
            result = self.start_phasing()
            return result
        
        mp_mode = Multiprocessing.MPMode_Queue(assemble_outputs = False)        
        mp_opt = settings.project.multi_process
        xprint(f'Spawning phasing processes executing:\n{self.mtip.loops_str()}')
        n_processes = Multiprocessing._read_number_of_processes(mp_opt.n_parallel_reconstructions)
        result = Multiprocessing.comm_module.request_mp_evaluation(generate_phasing_run_phasing,input_arrays=[np.arange(n_processes)],mode=mp_mode,call_with_multiple_arguments=False,n_processes = n_processes)
        xprint('\nFinished phasing!')
        return result


    def post_processing(self):
        db = database.project
        opt = settings.project
        save = db.save
        try:
            processed_results=self.results.get('MTIP',{})
            stats=self.results.get('stats',{})
            processed_results_list=[result_dict for result_dict in processed_results.values()]
            #log.info([type(result_dict) for result_dict in processed_results])
            #log.info([result_dict.shape for result_dict in processed_results])
            errors = []
            for result_dict in processed_results_list:
                grid_pair = result_dict.pop('grid_pair')
                ft_struct = result_dict.pop('fourier_transform_struct')
                errors.append(result_dict['error_dict']['main'][-1])
                projection_matrices = result_dict.pop('projection_matrices')                
            r_ids=np.argsort(errors)
            log.info('error sorted reconstruction = {} \n errors ={}'.format(r_ids,np.array(errors)[r_ids]))
            processed_results_dict={str(_id):processed_results_list[_id] for _id in r_ids}
            reciprocity_coefficient = _get_reciprocity_coefficient(settings.project.fourier_transform)
            data_dict={'configuration':{'internal_grid':grid_pair,'xray_wavelength':self.mtip.load_mtip_data()[0]['xray_wavelength'],'reciprocity_coefficient':reciprocity_coefficient},'reconstruction_results':processed_results_dict,'projection_matrices':projection_matrices,'stats':stats,'fourier_transform_struct':ft_struct}
            save('reconstructions',data_dict)
        except Exception as e:
            log.info(f'Error during postprocessing / saving with message:\n {e}')
            log.debug(traceback.format_exc())            
    
    def run(self):
        opt = settings.project
        start_time = time.time()
        if opt.multi_process.use:
            result=self.start_parallel_phasing()
        else:
            xprint('Generating phasing loop.')
            self.setup_phasing_loop()
            xprint('done.\n')
            xprint(f'Start phasing:\n{self.mtip.loops_str()} ')
            data = self.start_phasing()
            xprint('\nFinished phasing!')
            result = np.array([data],dtype = object)
            
        xprint('\nSaving results.')
        self.results['MTIP']=result
        self.results['stats']['run_time'] = time.time()-start_time
        self.post_processing()
        return result,locals()

class MTIP:
   # all subsequent parameters are set by a call to MTIP.preinit
    dimensions = 'not set'
    mtip_data = 'not_set'
    data_q_limits = 'not_set'
    
    data_number_of_radial_points = 'not_set'
    real_radial_points = 'not_set'
    reciprocal_radial_points = 'not_set'
    max_q = 'not_set'
    fourier_transform_weights = 'not_set'
    error_metric_parameters_dict = {} 
    preinit_was_called = False

    loops_opt = settings.project.main_loop.sub_loops


    @classmethod
    def preinit(cls,init_data = None):
        '''
        This Method prepares needed quantities which require multiprocessing that are equal for all MTIP instances before __init__ is called in a multi process environment. This is currently done only to preload the fourier transform weights.
        '''
        if init_data is None:
            tmp = cls.load_mtip_data()
        else:
            tmp = init_data
        opt = settings.project
        cls.dimensions = opt.dimensions
        cls.mtip_data = tmp[0]
        cls.data_q_limits = tmp[1]
        cls.data_number_of_radial_points = tmp[2]
        cls.max_q = opt.grid.max_q
        max_order = opt.grid.max_order
        n_radial_points = opt.grid.n_radial_points
        reciprocity_coefficient = opt.fourier_transform.reciprocity_coefficient
        ft_type = opt.fourier_transform.type
        n_processes = opt.multi_process.n_weight_generating_processes

        hankel_struct = HankelWeightStruct(dimension = opt.dimensions,
                                           n_radial_points = n_radial_points,
                                           angular_bandwidth = max_order+1,
                                           hankel_type = ft_type,
                                           n_processes_for_weight_generation = n_processes)
        weight_dict = load_fourier_transform_weights(hankel_struct,allow_weight_saving=opt.fourier_transform.allow_weight_saving)

        cls.fourier_transform_weights = weight_dict.pop('weights')
        cls.preinit_fourier_struct = SphericalFourierTransformStruct(**weight_dict)
        cls.preinit_fourier_struct.dimension = hankel_struct.dimension
        cls.preinit_fourier_struct.n_radial_points = hankel_struct.n_radial_points
        cls.preinit_fourier_struct.angular_bandwidth = hankel_struct.angular_bandwidth
        cls.preinit_fourier_struct.hankel_type = hankel_struct.hankel_type
        cls.preinit_fourier_struct.n_processes_for_weight_generation = hankel_struct.n_processes_for_weight_generation

    @classmethod
    def load_mtip_data(cls):
        opt = settings.project
        db = database.project
        data = db.load('invariants',path_modifiers={'structure_name':opt.structure_name,'dimensions':opt.dimensions})        
        data_q_limits = [data['data_radial_points'].min(),data['data_radial_points'].max()]
        data_number_of_radial_points = len(data['data_radial_points'])
        return [data,data_q_limits,data_number_of_radial_points]

    @classmethod
    def loops_str(cls):
        loops_str = 'Loops:\n'
        opt = cls.loops_opt
        max_loop_name_size = max([len(name)for name in loops_opt.order])
        for loop_name in opt.order:            
            loop_opt = opt.get(loop_name,{})
            iterations = opt.get('iterations','')
            methods_string = ''
            for method in opt.get('order',[]):
                method_opt = opt.methods.get(method,{})
                if isinstance(method_opt,(dict,DictNamespace)):
                    method_iteration = method_opt['iterations']
                else:
                    method_iteration = method_opt
            methods_string += f'{method_iteration}x{method} '
        loops_str += f'\t{loop_name}:'+' '*(max_loop_name_size-len(loop_name))+ f'\t {iterations}x( '+''.join(methods_string)+')\n'
        return loops_str
        
    def __init__(self):
        set_globals()
        self.opt = settings.project
        self.results={}        
        
        self.fourier_trf,self.harmonic_trf,self.grid_pair = self.assemble_transform_op_and_grid()
        self.max_order = self.harmonic_trf.max_order
        self.inv_proj = ReciprocalProjection(self.grid_pair['reciprocal'],self.mtip_data,self.max_order)
        self.real_proj =  RealProjectionSNR(opt.projections.real.projections,self.fourier_trf.real_grid)
        self.error_routines = self.assemble_error_routines()
        self.main_error_routine = generate_main_error_routine(self.opt.main_loop.error.methods.main.metrics,
                                                         self.opt.main_loop.error.methods.main.type)
        self.op_dict = {"fourier_transform":self.fourier_trf,
                        "harmonic_transform":self.harmonic_trf,
                        "invariant_projection":self.inv_proj,
                        "real_projection": self.real_proj}
        
        self.phasing_sketch = self.get_phasing_sketch()

    def assemble_transform_op_and_grid(self):
        max_q = self.max_q
        max_q_is_set =  isinstance(max_q,float) or (isinstance(max_q,int) and (not isinstance(max_q,bool)))
        if not max_q_is_set:
            self.max_q = max_q = self.data_q_limits[1]
        Nr,Nq = HankelTransformWeights._read_n_points(opt.grid.n_radial_points)
        
        harmonic_transform_opt = { i:opt.grid.get(i,0) for i in ['n_phi','n_theta']}
        max_nonzero_r = opt.grid.max_nonzero_r
        if max_nonzero_r is None:
            r_support = None
        else:
            r_support =  opt.particle_radius*max_nonzero_r


        weights = self.fourier_transform_weights
        struct = self.preinit_fourier_struct
        struct.max_q = max_q
        struct.max_nonzero_r = r_support
        struct.use_gpu = opt.GPU.use
        if 'n_phi' in harmonic_transform_opt:
            struct.n_polar_angles = harmonic_transform_opt['n_phi']
        if 'n_theta' in harmonic_transform_opt:
            struct.n_azimutal_angles = harmonic_transform_opt['n_theta']

        sft = SphericalFourierTransform(struct,weights = weights)

        if self.dimensions == 2:
            bandwidth = self.fourier_transform_weights['bandwidth']
            ht =  get_harmonic_transform(bandwidth, dimensions = dimensions, options=harmonic_transform_opt) 
        elif self.dimensions == 3:
            ht = sft.harm
        grid_pair = {'real':sft.real_grid,'reciprocal':sft.reciprocal_grid}
        return sft,ht,grid_pair
    
    def assemble_error_routines(self):
        grid_pair = self.grid_pair
        rp = self.inv_proj
        error_opt = opt.main_loop.error
        deg2_invariants = rp.deg2_invariants
        used_orders = rp.used_orders
        xray_wavelength = self.mtip_data['xray_wavelength']
        invariant_mask = rp.radial_mask[:,:,None] * rp.radial_mask[:,None,:]
        error_routines = generate_error_routines(error_opt,
                                                 grid_pair,
                                                 deg2_invariants = deg2_invariants,
                                                 projection_matrices=rp.projection_matrices,
                                                 used_orders = used_orders,
                                                 n_particles = 1,
                                                 invariant_mask = invariant_mask,
                                                 xray_wavelength = xray_wavelength)
        return error_routines
    
    def init_error_dict(self):
        error_names = self.opt.main_loop.error.methods
        err_dict = {}
        err_dict['real'] = {name:[] for name in error_names['real']['calculate']}        
        err_dict['reciprocal'] = {name:[] for name in error_names['reciprocal']['calculate']}        
        err_dict['main'] = []
        return err_dict

    def get_phasing_sketch(self):
        phasing_sketch = {}
        opt = self.opt.main_loop.sub_loops
        loop_names = opt.order
        for loop_name in loop_names:
            loop_opt = opt[loop_name]
            phasing_sketch[loop_name]=({},loop_opt.iterations)
            method_names = loop_opt.order
            for method_name in method_names:
                mopt = loop_opt.methods[method_name]
                camel_name = snake_to_camel_simple(method_name)
                method_cls = globals().get(camel_name,None) 
                if method_cls is not None:
                    method = method_cls(self.op_dict,mopt)
                    phasing_sketch[loop_name][0][method_name]=(method,mopt.iterations)
                else:
                    raise ValueError(f"Class '{camel_name}' for method '{method_name}' does not exist.")
        return phasing_sketch
    def create_initial_density(self):
        opt=settings.project.density_guess
        real_grid = self.grid_pair["real"]
        if opt['amplitude_function']=='random':
            def amplitude_function(points):
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                return 1+1/opt.random.SNR*np.random.rand(*points.shape[:-1])
        radius = opt['radius']
        if isinstance(radius,bool):
            radius = settings.project.particle_radius


        if self.dimensions == 2:                
            integrator = PolarIntegrator(real_grid[:])
        elif self.dimensions == 3:
            integrator = SphericalIntegrator(real_grid[:])
            
        total_intensity = self.inv_proj.integrated_intensity

        if radius<0:
                radius = np.max(real_grid[...,0])
                
        if opt["type"]=="ball":
            coord_sys=opt['coord_sys']
            guessed_density_function=SampleShapeFunctions.get_disk_function(radius,amplitude_function=amplitude_function,coordSys=coord_sys)
            density = guessed_density_function(real_grid)
            total_squared_density = integrator((density*density.conj()).real)
            density*=np.sqrt(total_intensity/total_squared_density)
            density = density.astype(complex)            
                
        elif opt["type"]=="bump":
            slope = opt.bump.slope
            bump_func = get_test_function(support=[-radius,radius],slope=slope)

            A=amplitude_function(real_grid[:])
            density = A*bump_func(real_grid[...,0])
            total_squared_density = integrator((density*density.conj()).real)
            density*=np.sqrt(total_intensity/total_squared_density)
            density = density.astype(complex)
        else:
            raise ValueError(f"Initial density type '{opt["type"]}' is unknown.")    
        return density
    def create_initial_state(self):
        density = self.create_initial_density()
        ft_density = self.fourier_trf.forward_cmplx(density)
        error_dict = self.init_error_dict()
        state = PhasingState(density_history=[None,density],
                             ft_density_history=[None,ft_density],
                             error_dict=error_dict,
                             initial_density=density.copy(),
                             volume_history = [self.real_proj.vol])
        return state
    def update_errors(self,state):
        real = self.error_routines["real"][0](state.density,state.intermediate_density)
        reciprocal = self.error_routines["reciprocal"][0](state.ft_density,
                                                       state.ft_density_history[-2],
                                                       state.intensity_harmonic_coefficients)
        errs = state.error_dict
        for key,val in real.items():
            errs["real"][key].append(val)
        for key,val in reciprocal.items():
            errs["reciprocal"][key].append(val)
        main_error = self.main_error_routine(errs)
        errs['main'].append(main_error)
        state.error_dict = errs
        
    def do_phasing(self):
        state = self.create_initial_state()
        lopt = self.opt.main_loop
        update_errors = self.update_errors
        
        for loop_name,(loop,n) in self.phasing_sketch.items():
            for i in range(n):
                for method_name,(step,m) in loop.items():
                    for j in range(m):
                        # Run optimization step
                        state = step(state)
                    
                        # update_some parametrs
                        state.iteration += 1
                        state.volume_history.append(self.real_proj.vol)
                        update_errors(state)

                # Print output 
                xprint('P{}:  Loop:{} Part:{} Method:{} Main Error: {} \n volume = {}, max_density={}'.format(Multiprocessing.get_process_name(),
                                                                                                      state.iteration+1,
                                                                                                      loop_name,
                                                                                                      method_name,
                                                                                                      state.error_dict['main'][-1],
                                                                                                      state.volume_history[-1],
                                                                                                      np.max(state.density)))
        return state
   
def snake_to_camel_simple(name: str) -> str:
    parts = name.split("_")
    return "".join(part.capitalize() for part in parts)
def create_step_instances(kwargs):
    current_module = sys.modules[__name__]

    instances = {}
    for _, cls in inspect.getmembers(current_module, inspect.isclass):
        if (
            cls.__module__ == __name__      # only classes defined in this module
            and issubclass(cls, StepBase)   # subclass of StepBase
            and cls is not StepBase         # exclude base class itself
        ):
            instances[camel_to_snake_simple(cls.__name__)]=cls(kwargs)

    return instances

def load_fourier_transform_weights(struct:HankelWeightStruct=HankelWeightStruct(),allow_weight_saving = False):
    db = database.project
    log.info(f'ft name postfix = {struct.weight_name}')
    try:
        weights_dict = db.load('ft_weights',path_modifiers={'name':struct.weight_name})
    except FileNotFoundError as e:
        weights_dict = HankelTransformWeights.get_weights_dict(struct)
        if allow_weight_saving:
            db.save('ft_weights',weights_dict,path_modifiers={'name':struct.weight_name})                    
    return weights_dict
        
@dataclass
class PhasingState:
    initial_density:NDArray = None
    density_history:list = None
    ft_density_history:list = None
    intermediate_density:NDArray= None # After inv constraint before real constraint
    intensity_harmonic_coefficients:NDArray = None # harmonic coefficients of |ft_density|^2
    iteration:int = 0
    error_dict: dict = None
    volume_history:list = None
    custom_outputs:dict = None

    @property
    def density(self):
        return self.density_history[-1]
    @density.setter
    def density(self,value):
        self.density_history.pop(0)
        self.density_history.append(value)
        self._density = self.density_history[-1]
        
    @property
    def ft_density(self):
        return self.ft_density_history[-1]
    @ft_density.setter
    def ft_density(self,value):
        self.ft_density_history.pop(0)
        self.ft_density_history.append(value)
        self._ft_density = self.ft_density_history[-1]

    
class StepBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self,state:PhasingState)->PhasingState:
        pass

    
class DuglasRachford(StepBase):
    def __init__(self,operators,opt):
        self.operators = operators
        self.beta = opt.beta
        self.ft=operators["fourier_transform"]
        self.ht = operators["harmonic_transform"]
        self.real_proj  = operators["real_projection"]
        self.inv_proj  = operators["invariant_projection"]
        self.tmp_intensity = np.zeros(self.ft.reciprocal_grid.shape[:-1],complex)
        self.ft_stab = opt.ft_stab
        
    def p1(self,density):
        "apply invariant constarint"
        ft,ht,inv_proj = self.ft,self.ht,self.inv_proj
    
        ft_d = ft.forward_cmplx(density)
        np.multiply(ft_d,ft_d.conj(),out = self.tmp_intensity)
        coeff = ht.forward_cmplx(self.tmp_intensity)        
        unknown_U = inv_proj.approximate_unknowns(coeff)
        new_coeff = inv_proj.mtip_projection(coeff,unknown_U)
        new_I = ht.inverse_cmplx(new_coeff)
        
        old_I_from_coeff = ht.inverse_cmplx(coeff)
        #new_ft_d = inv_proj.project_to_modified_intensity(ft_d,
        #                                                  self.tmp_intensity,
        #                                                  old_I_from_coeff,
        #                                                  new_I)
        new_ft_d = np.where(self.tmp_intensity!=0,ft_d/np.sqrt(self.tmp_intensity)*np.sqrt(new_I),np.sqrt(new_I))
        new_density = ft.inverse_cmplx(new_ft_d)
        if self.ft_stab:
            ft_error = density-ft.inverse_cmplx(ft_d)
            new_density += ft_error
        return new_density,new_ft_d,new_coeff
        
    def __call__(self,state:PhasingState)->PhasingState:
        d = state.density
        
        d1,ft_d1,new_coeff = self.p1(d)
        state.intermediate_density = d1
        state.intensity_harmonic_coefficients = new_coeff
        d2 = self.real_proj((self.beta+1)*d1-d)
        d2.imag = 0
        d2[d2<0]=0
        
        new_density =  d+d2-self.beta*d1

        state.density[...] = new_density
        state.ft_density[...] = ft_d1
        
        return state
    
class ErrorReduction(StepBase):
    def __init__(self,operators,opt):
        self.operators = operators
        self.ft=operators["fourier_transform"]
        self.ht = operators["harmonic_transform"]
        self.real_proj  = operators["real_projection"]
        self.inv_proj  = operators["invariant_projection"]
        self.tmp_intensity = np.zeros(self.ft.reciprocal_grid.shape[:-1],complex)
        self.ft_stab = opt.ft_stab
    def p1(self,density):
        "apply invariant constarint"
        ft,ht,inv_proj = self.ft,self.ht,self.inv_proj
        ft_d = ft.forward_cmplx(density)
        np.multiply(ft_d,ft_d.conj(),out = self.tmp_intensity)
        coeff = ht.forward_cmplx(self.tmp_intensity)        
        unknown_U = inv_proj.approximate_unknowns(coeff)
        new_coeff = inv_proj.mtip_projection(coeff,unknown_U)
        new_I = ht.inverse_cmplx(new_coeff)
        old_I_from_coeff = ht.inverse_cmplx(coeff)
        new_ft_d = inv_proj.project_to_modified_intensity(ft_d,
                                                          self.tmp_intensity,
                                                          old_I_from_coeff,
                                                          new_I)
        new_density= ft.inverse_cmplx(new_ft_d)
        if self.ft_stab:
            ft_error = density-ft.inverse_cmplx(ft_d)
            new_density += ft_error
        return new_density,new_ft_d,new_coeff
        
    def __call__(self,state:PhasingState)->PhasingState:
        d = state.density
        
        d1,ft_d1,new_coeff = self.p1(d)
        state.intermediate_density = d1
        state.intensity_harmonic_coefficients = new_coeff
        
        state.density = self.real_proj(d1)
        state.ft_density[...] = ft_d1
        return state

