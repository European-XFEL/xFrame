import time
import logging
import sys
import os
import numpy as np
import traceback
file_path = os.path.realpath(__file__)
plugin_dir = os.path.dirname(file_path)
os.chdir(plugin_dir)

from xframe.library import mathLibrary as mLib
from xframe.library.mathLibrary import SampleShapeFunctions,ExponentialRamp,LinearRamp
from xframe.library.gridLibrary import uniformGrid_func
from xframe.library.gridLibrary import ReGrider
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import GridFactory
from xframe.library.pythonLibrary import plugVariableIntoFunction
from xframe.library.pythonLibrary import DictNamespace
from xframe.library.pythonLibrary import RecipeFactory
from xframe.library.pythonLibrary import selectElementOfFunctionOutput_decorator
from xframe.library.pythonLibrary import xprint
from xframe.interfaces import ProjectWorkerInterface


from .projectLibrary.hankel_transforms import generate_ht
from .projectLibrary.hankel_transforms import generate_weightDict
from .projectLibrary.ft_grid_pairs import get_grid
from .projectLibrary.ft_grid_pairs import polarFTGridPair_PNAS
from .projectLibrary.ft_grid_pairs import radial_grid_func_zernike
from .projectLibrary.ft_grid_pairs import spherical_ft_grid_pair_zernike
from .projectLibrary.ft_grid_pairs import polar_ft_grid_pair_zernike
from .projectLibrary.ft_grid_pairs import polarFTGridPair_SinCos_new
from .projectLibrary.fourier_transforms import generate_ft,load_fourier_transform_weights
from .projectLibrary.misk import getAnalysisRecipeFacotry
from .projectLibrary.misk import get_analysis_process_factory
from .projectLibrary.misk import generate_calc_center
from .projectLibrary.misk import generate_absolute_value
from .projectLibrary.misk import generate_square
from .projectLibrary.misk import _get_reciprocity_coefficient
from .projectLibrary.misk import save_to_dict
from .projectLibrary.misk import copy as copy_operation
from .projectLibrary.misk import diff as diff_operation
from .projectLibrary.misk import add_above_zero_index as add_above_zero_index
from .projectLibrary.misk import load_from_dict
from .projectLibrary.fxs_Projections import ReciprocalProjection
from .projectLibrary.fxs_Projections import ShrinkWrapParts
from .projectLibrary.fxs_Projections import generate_negative_shift_operator,generate_shift_by_operator
from .projectLibrary.fxs_Projections import generate_initial_support_mask
from .projectLibrary.fxs_Projections import generate_orientation_matching
from .projectLibrary.fxs_Projections import generate_fix_point_inversion
from .projectLibrary.fxs_Projections import RealProjection
from .projectLibrary.fxs_invariant_tools import generate_estimate_number_of_particles_new,generate_estimate_number_of_particles_new_2
from .projectLibrary.fxs_IO_methods import generate_error_routines
from .projectLibrary.fxs_IO_methods import HIOProjection
from .projectLibrary.fxs_IO_methods import generate_main_error_routine
from .projectLibrary.fxs_IO_methods import error_reduction
from .projectLibrary.harmonic_transforms import HarmonicTransform
from .projectLibrary.fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_2d
from .projectLibrary.fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_3d
from .projectLibrary.fxs_invariant_tools import ccd_associated_legendre_matrices
from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe import settings
from xframe import database
from xframe import Multiprocessing
import xframe


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

def update_settings_and_db(new_opt):
    settings.project = new_opt
    db.update_settings(new_opt.dict())
    global opt
    opt = new_opt


class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        #log.info('analysis worker init')
        self.opt = settings.project
        self.process_factory=RecipeFactory({})
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
            #xprint('Generating phasing loop.',)
            self.setup_phasing_loop()
            #xprint('done.\n')
            #xprint(f'{self.mtip.loops_str}start phasing')
            result = self.start_phasing()
            #xprint('Finished phasing!\n')
            return result
        
        mp_mode = Multiprocessing.MPMode_Queue(assemble_outputs = False)        
        mp_opt = settings.project.multi_process
        xprint(f'Spawning phasing processes executing:\n{self.mtip.loops_str}')
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
                errors.append(result_dict['error_dict']['main'][-1])
                projection_matrices = result_dict.pop('projection_matrices')                
            r_ids=np.argsort(errors)
            log.info('error sorted reconstruction = {} \n errors ={}'.format(r_ids,np.array(errors)[r_ids]))
            processed_results_dict={str(_id):processed_results_list[_id] for _id in r_ids}
            reciprocity_coefficient = _get_reciprocity_coefficient(settings.project.fourier_transform)
            data_dict={'configuration':{'internal_grid':grid_pair,'xray_wavelength':self.mtip.load_mtip_data()[0]['xray_wavelength'],'reciprocity_coefficient':reciprocity_coefficient},'reconstruction_results':processed_results_dict,'projection_matrices':projection_matrices,'stats':stats}
            save('reconstructions',data_dict)
        except Exception as e:
            log.info(f'Error during postprocessing / saving with message:\n {e}')
            log.debug(traceback.format_exc())
            
    
    def run(self):
        #log.info(self)
        #log.info(self.mtip_opt)
        opt = settings.project
        start_time = time.time()
        #self.MTIP.preinit()
        if opt.multi_process.use:
            result=self.start_parallel_phasing()
        else:
            xprint('Generating phasing loop.')
            self.setup_phasing_loop()
            xprint('done.\n')
            xprint(f'Start phasing:\n{self.mtip.loops_str} ')
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

    loops_str = 'Loops:\n'
    loops_opt = settings.project.main_loop.sub_loops
    max_loop_name_size = max([len(name)for name in loops_opt.order])
    for loop_name in loops_opt.order:            
        loop_opt = loops_opt.get(loop_name,{})
        iterations = loop_opt.get('iterations','')
        methods_string = ''
        for method in loop_opt.get('order',[]):
            method_opt = loop_opt.methods.get(method,{})
            if isinstance(method_opt,(dict,DictNamespace)):
                method_iteration = method_opt['iterations']
            else:
                method_iteration = method_opt
            methods_string += f'{method_iteration}x{method} '
        loops_str += f'\t{loop_name}:'+' '*(max_loop_name_size-len(loop_name))+ f'\t {iterations}x( '+''.join(methods_string)+')\n'
   
    @classmethod
    def preinit(cls):
        ''' 
        This Method prepares needed quantities which require multiprocessing that are equal for all MTIP instances before __init__ is called in a multi process environment. This is currently done only to preload the fourier transform weights.
        '''
        opt = settings.project
        cls.dimensions = opt.dimensions
        tmp = cls.load_mtip_data()
        cls.mtip_data = tmp[0]
        cls.data_q_limits = tmp[1]
        cls.data_number_of_radial_points = tmp[2]

        # create fourier transform weights
        ht_opt={
            'dimensions':opt.dimensions,
            **opt.grid,            
        }
        max_q = opt.grid.max_q
        if not isinstance(max_q,float):
            max_q = cls.data_q_limits[1]
        cls.max_q=max_q
        # generate mock angular grid parameters to use the standard grid routine to get the internal radial grid.
        if opt.dimensions ==2:
            grid_param = {'phis':np.array([1.0,2.0])}
        elif opt.dimensions ==3:
            grid_param = {'phis':np.array([1.0,2.0]),'thetas':np.array([1.0,2.0])}
        mock_grid_pair = get_grid({**opt.fourier_transform,**ht_opt,**grid_param,'max_q':max_q,'n_radial_points_from_data':cls.data_number_of_radial_points})        
        qs = mock_grid_pair.reciprocalGrid.__getitem__((slice(None),)+(0,)*cls.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        cls.reciprocal_radial_points = qs
        rs = mock_grid_pair.realGrid.__getitem__((slice(None),)+(0,)*cls.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        cls.real_radial_points = rs        
        n_radial_points = len(qs)
        r_max = np.max(rs)
        #cls.fourier_transform_weights = cls.load_fourier_transform_weights(n_radial_points,r_max)
        cls.fourier_transform_weights = load_fourier_transform_weights(opt.dimensions,opt.fourier_transform,opt.grid,database.project)
        cls.preinit_was_called = True
    @classmethod
    def load_mtip_data(cls):
        opt = settings.project
        db = database.project
        data = db.load('invariants',path_modifiers={'structure_name':opt.structure_name,'dimensions':opt.dimensions})        
        data_q_limits = [data['data_radial_points'].min(),data['data_radial_points'].max()]
        data_number_of_radial_points = len(data['data_radial_points'])
        #log.info('data points = {}'.format(data['data_radial_points']))
        return [data,data_q_limits,data_number_of_radial_points]
    

    def __init__(self,process_factory):
        set_globals()
        #self.opt = settings.project
        self.process_factory=process_factory
        self.results={}
        
        # will be set by generate_phasing_loop
        self.grid_pair = False
        self.routines = False
        self.projection_objects=False
        self.mtip_part_names = False
        self.mtip_io_names = False
        self.phasing_loop = False        




    def assemble_operators(self):
        #order is important, e.g. assemble_error_routines depents on self.projection_objects
        transforms,grid_pair,max_order=self.assemble_transform_op_and_grid()        
        reciprocal_projection,rp_obj=self.assemble_reciprocal_projection_op(grid_pair,max_order)
        real_projections,hio_projection,real_projection_obj,shrink_wrap_obj=self.assemble_real_projection_op(grid_pair,rp_obj,transforms)
        self.projection_objects = {'reciprocal':rp_obj,'hio':hio_projection,'real':real_projection_obj,'sw':shrink_wrap_obj}
        orientation_fixing=self.assemble_orientation_fixing_op(grid_pair)
        errors = self.assemble_error_routines(grid_pair)
        misk = self.assemble_misk_oppesators(grid_pair)


        #log.info(f'{transforms}\n {reciprocal_projection}\n{real_projections}\n{orientation_fixing}\n{errors}\n {misk}')
        operators={**transforms,**real_projections,**reciprocal_projection,**orientation_fixing,**errors,**misk}
        return operators,grid_pair
    
    def generate_fourier_transforms(self,harm_trf):
        dimensions = self.dimensions
        ft_opt = opt.fourier_transform
        ft_type = ft_opt['type']
        pos_orders= np.arange(harm_trf.max_order+1)
        pi_in_q = ft_opt.get('pi_in_q',None)

        reciprocity_coefficient=_get_reciprocity_coefficient(ft_opt)
        
        use_gpu = opt['GPU']['use']                    
        weights_dict = self.fourier_transform_weights
        r_max = np.max(self.real_radial_points)
        #log.info('Use GPU = {}'.format(use_gpu))
        fourierTransform,inverseFourierTransform=generate_ft(r_max,weights_dict,harm_trf,dimensions,pos_orders=pos_orders,reciprocity_coefficient=reciprocity_coefficient,use_gpu = use_gpu,mode = ft_type)
        
        #d= np.ones((settings.project.grid.n_radial_points,pos_orders.max()*2+1))
        #d2 = inverseFourierTransform(fourierTransform(d))
        #log.info(d2[:,0])
        return fourierTransform,inverseFourierTransform

    def assemble_transform_op_and_grid(self):
        ht_opt={
            'dimensions':opt.dimensions,
            **opt.grid            
        }
        #log.info(f'ht opt = {ht_opt}')
        # Create harmonic transforms (needs to happen before grid selection since harmonic transform plugin can choose the angular part of the grid.)
        cht=HarmonicTransform('complex',ht_opt)
        #ht_data_type = opt['harmonic_transform']['data_type']
        if self.dimensions == 2:
            ht =  HarmonicTransform('real',ht_opt)
        elif self.dimensions == 3:
            ht=cht
        cht_forward,cht_inverse = cht.forward, cht.inverse
        #log.info(cht_forward)
        #log.info(f'spat shape = {cht._sh._sh.spat_shape} ht_opt = {ht_opt}')
        ht_forward,ht_inverse = ht.forward, ht.inverse
        #log.info(ht_forward)
        #log.info("ht grid param = {}".format(cht.grid_param))
        #log.info(f'spat shape = {cht._sh._sh.spat_shape}')
        grid_pair=get_grid({**opt.fourier_transform,**ht_opt,**cht.grid_param,'max_q':self.max_q,'n_radial_points_from_data':self.data_number_of_radial_points})
        # fourier transforms
        ft_forward,ft_inverse=self.generate_fourier_transforms(cht)
        #log.info(f'grid shape = {grid_pair.realGrid[:].shape}')
        #log.info("test ft:")
        #a = np.ones(grid_pair.realGrid.shape)
        #a[20:]=0
        #for i in range(20):
        #    a=ft_inverse(ft_forward(a))
        #log.info('mean at 10 = {} mian at 40 = {}'.format(np.mean(a[10]),np.mean(a[40])))
        #raise Exception()
        
        transform_op={'fourier_transform':ft_forward,'inverse_fourier_transform':ft_inverse,'harmonic_transform':ht_forward,'inverse_harmonic_transform':ht_inverse,'complex_harmonic_transform':cht_forward,'complex_inverse_harmonic_transform':cht_inverse}
        return transform_op,grid_pair,ht.max_order

    def assemble_reciprocal_projection_op(self,grid_pair,max_order):
        #opt=opt['projections']['reciprocal']
        rp=ReciprocalProjection(grid_pair.reciprocalGrid,self.mtip_data,max_order)
        self.rprojection = rp
        self.results['n_particles']=[]
        self.results['n_particles_fraction']=[]
        self.results['n_particles_gradients']=[]        
        def save_number_of_particles():
            n_dict = rp.number_of_particles_dict
            #log.info(n_dict['number_of_particles'])
            self.results['n_particles'].append(n_dict['number_of_particles'])
            neg_fractions = n_dict['negative_fraction']
            gradient = n_dict['gradient']
            if isinstance(neg_fractions,np.ndarray):
                self.results['n_particles_fraction'].append(neg_fractions)
            if isinstance(gradient,np.ndarray):
                self.results['n_particles_gradients'].append(gradient)
                
        reciprocal_projection_op={'mtip_projection':rp.mtip_projection,'approximate_unknowns':rp.approximate_unknowns,'project_to_modified_intensity':rp.project_to_modified_intensity,'project_to_fixed_intensity':rp.project_to_fixed_intensity,'save_number_of_particles':save_number_of_particles}
        return reciprocal_projection_op,rp
    
    def assemble_real_projection_op(self,grid_pair,rp_obj,transforms):
        #log.info('assemble real projection')
        r_opt=opt.projections.real
        r_grid=grid_pair.realGrid
        q_grid=grid_pair.reciprocalGrid

        # create autocorrelation
        pr = rp_obj.full_projection_matrices
        icht = transforms['inverse_harmonic_transform']
        ift = transforms['inverse_fourier_transform']
        #log.info(f'len pr = {len(rp_obj.used_orders)}')
        #log.info(f'pr shapes = {[p.shape for p in pr]}')
        #log.info(f'len pr = {len(pr)}')

        if self.dimensions ==3:
            auto_correlation_guess = ift(icht(pr)).real
        elif self.dimensions ==2:
            #log.info(f'pr.shape = {pr.shape}')
            auto_correlation_guess = ift(icht(pr.T)).real
        
        metadata={'integrated_intensity':rp_obj.integrated_intensity,'real_grid':r_grid,'auto_correlation':auto_correlation_guess}

        real_projection_obj = RealProjection(r_opt.projections,metadata)

        sw_opt=r_opt['shrink_wrap']
        sw_obj = ShrinkWrapParts(r_grid,q_grid,real_projection_obj.initial_support)
        #real_support_projection=generate_real_support_projection(initial_support_mask)
        
        #other_real_projections=generate_other_real_projections(r_opt.non_support_projections,metadata)
    
        
        multiply_ft_gaussian=sw_obj.multiply_with_ft_gaussian  #generateSW_multiply_ft_gaussian(sw_opt['sigma'],q_grid)
        calculate_support_mask=sw_obj.get_new_mask
        #hio = generate_HIO(r_opt['HIO'])
        hio_opt = r_opt.HIO
        hio = HIOProjection(hio_opt.beta[0],considered_projections=hio_opt.get('considered_projections',['all']))
        #hio = generate_HIO({'beta':self.beta})
        
        #real_projection_op={'real_support_projection':real_support_projection,'other_real_projections':other_real_projections,'multiply_ft_gaussian':multiply_ft_gaussian,'calculate_support_mask':calculate_support_mask,'hybrid_input_output':hio.projection,'real_projection':real_projection_obj.projection}
        
        real_projection_op={'multiply_ft_gaussian':multiply_ft_gaussian,'calculate_support_mask':calculate_support_mask,'hybrid_input_output':hio.projection,'error_reduction':error_reduction,'real_projection':real_projection_obj.projection}
        
        return real_projection_op,hio,real_projection_obj,sw_obj
    
    def assemble_orientation_fixing_op(self,grid_pair):
        ft_opt=opt['fourier_transform']
        r_grid=grid_pair.realGrid
        q_grid=grid_pair.reciprocalGrid
        calc_center_routine=generate_calc_center(r_grid)
        negative_shift_routine = generate_shift_by_operator(grid_pair.reciprocalGrid,opposite_direction = True)
        #negative_shift_routine=generate_negative_shift_operator(q_grid,ft_opt['type'])
        #negative_shift_routine=generate_negative_shift_operator(q_grid,ft_opt['type'])
        orientation_fixing_op={'calc_center':calc_center_routine,'negative_shift':negative_shift_routine}
        if opt.dimensions==2:
            if opt.output_density_modifiers.fix_orientation:
                remaining_SO_projection=self.projection_objects['reciprocal'].remaining_SO_projection
                orientation_fixing_op['fix_remaining_SO_freedom']=remaining_SO_projection
        return orientation_fixing_op

    def assemble_error_routines(self,grid_pair):
        error_opt = opt.main_loop.error
        rp = self.projection_objects['reciprocal']
        deg2_invariants = rp.deg2_invariants
        number_of_particles = rp.number_of_particles
        used_orders = rp.used_orders
        xray_wavelength = self.mtip_data['xray_wavelength']
        invariant_mask = rp.radial_mask[:,:,None] * rp.radial_mask[:,None,:]
        initial_mask = self.projection_objects['real'].initial_support
        error_routines = generate_error_routines(error_opt,grid_pair,deg2_invariants = deg2_invariants,projection_matrices=rp.projection_matrices,used_orders = used_orders,n_particles = rp.number_of_particles,invariant_mask = invariant_mask,xray_wavelength = xray_wavelength,initial_mask = initial_mask)
        #log.info('error_routines {}'.format(error_routines))
        return {'real_errors':error_routines['real'],'reciprocal_errors':error_routines['reciprocal']}
        
    def assemble_misk_oppesators(self,grid_pair):
        dtype = np.dtype(complex)
        abs_value = generate_absolute_value(grid_pair.realGrid[:].shape[:-1],dtype,cache_aware = settings.general.cache_aware,L2_cache = settings.general.L2_cache)        
        square = generate_square(grid_pair.realGrid[:].shape[:-1],dtype,cache_aware = settings.general.cache_aware,L2_cache = settings.general.L2_cache)
        if self.dimensions == 3:
            calc_deg2_invariant = harmonic_coeff_to_deg2_invariants_3d
        elif self.dimensions == 2:
            calc_deg2_invariant = harmonic_coeff_to_deg2_invariants_2d
        return {'square_grid':square,'abs_value':abs_value,'calc_deg2_invariant':calc_deg2_invariant,'save_to_dict':save_to_dict,'copy':copy_operation,'add_above_zero_index':add_above_zero_index,'diff':diff_operation,'load_from_dict':load_from_dict}
        
    
    def assemble_MTIP_routines(self):
        error_saving_parts = self.assemble_error_saving_parts()
        self.process_factory.addOperators(error_saving_parts)
        mtip_parts=self.assemble_mtip_parts()
        self.process_factory.addOperators(mtip_parts)
        io_variants=self.assemble_IO_variants()
        output_modifier=self.assemble_output_modifier(opt['output_density_modifiers'])
        calc_deg2_invariant = self.assemble_calc_deg2_invariant()
        #log.info(io_variants['SW'])
        return {'calc_deg2_invariant':calc_deg2_invariant,'output_modifier':output_modifier,**io_variants}

    def assemble_error_saving_parts(self):
        error_parts = {}
        process_factory = self.process_factory
        for name in opt.main_loop.error.methods:
            if name!='main':
                error_sketch = [name+'_errors',
                                [(0,),[('save_to_dict',(self.results,['errors',name],'iterative_append'))]]
                                ]
                try:
                    error_parts['calc_' + name + '_errors'] = process_factory.buildProcessFromSketch(error_sketch)
                except KeyError as e:
                    error_sketch = [(0,),['id']]
                    error_parts['calc_' + name +'_errors'] = process_factory.buildProcessFromSketch(error_sketch)
        
        return error_parts
            
    def assemble_mtip_parts(self):
        process_factory=self.process_factory
        routine_sketches = {}
        routine_sketches['MTIP_start']=[
            [(0,0),['copy','square_grid']],                           
            [(0,1,1),['id','harmonic_transform','copy']],
            [(0,1,1,2),['id','id','approximate_unknowns','id']],
            [(0,1,2,3),[('id',()),('id',()),('save_to_dict',(self.results,'fxs_unknowns','replace')),'id']],
            [(0,1,2,1,3),['id','mtip_projection','id','id']],
            [(0,1,2,3),['id','inverse_harmonic_transform','id','id']],
            [(0,0,3,1,2),['id','project_to_modified_intensity','save_number_of_particles','id']],
            [(0,1,2,1),['calc_reciprocal_errors','id']],
            [(1,),['id']]
        ]

        routine_sketches['MTIP_start_non_FXS']=[
            [(0,0),['copy','square_grid']],                           
            [(0,0,1,1),['id','project_to_fixed_intensity','save_number_of_particles','id']],
            [(0,1,2,1),['calc_reciprocal_errors','id']],
            [(1,),['id']]
        ]             

        #routine_sketches['MTIP_start_estimate_n_particles']=[
        #    [(0,0),['copy','square_grid']],                           
        #    [(0,1),['id','harmonic_transform']],
        #    [(0,1,1),['id','id','approximate_unknowns']],
        #    [(0,1,2),[('id',()),('id',()),('save_to_dict',(self.results,'fxs_unknowns','replace'))]],
        #    [(0,1,2,1),['id','mtip_projection','id']],
        #    [(0,1,2),['id','inverse_harmonic_transform','id']],
        #    [(0,1,2),['id','particle_number_projection','id']],
        #    [(0,0,1,2),['id','project_to_modified_intensity','save_number_of_particles','id']],
        #    [(0,1,2,1),['calc_reciprocal_errors','id']],
        #    [(1,),['id']]
        #]
        
        routine_sketches['MTIP_end']=['id',
                         ['id','copy'],
                         ['fourier_transform','id']
                        ]
        routines = {key:process_factory.buildProcessFromSketch(sketch) for key,sketch in routine_sketches.items()}
        self.mtip_part_names = tuple(routines.keys())
        return routines
        
    def assemble_IO_variants(self):
        estimate_number_of_particles = settings.project.projections.reciprocal.number_of_particles.get('estimate',False)
        estimate_in_variants = settings.project.projections.reciprocal.number_of_particles.get('estimate_in',[])
        #use_ht_difference = settings.project.main_loop.get('stabilize_harmonic_transform',False)       
        #log.info('using ft difference = {}'.format(use_ft_difference))
        #log.info('using ht difference = {}'.format(use_ht_difference))
        start_routine_name_initial='MTIP_start'
        routine_names = ('HIO','ER','HIO_non_FXS','ER_non_FXS')
        io_names={'HIO':'hybrid_input_output','ER':'error_reduction','HIO_non_FXS':'hybrid_input_output','ER_non_FXS':'error_reduction'}      
            
        routine_sketches={}
        for name in routine_names:
            start_routine_name=start_routine_name_initial
            if '_non_FXS' in name:
                start_routine_name+='_non_FXS'
            elif (name in estimate_in_variants) and estimate_number_of_particles:
                start_routine_name+='_estimate_n_particles'

            sketch = [
                [(1,1),['fourier_transform','id']],
                [np.array([0,0,1],dtype=int),[start_routine_name,'id']],
                [np.array([0,1,0],dtype=int),['inverse_fourier_transform','id','id']],
                [np.array([0,0,1,2],dtype=int),['copy','real_projection','id','id']],
                [(0,1,2,0,1,3),[io_names[name],'calc_real_errors','id']],
                [(2,0),['id','id']]
            ]
            sketch_ft_stab = [
                    [(1,1),['fourier_transform','id']],
                    [(0,0,0,1),[start_routine_name,'inverse_fourier_transform','id']],
                    [(0,2,1,2,0),['inverse_fourier_transform','diff','id','id']],
                    [(0,1,2,3),['add_above_zero_index','id','id']],
                    [(0,0,1,2),['copy','real_projection','id','id']],
                    [(0,1,2,0,1,3),[io_names[name],'calc_real_errors','id']],
                    [(2,0),['id','id']]
                    #[(0,),['MTIP_end']]                    
                ]
            
            routine_sketches[name]=sketch
            routine_sketches[name+'_ft_stab']=sketch_ft_stab

        routine_sketches['SW']=[
            'copy',
            ['abs_value','copy'],
            [(0,1),['fourier_transform','id']],
            [(0,1),['multiply_ft_gaussian','id']],
            [(0,1),['inverse_fourier_transform','id']],
            [(0,1),['calculate_support_mask']]
        ]
        routine_sketches['SW_center']=[
            [(0,0,0),['copy','fourier_transform','copy']],
            [(0,0,1,2),['abs_value','copy','id','id']],
            [(0,1,2,3),['fourier_transform','id','id','id']],
            [(0,1,2,3),['multiply_ft_gaussian','id','id','id']],
            [(0,1,2,3),['inverse_fourier_transform','id','id','id']],
            [(0,1,2,3),['calculate_support_mask','id','id']]
        ]
        self.mtip_io_names = tuple(routine_sketches.keys())
        #log.info(f'routine IO names = {self.mtip_io_names}')
        process_factory=self.process_factory
        routines = {key:process_factory.buildProcessFromSketch(sketch) for key,sketch in routine_sketches.items()}
        #log.info(f'routine_names = {routines.keys()}')
        return routines

    def assemble_mtip_parts_old(self):
        process_factory=self.process_factory

        #standard working phasing part
        mtip_start_sketch=['id',
                           ['copy','id'],                           
                           [(0,1),['id','square_grid']],                           
                           [(0,1),['id','harmonic_transform']],
                           [(0,1,1),['id','id','approximate_unknowns']],
                           [(0,1,2),[('id',()),('id',()),('save_to_dict',(self.results,'fxs_unknowns','replace'))]],
                           [(0,1,2,1),['id','mtip_projection','id']],
                           [(0,1,2),['id','inverse_harmonic_transform','id']],
                           [(0,0,1,2),['id','project_to_modified_intensity','save_number_of_particles','id']],
                           [(0,1,2,1),['calc_reciprocal_errors','id']],
                           [(1,),['inverse_fourier_transform']]
                           ]
        
        #mtip_start_estimate_n_particles_sketch=['id',
        #                                        ['copy','id'],                           
        #                                        [(0,1),['id','square_grid']],                           
        #                                        [(0,1),['id','harmonic_transform']],
        #                                        [(0,1,1),['id','id','approximate_unknowns']],
        #                                        [(0,1,2),[('id',()),('id',()),('save_to_dict',(self.results,'fxs_unknowns','replace'))]],
        #                                        [(0,1,2,1),['id','mtip_projection','id']],
        #                                        [(0,1,2),['id','inverse_harmonic_transform','id']],
        #                                        [(0,1,2),['id','particle_number_projection','id']],
        #                                        [(0,0,1,2),['id','project_to_modified_intensity','save_number_of_particles','id']],
        #                                        [(0,1,2,1),['calc_reciprocal_errors','id']],
        #                                        [(1,),['inverse_fourier_transform']]
        #                                        ]
        
        mtip_end_sketch=['id',
                         ['id','copy'],
                         ['fourier_transform','id']
                         ]

        mtip_start=process_factory.buildProcessFromSketch(mtip_start_sketch)
        mtip_start_estimate_n_particles = process_factory.buildProcessFromSketch(mtip_start_estimate_n_particles_sketch)
        mtip_end=process_factory.buildProcessFromSketch(mtip_end_sketch)        
        #mtip_start=selectElementOfFunctionOutput_decorator(mtip_start.run,0)
        
        return {'MTIP_start':mtip_start,'MTIP_end':mtip_end.run,'MTIP_start_estimate_n_particles':mtip_start_estimate_n_particles}
        
    def assemble_IO_variants_old(self):
        estimate_number_of_particles = settings.project.projections.reciprocal.number_of_particles.estimate
        estimate_in_variants = settings.project.projections.reciprocal.number_of_particles.estimate_in
        if estimate_number_of_particles and ('HIO' in estimate_in_variants):
            fxs_HIO_sketch=[
                [np.array([0,1],dtype=int),['MTIP_start_estimate_n_particles','id']],
                [np.array([0,0,1]),['copy','id','id']],       
                [np.array([0,1,2],dtype=int),['id','real_support_projection','id']],
                [np.array([0,1,2],dtype=int),['id','other_real_projections','id']],
                [(0,1,2,0,1),['hybrid_input_output','calc_real_errors']],
                [(0,),['MTIP_end']]
            ]
        else:
            fxs_HIO_sketch=[
                [np.array([0,1],dtype=int),['MTIP_start','id']],
                [np.array([0,0,1]),['copy','id','id']],       
                [np.array([0,1,2],dtype=int),['id','real_support_projection','id']],
                [np.array([0,1,2],dtype=int),['id','other_real_projections','id']],
                [(0,1,2,0,1),['hybrid_input_output','calc_real_errors']],
                [(0,),['MTIP_end']]
            ]


        if estimate_number_of_particles and ('ER' in estimate_in_variants):
            fxs_ER_sketch=[
                [(0,1),['MTIP_start_estimate_n_particles','id']],
                [(0,0),['copy','id']],
                [(0,1),['id','real_support_projection']],
                [(0,1),['id','other_real_projections']],
                [(0,1,0,1),['error_reduction','calc_real_errors']],
                [(0,),['MTIP_end']]
            ]
        else:
            fxs_ER_sketch=[
                [(0,1),['MTIP_start','id']],
                [(0,0),['copy','id']],
                [(0,1),['id','real_support_projection']],
                [(0,1),['id','other_real_projections']],
                [(0,1,0,1),['error_reduction','calc_real_errors']],
                [(0,),['MTIP_end']]
            ]
            
        SW_sketch=[
            'copy',
            ['abs_value','copy'],
            [(0,1),['fourier_transform','id']],
            [(0,1),['multiply_ft_gaussian','id']],
            [(0,1),['inverse_fourier_transform','id']],
            [(0,1),['calculate_support_mask']]
        ]

        process_factory=self.process_factory
        fxs_HIO=process_factory.buildProcessFromSketch(fxs_HIO_sketch)
        fxs_ER=process_factory.buildProcessFromSketch(fxs_ER_sketch)
        fxs_SW=process_factory.buildProcessFromSketch(SW_sketch)
        return {'HIO':fxs_HIO,'ER':fxs_ER,'SW':fxs_SW}

    def assemble_output_modifier(self,opt):

        output_modifier_sketch=[
            [(0,1,1),['id','fourier_transform','id']],                        
        ]
        _id = [['id','id']]

        shift_center = [
            [(0,1,1),['copy','fourier_transform','calc_center']],
            [(0,1,2),[['id',np.array([],dtype=object)],['id',np.array([],dtype=object)],['save_to_dict',np.array([self.results,'neg_center_pos','replace'],dtype=object)]]],
            [(0,2,1,2),['negative_shift','negative_shift']],
            [(0,1),['id','inverse_fourier_transform']]
            #[(0,),['id']]
        ]
        
        fix_orientation=[
            [(0,1),['complex_harmonic_transform','complex_harmonic_transform']],
            [(0,1),[['id',np.array([],dtype=object)],['id',np.array([],dtype=object)],['load_from_dict',np.array([self.results,'fxs_unknowns'],dtype=object)]]],
            [(0,2,1,2),['fix_remaining_SO_freedom','fix_remaining_SO_freedom']],
            [(0,1),['complex_inverse_harmonic_transform','complex_inverse_harmonic_transform']]
        ]


        output_modifier_sketch=_id
        if opt['shift_to_center']:
            output_modifier_sketch = shift_center
        if opt.get('fix_orientation',False):
            try:
                assert settings.project.projections.reciprocal.SO_freedom.use,'SO Freedom is not used in reciprocal projection (projections.reciprocal.SO_freedom.use = False) therfore no orientation fixing is possible. skipping its assembly.'
                assert self.dimensions == 2,'3D version of MTIP does not allow for orientation fixing (only in 2D this is possible). Skipping orientation fixing assembly.'
                output_modifier_sketch = shift_center+fix_orientation
            except AssertionError as e:
                log.info(e)
        output_modifier = self.process_factory.buildProcessFromSketch(output_modifier_sketch)        
        return output_modifier

    def assemble_calc_deg2_invariant(self):
        calc_deg2_invariant_sketch = [
            ['fourier_transform'],
            ['square_grid'],
            ['harmonic_transform'],
            ['calc_deg2_invariant']
        ]
        calc_deg2_invariant = self.process_factory.buildProcessFromSketch(calc_deg2_invariant_sketch)
        return calc_deg2_invariant


    def assemble_phasing_loop(self):
        #### get relevant options ####
        sw_opt = opt.projections.real.shrink_wrap
        supp_opt = opt.projections.real.projections.support
        hio_opt = opt.projections.real.HIO
        enforce_initial_support_opt = opt.projections.real.projections.support.enforce_initial_support
        loop_opt=opt['main_loop']

        #### get routines ####
        routines=self.routines
        fourier_transform=self.process_factory.operatorDict['fourier_transform']
        inverse_fourier_transform=self.process_factory.operatorDict['inverse_fourier_transform']
        inverse_harmonic_transform=self.process_factory.operatorDict['inverse_harmonic_transform']
        prm = self.projection_objects['reciprocal'].projection_matrices
        #low_res_coeff = self.projection_objects['reciprocal'].projection_matrices
        #tmp_coeff = [np.zeros(p.shape) for p in prm]
        #for i,Ilm in enumerate(low_res_coeff):
        #    tmp_coeff[i]=Ilm
        #I = inverse_harmonic_transform(tmp_coeff)
        #I = I.real
        #I[I<0]=0
        #rho_guess = inverse_fourier_transform(I.astype(complex))
        #real_grid=self.grid_pair.realGrid
        #reciprocal_grid=self.grid_pair.reciprocalGrid
        #db.save("/gpfs/exfel/theory_group/user/berberic/MTIP/test/inital_density/d1_low_res.vts",[np.abs(rho_guess),rho_guess.real],grid_type='spherical',grid = real_grid[:])
        #db.save("/gpfs/exfel/theory_group/user/berberic/MTIP/test/inital_density/Intensity_guess.vts",[np.abs(I),rho_guess.real],grid_type='spherical',grid = reciprocal_grid[:])
        real_grid=self.grid_pair.realGrid
        real_density_guess_method=self.generate_density_guess_method(opt['density_guess'],real_grid)
        
        #### generate main error routines ####
        main_error_names = opt.main_loop.error.methods.main.metrics
        main_error_routine = generate_main_error_routine(main_error_names,opt.main_loop.error.methods.main.type)
        #check_error_target,check_error_gain_limit = self.generate_check_error_routines()
        init_error_dict = self.init_error_dict
        update_shrink_wrap = self.generate_update_shrink_wrap(self.projection_objects['sw'])
        #### assemble loops ####
        def read_method_settings(method_name,method_opt):
            if isinstance(method_opt,(dict,DictNamespace)):
                iterations = method_opt.get('iterations',0)
                out_dict = {'process':routines[method_name],'iterations':iterations,'options':method_opt}
                    
            else:
                out_dict = {'process':routines[method_name],'iterations':method_opt,'options':{}}
            return out_dict
            
            
        def generate_loop_method(loop_name,loop_opt,loop_number):
            max_iterations = loop_opt.iterations
            order = loop_opt.order
                
            methods = {key:read_method_settings(key,loop_opt['methods'][key]) for key in order}
            # HIO parameter #
            n_hio_betas = len(hio_opt.beta)
            if n_hio_betas-1 < loop_number:
                hio_beta = [0.5,0.5,-1/700,1600]
                log.info(f'No HIO beta specified in settings for loop with id {loop_number} in {loop_opt.order}. Setting beta = {hio_beta}')
            else:
                hio_beta = hio_opt.beta[loop_number]                
            hio_beta_ramp = ExponentialRamp(*hio_beta)
    
            #log.info(f'{loop_name} : threshold = {sw_threshold} from thresholds = {sw_opt.thresholds}')
            # support #
            enforce_initial_support_opt = supp_opt.enforce_initial_support
            if enforce_initial_support_opt.apply:
                enforce_initial_support_error_limit=[enforce_initial_support_opt['if_error_bigger_than']]
            else:
                enforce_initial_support_error_limit=[np.inf]

            def change_to_ft_stab(process_opt,process_name,enforce_initial_support_list):
                apply_ft_stabilization = False
                process_not_ft_stabilized = (process_name[-8:]!='_ft_stab')
                if process_not_ft_stabilized:
                    if 'ft_stab' in (process_opt):
                        if isinstance(process_opt.ft_stab,bool):
                            apply_ft_stabilization = process_opt.ft_stab
                        elif process_opt.ft_stab == 'link_to_enforce_initial_support':
                            delay = max(int(process_opt.link_to_enforce_initial_support.delay),1)
                            if len(enforce_initial_support_list)>=delay:
                                enforced_support_in_last_iterations = (np.array(enforce_initial_support_list[-delay:])==True).any()
                                #log.info(f'support_enforced_list = {enforce_initial_support_list}')
                                apply_ft_stabilization = not enforced_support_in_last_iterations
                #log.info(f'apply_ft_stabilization = {apply_ft_stabilization}')
                return apply_ft_stabilization
                            
                    
                
            def loop(state):
                if 'SW' in methods:
                    update_shrink_wrap(0,loop_number)                
                error_dict = state['error_dict']
                #log.info(f'initial error dict = \n {error_dict}')
                hist = state['density_pair_history']
                enforce_initial_support_list = state.get('enforce_initial_support_list',[])
                #log.info(enforce_initial_support_list)
                iteration = 0
                step = 0
                real_pr =  self.projection_objects['real']
                copy = np.array
                latest_intensity = False
                sw_step = 0
                for iteration in range(1,max_iterations+1):
                    #error_target_is_reached = check_error_target(error_dict)
                    #if error_target_is_reached:
                    #    break
                    for key in order:                    
                        process=methods[key]['process']
                        repeats=methods[key]['iterations']
                        process_opt = methods[key].get('options',{})
                        #log.info('Loop:{} Running {} steps of {} with error_limit {}:'.format(iteration,repeats,key,relative_error_limit))
                        if key=='SW':
                            support = process.run(state['density_pair_history'][-1][1])
                            enforce_initial_support = error_dict['main'][-1:]>enforce_initial_support_error_limit
                            enforce_initial_support_list.append(enforce_initial_support)
                            real_pr.enforce_initial_support = enforce_initial_support
                            real_pr.support = support
                            state['mask'] = real_pr.support #mask #mask
                            sw_step+=1
                            update_shrink_wrap(sw_step,loop_number)
                        elif key == 'SW_center':
                            enforce_initial_support = error_dict['main'][-1]>enforce_initial_support_error_limit
                            real_pr.enforce_initial_support = enforce_initial_support
                            enforce_initial_support_list.append(enforce_initial_support)
                            for i in range(repeats):
                                #log.info('apply SW center')
                                support,ft_density,density = process.run(state['density_pair_history'][-1][1])
                                real_pr.support = support
                                state['mask'] = real_pr.support #mask #mask
                                state['density_pair_history']=hist[1:]+((ft_density,density),)
                                sw_step+=1
                                update_shrink_wrap(sw_step,loop_number)
                        else:
                            if key in ['ER_non_FXS','HIO_non_FXS']:
                                if isinstance(latest_intensity,bool):
                                    latest_intensity = np.abs(hist[-1][0]).real
                                    self.projection_objects['reciprocal'].fixed_intensity=latest_intensity
                            else:
                                latest_intensity = False                        
                                
                            if change_to_ft_stab(process_opt,key,enforce_initial_support_list):
                                #log.info('\n\n changing to ft stab \n')
                                process = routines[key+'_ft_stab']
                            errs={}
                            for i in range(repeats):
                                self.projection_objects['hio'].beta = hio_beta_ramp.eval(step)
                                #log.info(f'beta value = {self.projection_objects["hio"].beta}')
                                hist = state['density_pair_history']
                                # Break loop if relative error limits are breached. Start new loop with best known reconstruction.
                                #if i>2:
                                #    rel_limit_is_reached = check_error_gain_limit(error_dict,key)
                                #    if rel_limit_is_reached:
                                #        #log.info('break {} loop.'.format(key))
                                #        if len(hist)>2:
                                #            state['density_pair_history'] = state['best_density_pair']
                                #        break                            
                                #log.info('current density shapes reciprocal={} real={}'.format(density_pairs[-1][0].shape,density_pairs[-1][1].shape))
                                        
                                new_density_pair = process.run(*hist[-1])                            
                                copied_density_pair = tuple(copy(a) for a in new_density_pair)
                                state['density_pair_history']=hist[1:]+(copied_density_pair,)
                                main_error = main_error_routine(error_dict)
                                #if key == 'ER':
                                #    if error_dict['main'][-1] <= 2e-3:
                                #        #methods['ER']['process'] = routines['ER_ft_stab']
                                #        pass
                                error_dict['main'].append(main_error)
                                
                                if  state['best_error'] > main_error:                                
                                    state['best_error'] = main_error
                                    state['best_density_pair'] = copied_density_pair
                                    state['best_iteration'] = iteration
                                    state['best_mask'] = state['mask']                                
                                step+=1                                
                            #log.info(f'{error_dict}')
                            errs = {category+'_'+key:error_dict[category][key][-1]  for category in error_dict for key in error_dict[category] if category != 'main'}                        
                            errs['main'] = error_dict['main'][-1]
                            xprint('P{}: {} Loop:{} Method:{} Last Errors: \n{}  Best Error: {}\n number of particles = {}, max_density={}'.format(Multiprocessing.get_process_name(),loop_name,iteration,key,errs,state['best_error'],self.rprojection.number_of_particles,np.max(new_density_pair[1])))

                if state['best_iteration']>loop_opt.get('best_density_not_in_first_n_iterations',np.inf):
                    log.info('Selecting density with lowest error metric and continue.')
                    state['density_pair_history']=state['density_pair_history'][1:]+(state['best_density_pair'],)
                    real_pr.support = state['best_mask']
                    state['mask']=state['best_mask']
                state['enforce_initial_support_list']=enforce_initial_support_list
                return state,iteration
            return loop
        loops = [generate_loop_method(name,loop_opt.sub_loops[name],_id) for _id,name in enumerate(loop_opt.sub_loops.order) ]        
        self.loops=loops

        #### assemble main loop ####
        def create_initial_state():
            initial_support = self.projection_objects['real'].initial_support
            real_density_guess = real_density_guess_method()
            #log.info('density guess type = {}'.format(real_density_guess.dtype))
            reciprocal_density_guess=fourier_transform(real_density_guess)
            real_density_guess = inverse_fourier_transform(reciprocal_density_guess)
            #real_density_guess.imag = 0
            #real_density_guess[real_density_guess.real<0]=0
            #real_density_guess[~initial_support]=0
            density_pairs=((reciprocal_density_guess,real_density_guess),)*loop_opt.get('history_length',3)            
                
            error_dict = init_error_dict()

            #first loop
            initial_state = {
                'density_pair_history':density_pairs,
                'error_dict':error_dict,
                'mask':initial_support,
                'best_density_pair':density_pairs[-1],
                'best_error':np.inf,
                'best_iteration':0,
                'best_mask':initial_support}
            return initial_state
        def generate_output(state,iterations,initial_densities,initial_mask):
            best_density_pair = state['best_density_pair']
            best_error = state['best_error']
            best_iteration = state['best_iteration']
            best_mask = state['best_mask']

            output_modifier=routines['output_modifier']
            log.info('last density pair dtypes = {},{}'.format(state['density_pair_history'][-1][0].dtype,state['density_pair_history'][-1][1].dtype))
            out_density_pair = output_modifier.run(*best_density_pair[:2])
            out_last_density_pair = output_modifier.run(*state['density_pair_history'][-1][:2])
            log.info('out last density pair dtypes = {},{}'.format(out_last_density_pair[0].dtype,out_last_density_pair[1].dtype))
            log.info("last density shape = {}".format(out_last_density_pair[1].shape))
            calc_deg2_invariant=routines['calc_deg2_invariant']
            last_deg2_invariant = calc_deg2_invariant.run(out_last_density_pair[1])
            #last_deg2_invariant = calc_deg2_invariant.run(state['density_pair_history'][-1][1])
            
            #log.info("fraction shapes ={}".format([i.shape for i in self.results.get('n_particles_fraction',np.array([]))]))
            error_dict = self.arrayfy_error_dict(state['error_dict'])
            masked_projection_matrices = []
            for mask, matrix in zip(self.rprojection.radial_mask,self.rprojection.projection_matrices):
                tmp = np.array(matrix)
                tmp[~mask]=0
                masked_projection_matrices.append(tmp)
            resultDict={'real_density':out_density_pair[1],
                        'last_real_density':out_last_density_pair[1],
                        'reciprocal_density':out_density_pair[0],
                        'last_reciprocal_density':out_last_density_pair[0],
                        'final_error':best_error,
                        'initial_density':initial_densities[1],
                        'initial_support':initial_mask,
                        'error_dict':error_dict,
                        'support_mask':best_mask,
                        'last_support_mask':state['mask'],
                        'loop_iterations':np.sum(iterations)+1,
                        'fxs_unknowns':self.results['fxs_unknowns'],
                        'n_particles':np.array(self.results.get('n_particles',[])),
                        'n_particles_gradients':np.array(self.results.get('n_particles_gradients',[])),
                        'n_particles_fraction':np.array(self.results.get('n_particles_fraction',[])),
                        "grid_pair":{"real_grid":self.grid_pair.realGrid,
                                     "reciprocal_grid":self.grid_pair.reciprocalGrid},
                        'projection_matrices':masked_projection_matrices,
                        'last_deg2_invariant':last_deg2_invariant}
            return resultDict
        def main_loop(*args,**kwargs):
            initial_state = create_initial_state()
            initial_densities = tuple(d.copy() for d in initial_state['best_density_pair'])
            initial_mask = initial_state['mask'].copy()
            iterations = []
            state = initial_state
            next_state = False
            for loop in loops:
                next_state,iteration = loop(state)
                iterations.append(iteration)
                state = next_state
            out = generate_output(state,iterations,initial_densities,initial_mask)
            return out        
        return main_loop
        
        
    def init_error_dict(self):
        error_names = opt.main_loop.error.methods
        self.results['errors'] = {}
        self.results['errors']['real'] = {name:[] for name in error_names['real']['calculate']}        
        self.results['errors']['reciprocal'] = {name:[] for name in error_names['reciprocal']['calculate']}        
        self.results['errors']['main'] = []
        return self.results['errors']

    def generate_internal_error_limits(self):
        error_methods = opt.main_loop.error.methods
        error_limits = opt.main_loop.error.limits
        error_gain_limits = opt.main_loop.error.gain_limits
        error_names = [*['real_'+key for key in error_methods['real']],
                       *['reciprocal_'+key for key in error_methods['reciprocal']],
                       *['main']]
        
        gain_limits = {name:{phasing_part:lims[_id] for phasing_part,lims in error_gain_limits.items()} for _id,name in enumerate(error_names)}
        limits = {name:lim for name,lim in zip(error_names,error_limits)}
        log.info(limits)
        log.info(gain_limits)
        return limits,gain_limits

    def generate_check_error_routines(self):
        limits,gain_limits = self.generate_internal_error_limits()
        def check_gain_limit(error_dict,loop_method_name):
            return self.check_error_gain_limit(error_dict,gain_limits,loop_method_name)
        def check_limit(error_dict):
            return self.check_error_target_reached(error_dict,limits)
        return check_limit,check_gain_limit
    
    def check_error_gain_limit(self,error_dict,relative_error_limits,loop_method_name):
        error_dict = {**{'real_'+key:err for key,err in error_dict['real'].items()},
                      **{'reciprocal_'+key:err for key,err in error_dict['reciprocal'].items()},
                      **{'main':error_dict['main']}
                      }
        rel_limit_is_reached = False
        for err_name in error_dict:
            limit = relative_error_limits[err_name][loop_method_name]
            if not isinstance(limit , bool):
                errs = error_dict[err_name]
                if len(errs)>=3:
                    rel_err = 1-errs[-2]/errs[-1]
                    #log.info ('\n {} = {} \n'.format(err_name,rel_err))
                    rel_limit_is_reached = (rel_err >= limit) 
                    if rel_limit_is_reached:
                        log.info('Error limit reached: {} change = {}% limit = {}%'.format(err_name,(rel_err)*100,limit*100))
                        break
        return rel_limit_is_reached
    def check_error_target_reached(self,error_dict,error_targets):        
        error_names = opt.main_loop.error.methods
        target_is_reached = False
        error_list_to_short = False
        try:
            error_dict = {**{'real_'+key:err[-1] for key,err in error_dict['real'].items()},
                          **{'reciprocal_'+key:err[-1] for key,err in error_dict['reciprocal'].items()},
                          **{'main':error_dict['main'][-1]}
                          }
        except IndexError as e:
            error_list_to_short = True
        if not error_list_to_short:    
            for err_name in error_dict:
                target = error_targets[err_name]
                if not isinstance(target , bool):
                    err = error_dict[err_name]
                    target_is_reached = (err < target)
                    if target_is_reached:
                        log.info('Target error reached: {}={} limit = {}%'.format(err_name,errs[-1],target))
                        break
        return target_is_reached
                          
    def generate_update_support_mask(self,shrink_wrap_specifier):
        real_pr = self.projection_objects['real']
        def update_real_support_mask(support_mask,*args):
            real_pr.mask = support_mask
        return update_real_support_mask

    def generate_density_guess_method(self,density_guess_specifier,real_grid):
        opt=settings.project.density_guess
        if density_guess_specifier['amplitude_function']=='random':
            def amplitude_function(points):
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                return 1+1/opt.random.SNR*np.random.rand(*points.shape[:-1])
        radius = density_guess_specifier['radius']
        if isinstance(radius,bool):
            radius = settings.project.particle_radius
        #log.info('initial density radius {}'.format (radius))
        
        if self.dimensions == 2:                
            integrator = mLib.PolarIntegrator(real_grid[:])
        elif self.dimensions == 3:
            integrator = mLib.SphericalIntegrator(real_grid[:])
            
        total_intensity = self.rprojection.integrated_intensity
        data_type = np.dtype(complex)#settings.project.fourier_transform.data_type

        
        density_type=density_guess_specifier['type']        
        if density_type=='ball':
            if radius<0:
                radius = np.max(real_grid[...,0])
            coord_sys=density_guess_specifier['coord_sys']
            guessed_density_function=SampleShapeFunctions.get_disk_function(radius,amplitude_function=amplitude_function,coordSys=coord_sys)
                
            
            def generate_density():
                #log.info('it is called really')
                density = guessed_density_function(real_grid)
                #scale density to projection data
                total_squared_density = integrator.integrate((density*density.conj()).real)
                #log.info("total squared density = {}".format(total_squared_density))
                #log.info("total intensity = {}".format(total_intensity))                
                density*=np.sqrt(total_intensity/total_squared_density)
                if data_type == 'complex':
                    density = density.astype(complex)
                #log.info('initial total density = {}'.format(integrator.integrate(density**2)))
                return density
        elif density_type=='bump':
            if radius<0:
                radius = np.max(real_grid[...,0])
            slope = opt.bump.slope
            bump_func = mLib.get_test_function(support=[-radius,radius],slope=slope)
            def generate_density():
                A=amplitude_function(real_grid[:])
                #log.info(real_grid[...,0])
                density = A*bump_func(real_grid[...,0])
                #density = bump_func(real_grid[...,0])
                #log.info(density[:,0,0])
                total_squared_density = integrator.integrate((density*density.conj()).real)
                density*=np.sqrt(total_intensity/total_squared_density)
                #ft_density = ft(density.astype(complex))
                #m = ft_density !=0
                #ft_density[m]= ft_density[m]*(np.sqrt(np.abs(I[m]).real)/np.abs(ft_density[m]))
                #density = ift(ft_density)
                if data_type == 'complex':
                    density = density.astype(complex)
                return density
        elif density_type == 'low_resolution_autocorrelation':
            pr = self.projection_objects['reciprocal'].full_projection_matrices
            icht = self.process_factory.get_operator('inverse_harmonic_transform')
            ift = self.process_factory.get_operator('inverse_fourier_transform')
            ft = self.process_factory.get_operator('fourier_transform')
            I = icht(pr)
              
            #thresh = opt.low_resolution_autocorrelation.threshold_to_max
            V_low = self.projection_objects['reciprocal'].low_resolution_intensity_coefficients
            pr = self.projection_objects['reciprocal'].projection_matrices
            V = [np.zeros(p.shape,dtype=complex) for p in pr]
            for vl,v in zip(V_low,V):
                v[:]=vl
            icht = self.process_factory.get_operator('inverse_harmonic_transform')
            ift = self.process_factory.get_operator('inverse_fourier_transform')
            low_res_autocorrelation = ift(icht(pr)).real
            low_res_autocorrelation[low_res_autocorrelation<0]=0
            _max = low_res_autocorrelation.max()
            low_res_density_guess = low_res_autocorrelation
            mean_density = low_res_density_guess.mean()
            #db.save("/gpfs/exfel/theory_group/user/berberic/MTIP/test/inital_density/d1_low_res.vts",[np.abs(low_res_density_guess)],grid_type='spherical',grid = real_grid[:])
            radius = settings.project.particle_radius
            bump_func = mLib.get_test_function(support=[-radius,radius],slope=0.1)
            def generate_density():
                A=amplitude_function(real_grid[:]).astype(complex)
                density = low_res_density_guess*A
                density[density<0]=0
                density*=bump_func(real_grid[...,0])
                total_squared_density = integrator.integrate((density*density.conj()).real)
                density*=np.sqrt(total_intensity/total_squared_density)
                return density
        else:
            e=AssertionError('density type "{}" is not known.'.format(density_type))
            log.error(e)
            raise e
        return generate_density

    def generate_update_shrink_wrap(self,sw_object):        
        default_sigma = sw_object.default_sigma
        opt = settings.project
        loop_opt = opt.main_loop
        sw_opt = opt.projections.real.shrink_wrap
        
        nloops = len(loop_opt.sub_loops.order)
        sw_sigma_ramps = []
        sw_thresh_ramps = []
        
        for lid in range(nloops): 
            n_sw_sigmas = len(sw_opt.sigmas)
            n_sw_thresholds = len(sw_opt.thresholds)
            if n_sw_sigmas-1 < lid:
                sw_sigma = False
                log.info(f'No SW sigma value in settings for loop with id = {lid} in {loop_opt.sub_loops.order} specified. Settings sigma to: {sw_sigma} ')
            else:
                sw_sigma = sw_opt.sigmas[lid]
            if not isinstance(sw_sigma,(list,tuple)):
                sw_sigma = [sw_sigma]
            sw_sigma_ramp = LinearRamp(*sw_sigma,default_start = sw_object.default_sigma,default_stop = sw_object.default_sigma)
            sw_sigma_ramps.append(sw_sigma_ramp)
            #log.info(f'{loop_name} : sigma = {sw_sigma} from simgas = {sw_opt.sigmas}')
            
            n_sw_thresholds = len(sw_opt.thresholds)            
            if n_sw_thresholds-1 < lid:
                sw_threshold = 0.1
                log.info(f'No SW threshold value in settings for loop with id = {lid} in {loop_opt.sub_loops.order} specified. Settings threshold to: {sw_threshold}')
            else:
                sw_threshold = sw_opt.thresholds[lid]

            if not isinstance(sw_threshold,(list,tuple)):
                sw_threshold = [sw_threshold]
            sw_thresh_ramp = LinearRamp(*sw_threshold)
            sw_thresh_ramps.append(sw_thresh_ramp)
        def update_shrink_wrap(iteration,loop_number):
            sw_sigma_ramp = sw_sigma_ramps[loop_number]
            if not sw_sigma_ramp.undefined:
                sw_object.gaussian_sigma = sw_sigma_ramp(iteration)
            sw_threshold_ramp = sw_thresh_ramps[loop_number]
            if not sw_threshold_ramp.undefined:
                sw_object.threshold = sw_threshold_ramp(iteration)
    
            #log.info(f'In {iteration}-th SW application of loop {loop_number} : sigma = {sw_object.gaussian_sigma}, threshold = {sw_object.threshold}')
            

        return update_shrink_wrap
    def arrayfy_error_dict(self,error_dict):
        for category in error_dict:
            if category == 'main':
                error_dict[category] = np.array(error_dict[category])
            else:
                for err_key,errs in error_dict[category].items():
                    error_dict[category][err_key] = np.array(error_dict[category][err_key])
        return error_dict


    def generate_phasing_loop(self):
        #self.load_mtip_data()
        log.info('Setting up operators')
        operators,grid_pair=self.assemble_operators()
        self.process_factory.addOperators(operators)
        self.grid_pair = grid_pair
        log.info('Setting up phasing routines (HIO,ER,etc.)')
        self.routines = self.assemble_MTIP_routines()
        log.info('Assemble phasing loop (HIO,ER,etc.)')
        self.phasing_loop = self.assemble_phasing_loop()
