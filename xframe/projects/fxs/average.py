import logging
import numpy as np
import traceback
import os
from numbers import Integral
from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct
from xframe.library.alignment import Aligner, AlignedAverager,DataSourceInterface,Map
from xframe.library.mathLibrary import SphericalIntegrator

from xframe.library.pythonLibrary import xprint
from xframe.interfaces import ProjectWorkerInterface

from xframe import settings
from xframe import database
from xframe import Multiprocessing

log=logging.getLogger('root')

#class Worker(RecipeInterface):
class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        comm_module = Multiprocessing.comm_module        
        self.opt = settings.project
        self.db = database.project
        #self.mtip=MTIP(self.process_factory,database,comm_module,opt)
        #self.mp_opt=opt['multi_process']
        #self.mtip_opt=opt
        self.comm_module=comm_module

        self.dim = None
        self.reconstruction_specifiers = np.array([],dtype = int)
        self.errors = []
        self.fourier_struct = SphericalFourierTransformStruct()

        opt = self.opt
        self.averager = None
        
        self.use_single_reference = False
        if opt.averaging.mode =='single_reference':
            if opt.averaging.mode == 'single_reference':
                self.use_single_reference = True
                
        self.reconstruction_files = None
        self.reconstruction_specifiers = None
        self.main_reconstruction_errors = None
        self.data_source = None
        self.averager = None
        self.input_proj_matrices = None
        self._single_reference_id = None

    def load_reconstruction_info(self):
        opt = self.opt
        assert len(opt.reconstruction_files)>0,'No reconstruction_files specified in settings. Stopping!'
        db = self.db

        file_paths = [os.path.join(db.get_path('reconstructions',is_file=False),path).format(today = db.get_time_string()) for path in opt.reconstruction_files]
        print(file_paths)
        _struct = SphericalFourierTransformStruct(**db.load(file_paths[0],h5_path='fourier_transform_struct'))
        _struct.use_gpu=self.opt.GPU.use
        self.fourier_struct = _struct
        self.averager = AlignedAverager(fourier_struct = self.fourier_struct,
                                        normalization_method = opt.alignment.normalization.method)
        self.dim = _struct.dimension
        input_proj_dict = db.load(file_paths[0],h5_path = 'projection_matrices')
        self.input_proj_matrices = tuple(input_proj_dict) #tuple(input_proj_dict[int(f'{i}')] for i in range(_struct.angular_bandwidth))
        rec_ids = []
        errors = []
        for file_id,path in enumerate(file_paths):
            dat = db.load(path,as_h5_object=True,path_modifiers = {'today':db.get_time_string()})
            rec_results=dat['reconstruction_results']
            if opt.selection.use_selection_file:
                keys = db.load('selection',h5_path=path)
            else:
                keys = list(rec_results.keys())
            for key in keys:
                rec = rec_results[str(key)]
                rec_ids.append((file_id,int(key)))
                errors.append(rec['error_dict/main'][-1])
            dat.close()
            
        rec_ids = np.array(rec_ids)
        errors = np.array(errors)
        
        self.reconstruction_files = file_paths
        self.reconstruction_specifiers = rec_ids
        self.main_reconstruction_errors = errors

    def select_reconstructions(self):
        ''' populates self.reconstruction_specifiers and self.main_reconstruction_errors based on error limits.
            The resulting arrays will be sorted by the error metric low to high
        '''
        opt = self.opt
        error_limits = opt.selection.error_limits
        if not isinstance(error_limits[0],(int,float)):
            error_limits[0]=-np.inf
        if not isinstance(error_limits[1],(int,float)):
            error_limits[1]=np.inf
        errors = self.main_reconstruction_errors
        error_mask = (errors>error_limits[0]) & (errors < error_limits[1])

        # how to make sure the reference is not maske if use_single_reference is true
            
        self.main_reconstruction_errors = errors[error_mask]
        self.reconstruction_specifiers = self.reconstruction_specifiers[error_mask]
        sort_ids = np.argsort(self.main_reconstruction_errors)
        
        if not isinstance(opt.selection.n_reconstructions,int):
            sl = slice(None)
        else:
            sl = slice(None,opt.selection.n_reconstructions,1)
        
        self.main_reconstruction_errors = self.main_reconstruction_errors[sort_ids][sl]
        self.reconstruction_specifiers = self.reconstruction_specifiers[sort_ids][sl]
        
        if self.use_single_reference:
            if self.opt.averaging.single_reference.reference_selection_mode == 'least_error':
                self._single_reference_id = 0
            else:
                reference_specifier = opt.averaging.single_reference.custom
                ref_id = int(np.flatnonzero(np.all(self.reconstruction_specifiers == reference_specifier, axis=1))[0])
                self._single_reference_id = ref_id

    def average(self):
        opt = self.opt
        do_rotational_alignment = self.opt.alignment.rotational.apply
        if self.dim == 2:
            do_rotational_alignment = False
            
        if self.opt.averaging.mode == 'pairwise':
            print(opt.alignment)
            averages = self.averager.average_pairwise(
                self.data_source,
                anchor_map_id = opt.alignment.align_by,
                consider_point_inverse = opt.alignment.consider_point_inverse,
                do_centering = opt.alignment.centering.apply,
                do_normalization = opt.alignment.normalization.apply,
                do_rotational_alignment = do_rotational_alignment,
                so3_radial_limits = opt.alignment.rotational.r_limits,
                n_processes = opt.multi_processing.n_processes                
            )
        elif self.opt.averaging.mode == 'single_reference':
            reference_id = self._single_reference_id
            averages = self.averager.average_single_reference(
                self.data_source,
                reference_id = reference_id,
                anchor_map = opt.alignment.align_by,
                consider_point_inverse = opt.alignment.consider_point_inverse,
                do_centering = opt.alignment.centering.apply,
                do_normalization = opt.alignment.normalization.apply,
                do_rotational_alignment = do_rotational_alignment,
                so3_radial_limits = opt.alignment.rotational.r_limits,
                n_processes = opt.multi_processing.n_processes                
            )
        return averages
        
    def run(self):
        self.load_reconstruction_info()
        self.select_reconstructions()
        self.data_source = DataSource(self.reconstruction_files,self.reconstruction_specifiers)
        res = self.average()
        out_dict = self.post_processing(res)
        try:
            self.db.save('average_results',out_dict)
        except Exception as e:
            traceback.print_exc()
            log.error(e)
            
    def compute_resolution_metrics(self,results):
        ft = self.averager.fourier
        main_res = results[0][0]
        rmean = np.abs(ft.forward_cmplx(main_res.mean)).real**2
        qmean = np.abs(results[0][1].mean).real**2
        qvar = results[0][1].variance
        
        integrator = SphericalIntegrator(ft.real_grid)
        mean_std_integrated_progression = np.array([integrator(res[0].mean.real/np.sqrt(np.abs(res[0].variance).real),normed = True) for res in results[2]])
        progression_counts = np.array([res[0].count for res in results[2]])

        print("yay new PRTFS",f"qmean{qmean.shape} qvar {qvar.shape}")
        PRTF1 = integrator.angular_only(np.sqrt(qmean/(qvar+qmean)),normed = True)
        PRTF2 = integrator.angular_only(np.sqrt(rmean/(qvar+rmean)),normed = True)
        metrics = {"PRTF":PRTF2,"PRTF_from_scattering_amplitude":PRTF1,'mean_std_integrated_progression':mean_std_integrated_progression,'progression_counts':progression_counts}
        return metrics
    
    def post_processing(self,results):
        var_objects = results[0]
        resolution_metrics_dict = self.compute_resolution_metrics(results)
        ft_d = self.averager.fourier.forward_cmplx(var_objects[0].mean)
        I = (ft_d*ft_d.conj()).real
        SAXS = np.mean(I,axis=tuple(range(1,I.ndim)))
        out = {'real_density':var_objects[0].mean,
               'real_count':var_objects[0].count,
               'real_variance':var_objects[0].variance,
               'reciprocal_density':var_objects[1].mean,
               'reciprocal_count':var_objects[1].count,
               'reciprocal_variance':var_objects[1].variance,
               'mask':var_objects[2].mean,
               'mask_count':var_objects[2].count,
               'mask_variance':var_objects[2].variance,
               'resolution_metrics': resolution_metrics_dict,
               'n_used_reconstructions':var_objects[0].count,
               'fourier_transform_struct':self.fourier_struct.__dict__,
               'initial_ids': results[1],
               'reconstruction_specifiers':self.data_source.reconstruction_specifiers,
               'reconstruction_files':self.data_source.paths,
               'input_proj_matrices':self.input_proj_matrices,
               "SAXS_from_density":SAXS
               }
        first_variance_per_step = {f'{i}':{label:{'mean':v[n].mean,'count':v[n].count,'variance':v[n].variance} for n,label in enumerate(['real','reciprocal','mask'])} for i,v in enumerate(results[2])}
        out['variance_step_progression']=first_variance_per_step
        return out        
                

class DataSource(DataSourceInterface):
    def __init__(self,file_paths,specifiers):
        self.paths = file_paths
        self.current_h5_object = None
        self.current_path = None
        self.reconstruction_specifiers = specifiers
        self.db = database.project
        db =self.db
        self.n_reconstructions = len(specifiers)
        
    def _open_file(self,path_id):
        path = self.paths[path_id]
        if path != self.current_path:
            if self.current_path is not None:
                self.current_h5_object.close()
            self.current_h5_object = self.db.load(path,as_h5_object = True)
        o= self.current_h5_object
        return o
    
    def __getitem__(self,items):
        anchor_id = None
        if isinstance(items,tuple):
            if len(items)==2 and isinstance(items[0],Integral) and isinstance(items[1],Integral):
                rec_id = items[0]
                anchor_id = items[1]
            else:
                IndexError('DataSource only supports integer indexing of dimensons 1 or 2. but index {items} was given')
        elif isinstance(items,Integral):
            rec_id=items
        else:
            IndexError('DataSource only supports integer indexing of dimensons 1 or 2. but index {items} was given')
            
        path_id,data_id = self.reconstruction_specifiers.__getitem__(rec_id)
        o = self._open_file(path_id)
        result = o[f'reconstruction_results/{data_id}']
        dataset = (Map(result['density_history'][-1].astype(complex,copy=False)),
                   Map(result['ft_density_history'][-1].astype(complex,copy=False),
                       domain='fourier'),
                   Map(result['support'][:].astype(complex,copy=False),
                       is_support=True))
        o.close()
        if anchor_id is not None:
            return dataset[anchor_id]
        else:
            return dataset
    
    def __len__(self):
        return len(self.reconstruction_specifiers)

