import logging
import numpy as np
import traceback
import os

from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct
from xframe.library.mathLibrary import Alignment, AlignedAveragerStruct, AlignedAverager,SphericalIntegrator
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
        self.averager_struct = AlignedAveragerStruct(fourier_struct = self.fourier_struct,
                                                     alignment = opt.alignment,
                                                     averaging = opt.averaging,
                                                     n_processes = opt.multi_processing.n_processes)
        
        self.use_specific_reference = False
        if opt.averaging.mode =='single_reference':
            if isinstance(opt.averaging.single_reference.selection,(list,tuple)):
                self.use_specific_reference = True
                
        self.reconstruction_files = None
        self.reconstruction_specifiers = None
        self.main_reconstruction_errors = None
        self.data_loader = None
        self.averager = None
        self.input_proj_matrices = None

    def load_reconstruction_info(self):
        opt = self.opt
        assert len(opt.reconstruction_files)>0,'No reconstruction_files specified in settings. Stopping!'
        db = self.db
        file_paths = [os.path.join(db.get_path('reconstructions',is_file=False),path) for path in opt.reconstruction_files]
        _struct = SphericalFourierTransformStruct(**db.load(file_paths[0],h5_path='fourier_transform_struct'))
        _struct.use_gpu=self.opt.GPU.use
        self.fourier_struct = _struct
        self.averager_struct = AlignedAveragerStruct(fourier_struct = self.fourier_struct,
                                                     alignment = opt.alignment,
                                                     averaging = opt.averaging,
                                                     n_processes = opt.multi_processing.n_processes)
        self.dim = _struct.dimension
        input_proj_dict = db.load(file_paths[0],h5_path = 'projection_matrices')
        self.input_proj_matrices = tuple(input_proj_dict[f'{i}'] for i in range(_struct.angular_bandwidth))
        rec_ids = []
        errors = []
        for file_id,path in enumerate(file_paths):
            dat = db.load(path,as_h5_object=True)
            rec_results=dat['reconstruction_results']
            if opt.selection.use_selection_file:
                keys = db.load('selection',h5_path=path)
            else:
                keys = list(rec_results.keys())
            for key in keys:
                if self.use_specific_reference:
                    if (np.array(file_id)==np.array(self.opt.averaging.single_reference.selection)).all():
                        continue
                rec = rec_results[str(key)]
                rec_ids.append((file_id,int(key)))
                errors.append(rec['error_dict/main'][-1])
            dat.close()
            
        if self.use_specific_reference:
            rec_ids = [(self.opt.averaging.single_reference.selection[0],self.opt.averaging.single_reference.selection[1])]+rec_ids
            errors =  [-np.inf]+errors 
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

        if self.use_specific_reference:
            error_mask[0]=True #reference is at 0'th position and should not be filtered
            
        self.main_reconstruction_errors = errors[error_mask]
        self.reconstruction_specifiers = self.reconstruction_specifiers[error_mask]
        sort_ids = np.argsort(self.main_reconstruction_errors)
        
        if not isinstance(opt.selection.n_reconstructions,int):
            sl = slice(None)
        else:
            sl = slice(None,opt.selection.n_reconstructions,1)
        
        self.main_reconstruction_errors = self.main_reconstruction_errors[sort_ids][sl]
        self.reconstruction_specifiers = self.reconstruction_specifiers[sort_ids][sl]

        
        
        
    def average_2d(self):
        averager_struct = self.averager_struct
        averager_struct.alignment.rotational.apply=False
        averager = AlignedAverager(self.averager_struct)
        self.averager = averager
        averages = averager.average(self.data_loader.load,self.data_loader.n_reconstructions)
    def average_3d(self):        
        averager = AlignedAverager(self.averager_struct,dataset_length = 3)        
        self.averager = averager
        averages = averager.average(self.data_loader.load,self.data_loader.n_reconstructions)        
        return averages
        
        
    def run(self):
        self.load_reconstruction_info()
        self.select_reconstructions()
        self.data_loader = DataLoader(self.reconstruction_files,self.reconstruction_specifiers)
        
        if self.dim ==2:
            res = self.average_2d()
        else:
            res = self.average_3d()

        out_dict = self.post_processing(res)
        try:
            self.db.save('average_results',out_dict)
        except Exception as e:
            traceback.print_exc()
            log.error(e)
            
    def compute_resolution_metrics(self,results):
        ft = self.averager.fourier
        angular_axes = tuple(i for i in range(-1,-ft.dimensions,-1))
        main_res = results[0][0]
        rmean = np.abs(ft.forward_cmplx(main_res[0].mean)).real**2
        qmean = np.abs(main_res[1].mean).real**2
        var = main_res[1].variance
        PRTF1 = np.mean(np.sqrt(qmean/(var+qmean)),axis = angular_axes)
        PRTF2 = np.mean(np.sqrt(rmean/(var+rmean)),axis = angular_axes)
        
        integrate = SphericalIntegrator(ft.real_grid).integrate_normed
        mean_std_integrated_progression = np.array([ integrate(res[0].mean.real/np.sqrt(np.abs(res[0].variance).real)) for res in results[2] ])
        progression_counts = np.array([res[0].count for res in results[2]])
        
        metrics = {"PRTF":PRTF2,"PRTF_from_scattering_amplitude":PRTF1,'mean_std_integrated_progression':mean_std_integrated_progression,'progression_counts':progression_counts}
        return metrics
    def post_processing(self,results):
        var_objects = results[0][0]
        resolution_metrics_dict = self.compute_resolution_metrics(results)
        out = {'real_density':var_objects[0].mean,
               'real_variance':var_objects[0].variance,
               'reciprocal_density':var_objects[1].mean,
               'reciprocal_variance':var_objects[1].variance,
               'mask':var_objects[2].mean,
               'mask_variance':var_objects[2].variance,
               'resolution_metrics': resolution_metrics_dict,
               'n_used_reconstructions':var_objects[0].count,
               'fourier_transform_struct':self.fourier_struct.__dict__,
               'initial_ids':results[1],
               'input_proj_matrices':self.input_proj_matrices
               }
        first_variance_per_step = {f'{i}':{label:{'mean':v[n].mean,'count':v[n].count,'variance':v[n].variance} for n,label in enumerate(['real','reciprocal','mask'])} for i,v in enumerate(results[2])}
        out['variance_step_progression']=first_variance_per_step
        all_variances = {f'{i}':{label:{'mean':v[n].mean,'count':v[n].count,'variance':v[n].variance} for n,label in enumerate(['real','reciprocal','mask'])} for i,v in enumerate(results[0][1:])}
        out['other_index_combinations'] = all_variances
        return out        
                

class DataLoader:
    def __init__(self,file_paths,specifiers):
        self.paths = file_paths
        self.current_h5_object = None
        self.current_path = None
        self.reconstruction_specifiers = specifiers
        self.db = database.project
        db =self.db
        self.n_reconstructions = len(specifiers)
        
        
    def load(self,rec_id):
        path_id,data_id = self.reconstruction_specifiers[rec_id]
        path = self.paths[path_id]
        if path != self.current_path:
            if self.current_path is not None:
                self.current_h5_object.close()
            self.current_h5_object = self.db.load(path,as_h5_object = True)
        o= self.current_h5_object
        result = o[f'reconstruction_results/{data_id}']
        dataset = tuple((result['last_real_density'][:],result['last_reciprocal_density'][:],result['last_support_mask'][:].astype(complex)))
        return dataset
        


