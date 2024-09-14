import logging
import numpy as np
import traceback

from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct
from xframe.library.mathLibrary import Alignment, AlignedAveragerStruct, AlignedAverager
from xframe.library.pythonLibrary import xprint

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
        self.comm_module=comm_modul

        self.dim = None
        self.reconstruction_specifiers = np.array([],dtype = int)
        self.errors = []
        self.fourier_struct = SphericalFourierTransformStruct()

        oft = self.opt
        self.averager_struct = AlignedAveragerStruct(fourier_struct = self.fourier_struct,
                                                     alignment = opt.alignment,
                                                     averaging = opt.averaging,
                                                     n_processes = opt.multi_processing.n_processes)
        
        self.reconstruction_files = None
        self.reconstruction_specifiers = None
        self.main_reconstruction_errors = None
        self.data_loader = None
        self.averager = None

    def load_reconstruction_info(self):
        assert len(self.opt.reconstruction_files)>0,'No reconstruction_files specified in settings. Stopping!'
        db = self.db
        file_paths = [os.path.join(db.get_path('reconstructions',is_file=False),path) for path in self.opt.reconstruction_files]
        _struct = SphericalFourierTransformStruct(**db.load(file_paths[0],h5_path='fourier_transform_struct'))
        self.fourier_struct = _struct
        self.dim = _struct.dimension
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
                rec = rec_results[str(key)]
                rec_ids.append((file_id,int(key)))
                errors.append(rec['error_dict/main'][-1])
            dat.close()
        rec_ids = np.array(rec_ids)
        errors =  np.array(errors)
        self.reconstruction_files = file_paths
        self.reconstruction_specifiers = rec_ids
        self.main_reconstruction_errors = errors

    def select_reconstructions(self):
        error_limits = opt.selection.error_limits
        if not isinstance(error_limits[0],(int,float)):
            error_limits[0]=-np.inf
        if not isinstance(error_limits[1],(int,float)):
            error_limits[1]=np.inf
        errors = self.main_reconstruction_errors
        error_mask = (errors>error_limits[0]) & (errors < error_limits[1])
        self.main_reconstruction_errors = errors[error_mask]
        self.reconstruction_specifiers = self.reconstruction_specifiers[error_mask]
        
    def average_2d(self):
        averager_struct = self.averager_struct
        averager_struct.alignment.rotational.apply=False
        averager = AlignedAverager(self.averager_struct)
        self.averager = averager
        averages = averager.average(self.data_loader.load,self.data_loader.n_reconstructions)
    def average_3d(self):
        averager = AlignedAverager(self.averager_struct)
        self.averager = averager
        averages = averager.average(self.data_loader.load,self.data_loader.n_reconstructions)
        
        
    def run(self):
        self.load_reconstruction_info()
        self.select_reconstructions()
        self.data_loader = DataLoader(self.reconstruction_files,self.reconstruction_specifiers)
        
        if self.dim ==2:
            res = self.average_2d()
        else:
            res = self.average_3d()
    def post_processing(self,result):
        var_objects = results[0][0]
        out = {'real_density':var_objects[0].mean,
               'normalized_real_density':self.averager.normalize(var_objects[0].mean.real),
               'real_variance':var_objects[0].variance,
               'reciprocal_density':var_objects[1].mean,
               'reciprocal_variance':var_objects[1].variance,
               'mask':var_objects[2].mean,
               'mask_variance':var_objects[2].variance,
               'input_meta':{'ft_struct'
               }
        
        

class DataLoader:
    def __init__(self,file_paths,specifiers):
        self.paths = file_paths
        self.current_h5_object = None
        self.current_path = None
        self.reconstruction_specifiers
        self.db = database.project
        db =self.db
        self.n_reconstructions = len(self.reconstruction_specifiers)
        
        
    def load(self,rec_id):
        path_id,data_id = self.reconstruction_specifiers[rec_id]
        path = self.paths[path_id]
        if path != self.current_path:
            self.current_h5_object.close()
            self.current_h5_object = self.db.load(path,as_h5_object = True)
        o= self.current_h5_object
        result = o[f'reconstruction_results/{data_id}']
        dataset = tuple((result['last_real_density'][:],result['last_reciprocal_density'][:],result['last_support_mask'][:].astype(complex)))
        return dataset
        


