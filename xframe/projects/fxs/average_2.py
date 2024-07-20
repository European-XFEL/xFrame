import logging
import numpy as np
import traceback

from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct
from xframe.library.mathLibrary import Alignment
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

    def load_reconstruction_info(self):
        assert len(self.opt.reconstruction_files)>0,'No reconstruction_files specified in settings. Stopping!'
        db = self.db
        file_paths = [os.path.join(db.get_path('reconstructions',is_file=False),path) for path in self.opt.reconstruction_files]
        _struct = SphericalFourierTransformStruct(**db.load(file_paths[0],h5_path='fourier_transform_struct'))
        self.fourier_struct = _struct
        self.dim = _struct.dimension
        for file_id,path in enumerate(file_paths):
            
            
    
    def run(self):
        self.load_reconstruction_info()
