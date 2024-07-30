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
        rec_ids = []
        errors = []
        for file_id,path in enumerate(rec_files):
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

    def run(self):
        self.load_reconstruction_info()
        self.select_reconstructions()


from dataclasses import dataclass,field
from xframe.library.math_transforms import SphericalFourierTransformStruct    
from xframe.library.pythonLibrary import DictNamespace

class AveragerStruct:
    defaults = DictNamespace.dict_to_dictnamespace(
        {'fourier_struct': SphericalFourierTransformStruct(),
                'alignment':{'align_by_masks':False,
                             'normalization':{'apply':True,
                                              'method':np.max},
                             'positional':{'apply':True},
                             'rotational':{'apply':True,                                           
                                           'r_limits':[200,400],
                                           'consider_point_inverse':True}
                             },
                'averaging':{'n_datasets':'all',
                             'mode':'pairwise',
                             'single_reference':{'reference_selection_mode':'least_error',
                                                 'custom': (0,10)},
                             'n_random_orders':1
                             }
                }
    )
    
    def __init__(self,fourier_struct : SphericalFourierTransformStruct|None = None,alignment : dict|None =None,averaging : dict|None =None):

        if fourier_struct is None:
            self.fourier_struct=self.defaults.fourier_struct
        else:
            self.fourier_struct = fourier_struct
            
        if alignment is None:
            self.align_opt = self.defaults.alignment.copy()
        else:
            self.align_opt = DictNamespace.dict_to_dictnamespace(alignment)
            
        if averaging is None:
            self.aver_opt = self.defaults.averaging.copy()
        else:
            self.aver_opt = DictNamespace.dict_to_dictnamespace(averaging)
    
class Averager:
    pass
