import abc
import logging
import traceback
from xframe.startup_routines import _inject_dependency


log=logging.getLogger('root')

class ProjectWorkerInterface(abc.ABC):
    def __init__(self):
        from xframe.settings import project as settings
        from xframe.database import project as db
        from xframe.Multiprocessing import comm_module as comm
        self.settings = settings
        self.db = db
        self.comm = comm
        
    @abc.abstractclassmethod
    def run(self,*argr):
        pass

class ExperimentWorkerInterface(abc.ABC):
    def __init__(self):
        from xframe.settings import experiment as settings
        from xframe.database import experiment as db
        from xframe.Multiprocessing import comm_module as comm
        self.settings = settings
        self.db = db
        self.comm = comm
    
    @abc.abstractmethod
    def get_data(self,mode:str,range:dict,data_modifier : object = 'None'):
        pass
    @abc.abstractmethod
    def get_pixel_grid_reciprocal(self,coord_system:str):
        pass
    @abc.abstractmethod
    def run(self):
        pass           
    
class CommunicationInterface(abc.ABC):
    @abc.abstractmethod
    def get_data(self):        
        pass
    @abc.abstractmethod
    def get_geometry(self):        
        pass
    @abc.abstractmethod
    def request_mp_evaluation(self,func,**kwargs):
        pass

class DatabaseInterface(abc.ABC):
    @abc.abstractmethod
    def load(self,analysisType,restultId):
        pass
    @abc.abstractmethod
    def save(self,analysisType,restultId):
        pass

class PresenterInterface(abc.ABC):
    @abc.abstractmethod
    def present(self):
        pass
    @abc.abstractmethod
    def save(self,path):
        pass


def error_message(cls):
    return f'Optional Dependency {cls._external_dependency_name_} missing.'

class DependencyImportError(Exception):
    def __init__(self,message=""):
        super().__init__(message)
        
class DependencyMeta(abc.ABCMeta):
    '''
    If this meta class is used by a class it will behave as follows.
    After Class creation the __call__, __getattr__ and __getattribute__ are overwritten.
    This cause any further interaction e.g calling the class or getting access to attributes
    trigger an attempt to load the actual external dependency.
    '''
    def __init__(cls,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #new_meta._arm(cls)
        cls._armed = True
        
    def _try_to_import_dependency(cls,*args,**kwargs):
        dependency_name = cls._external_dependency_name_
        try:
            dependency = _inject_dependency(dependency_name)
        except Exception as e:            
            message = f'Requested external dependency "{dependency_name}" could not be imported.'
            log.error(message)
            raise DependencyImportError(message) from e
        return dependency
    

    def __call__armed(cls,*args,**kwargs):
        instance = cls._try_to_import_dependency()(*args,**kwargs)
        return instance
    def __getattr__armed(cls,key):
        if key[0]!='_':
            dep = cls._try_to_import_dependency()
            attr = getattr(dep,key)
            return attr
    def __getattribute__armed(cls,key):
        if key[0]!='_':
            dep = cls._try_to_import_dependency()
            attr = getattr(dep,key)
        else:
            attr = super().__getattribute__(key)
        return attr

    
    def __call__(cls,*args,**kwargs):
        if not cls._armed:
            return super().__call__(*args,**kwargs)
        else:
            return cls.__call__armed(*args,**kwargs)
    def __getattr__(cls,key):
         if key[0]!='_':
            if cls._armed:
                return cls.__getattr__armed(key)
            
    def __getattribute__(cls,key):
        attr = super().__getattribute__(key)
        if key[0]!='_':          
            if cls._armed:
              return cls.__getattr__armed(key)
        return attr


class ClickInterface(abc.ABC):
    _external_dependency_name_ = 'click'
    @property
    @abc.abstractmethod
    def click(self):
        pass    
class ClickDependency(ClickInterface,metaclass=DependencyMeta):
    pass
