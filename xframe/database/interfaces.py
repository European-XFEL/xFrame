import abc
import logging
log=logging.getLogger('root')
from xframe.interfaces import DependencyMeta

class HDF5Interface(abc.ABC):
    _external_dependency_name_ = 'hdf5'
    @abc.abstractmethod
    def save(path,data: dict):
        log.error('HDF5 dependency not injected. Is h5py python package installed ?')
    @abc.abstractmethod
    def load(path):
        log.error('HDF5 dependency not injected. Is h5py python package installed ?')

class HDF5Dependency(HDF5Interface,metaclass=DependencyMeta):
    pass
class YAMLInterface(abc.ABC):
    _external_dependency_name_ = 'yaml'
    @abc.abstractmethod
    def save(path,data: dict):
        log.error('YAML dependency not injected. Is the ruamel python package installed ? ')
    @abc.abstractmethod
    def load(path):
        log.error('YAML dependency not injected. Is the ruamel python package installed ?')
    @abc.abstractmethod
    def format_settings_dict(dict):
        log.error('YAML dependency not injected. Is the ruamel python package installed ?')

class YAMLDependency(YAMLInterface,metaclass = DependencyMeta):
    pass
class VTKInterface(abc.ABC):
    _external_dependency_name_ = 'vtk'
    @abc.abstractmethod
    def save(datasets, grid, file_path,dset_names = 'data', grid_type='cartesian'):
        log.error('VTK dependency not injected. Is the vtk package installed ')
    @abc.abstractmethod
    def load(path):
        log.error('VTK dependency not injected. Is the vtk python package installed ?')

class VTKDependency(VTKInterface,metaclass = DependencyMeta):
    pass
class PDBInterface(abc.ABC):
    _external_dependency_name_ = 'pdb'
    @abc.abstractmethod
    def load(pdb_id : str):
        log.error('PDB dependency not injected. Is the pdb_eda python package installed ? ')
    @abc.abstractmethod
    def save(path,data):
        log.error('PDB dependency not injected. Is the pdb_eda python package installed ?')

class PDBDependency(PDBInterface,metaclass = DependencyMeta):
    pass
class OpenCVInterface(abc.ABC):
    _external_dependency_name_ = 'opencv'
    @abc.abstractmethod
    def load(path):
        log.error('OpenCV dependency not injected. Is the opencv-python python package installed ? ')
    @abc.abstractmethod
    def save(path,data):
        log.error('OpenCV dependency not injected. Is the opencv-python python package installed ?')

class OpenCVDependency(OpenCVInterface, metaclass = DependencyMeta):
    pass
