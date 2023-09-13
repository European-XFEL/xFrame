from xframe.interfaces import DependencyMeta
import abc

#class DependencyInterfaceTemplate:
#    pass

class MatplotlibInterface(abc.ABC):
    _external_dependency_name_ = 'matplotlib'

class MatplotlibDependency(MatplotlibInterface,metaclass = DependencyMeta):
  pass
  

class MPLToolkitInterface(abc.ABC):
    _external_dependency_name_ = 'matplotlib_toolkit'
class MPLToolkitDependency(MPLToolkitInterface,metaclass = DependencyMeta):
  pass
class OpenCVInterface(abc.ABC):
    _external_dependency_name_ = 'opencv'
    __dependency__='cv2 (opencv-python)'
    @abc.abstractmethod
    def get_polar_image(*args,**kwargs):
        pass
class OpenCVDependency(OpenCVInterface,metaclass = DependencyMeta):
  pass
