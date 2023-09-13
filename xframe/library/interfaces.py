import abc
from xframe.interfaces import DependencyMeta
class DatabaseInterface(abc.ABC):
    @abc.abstractmethod
    def save(self,name,data,**kwargs):
        pass
    
    @abc.abstractmethod
    def load(self,name,**kwargs):
        pass


class SphericalHarmonicTransformInterface(abc.ABC):
    '''
    Interface for the external library shtns which supplies spherical harmonic transforms
    '''
    _external_dependency_name_ = 'shtns'
    @abc.abstractmethod
    def phi(self):
        pass
    @abc.abstractmethod
    def theta(self):
        pass
    @abc.abstractmethod
    def forward_l(self):
        pass
    @abc.abstractmethod
    def forward_m(self):
        pass
    @abc.abstractmethod
    def inverse_m(self):
        pass
    @abc.abstractmethod
    def inverse_l(self):
        pass

class SphericalHarmonicTransformDependency(SphericalHarmonicTransformInterface,metaclass=DependencyMeta):
    pass

class SoftInterface(abc.ABC):
    '''
    Interface for the external library shtns which supplies spherical harmonic transforms
    '''
    _external_dependency_name_ = 'pysofft'
    @abc.abstractmethod
    def forward_cmplx(self):
        pass
    @abc.abstractmethod
    def inverse_cmplx(self):
        pass

class SoftDependency(SoftInterface,metaclass=DependencyMeta):
    pass
    
class DiscreteLegendreTransform_interface(abc.ABC):
    '''
    Interface for the external library shtns which supplies spherical harmonic transforms
    '''
    _external_dependency_name_ = 'flt'
    @abc.abstractmethod
    def forward():
        pass
    @abc.abstractmethod
    def inverse():
        pass
    
class DiscreteLegendreTransformDependency(DiscreteLegendreTransform_interface,metaclass=DependencyMeta):
    pass

class GSLInterface(abc.ABC):
    '''
    Interface for the Gnu Scientific Library wraper 
    '''
    _external_dependency_name_ = 'gsl'
    @abc.abstractmethod
    def legendre_sphPlm_array(l_max,m_max,xs,return_orders = False,sorted_by_l = False):
        pass
    @abc.abstractmethod
    def bessel_jl(ls,xs):
        pass
    @abc.abstractmethod
    def hyperg_2F1(a,b,c,z):
        pass    
class GSLDependency(GSLInterface,metaclass=DependencyMeta):
    pass
    
class PeakDetectorInterface(abc.ABC):
    '''
    Interface for the Gnu Scientific Library wraper 
    '''
    _external_dependency_name_ = 'persistent_homology'
    @abc.abstractmethod
    def find_peaks(dim,data):
        raise AssertionError('PeakDetector was not jet dependency injected.')

class PeakDetectorDependency(PeakDetectorInterface,metaclass=DependencyMeta):
    pass
