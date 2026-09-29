import numpy as np
import traceback
import logging
from typing import Literal
from scipy.special import jv as bessel_jnu
from scipy.special import loggamma
from scipy.special import spherical_jn as bessel_spherical_jnu
from scipy.special import eval_gegenbauer
from xframe import Multiprocessing
from xframe import settings
from xframe.library.pythonLibrary import xprint
from xframe.library.interfaces import SphericalHarmonicTransformInterface
from xframe.library.mathLibrary import eval_single_ND_zernike_polynomial_gsl,ND_zernike_polynomials,cartesian_to_spherical
from dataclasses import dataclass,field
from typing import ClassVar
from time import time
log = logging.getLogger('root')


class PolarHarmonicTransform:
    '''
    Computes the Circular/Polar Harmonic Transform of datasets given on a regular polar grid.
    For each radial coordinate the Fourier series coefficients are computed along the angular coordinate.
    According to https://en.wikipedia.org/wiki/Fourier_series the fourier series coefficients are normalized by 1/period.
    Therefore we have to set norm=forward in numpy.fft calls to have the same normalization.
    '''
    def __init__(self,max_order=32):
        
        self.max_order = max_order
        self.n_points = 2*max_order        
        self.forward_cmplx = self._generate_forward_cmplx()
        self.forward_cmplx_masked = self._generate_forward_cmplx_masked()
        self.inverse_cmplx = self._generate_inverse_cmplx()
        self.forward_real = self._generate_forward_real()
        self.forward_real_masked = self._generate_forward_real_masked()
        self.inverse_real = self._generate_inverse_real()
        self.phis = np.arange(self.n_points)*2*np.pi/self.n_points
        self.angular_shape = (self.n_points,)

    def _generate_forward_cmplx(self):
        fft = np.fft.fft
        n_points = self.n_points
        def forward_cmplx(data,out = None):
            return fft(data,axis=-1,norm='forward',out = out)
        return forward_cmplx
    def _generate_forward_cmplx_masked(self):
        fft = np.fft.fft
        nsum =np.sum
        div = np.divide
        def forward_complex_masked(data,mask,out = None):
            n_unmasked_points = nsum(mask,axis=-1)
            nonzero = n_unmasked_points!=0
            coeff = fft(data*mask,axis=-1,norm='backward',out = out)
            div(coeff,n_unmasked_points.astype(float),out=coeff,where = nonzero)
            return coeff,nonzero
        return forward_complex_masked
    def _generate_inverse_cmplx(self):
        ifft = np.fft.ifft
        def inverse_cmplx(data,out = None):
            return ifft(data,axis=-1,norm='forward',out = out)
        return inverse_cmplx
    def _generate_forward_real(self):
        fft = np.fft.rfft
        def forward_real(data,out = None):
            return fft(data,axis=-1,norm='forward',out = out)
        return forward_real
    def _generate_forward_real_masked(self):
        fft = np.fft.rfft
        nsum =np.sum
        div = np.divide
        def forward_real_masked(data,mask,out = None):
            n_unmasked_points = nsum(mask,axis=-1)
            nonzero = n_unmasked_points!=0
            coeff = fft(data*mask,axis=-1,norm='backward',out = out)
            div(coeff,n_unmasked_points.astype(float),out=coeff,where = nonzero)
            return coeff,nonzero
        return forward_real_masked
    def _generate_inverse_real(self):
        ifft = np.fft.irfft
        n_points = self.n_points
        def inverse_real(data,n_points=n_points,out = None):
            return ifft(data,n_points,axis=-1,norm='forward',out = out)
        return inverse_real
    def get_empty_coeff(self,pre_shape=None,real=False):
        if real:
            angular_shape=(self.max_order+1,)
        else:
            angular_shape=(self.n_points,)
        if not isinstance(pre_shape,tuple):
            data = np.zeros(angular_shape,dtype=complex)
            return data
        else:
            data = np.zeros(pre_shape+angular_shape,dtype=complex)
            return data
        return np.zeros(self.n_points,complex)


def get_harmonic_transform(bandwidth,dimensions=3,n_phi=0,n_theta=0,options={}):
    #xprint(f"bw = {bandwidth}, dimensions = {dimensions}, options = {options}")
    if dimensions==2:
        return PolarHarmonicTransform(max_order=bandwidth)
    elif dimensions==3:
        keys = ['anti_aliazing_degree']
        shtns_opt = {key:options[key] for key in keys if key in options}
        shtns_opt['n_phi']=n_phi
        shtns_opt['n_theta']=n_theta
        from xframe.library.mathLibrary import shtns
        #xprint(f'shtns opt = {shtns_opt}')
        return shtns.ShSmall(bandwidth,**shtns_opt)

def get_harmonic_transform_from_array(array):
    #xprint(f"bw = {bandwidth}, dimensions = {dimensions}, options = {options}")
    shape = array.shape
    dimension = array.ndim
    if dimension ==2:
        bandwidth = shape[1]//2
        opt = {'n_angular_points':shape[1]}
    else:
        bandwidth = shape[1]
        opt = {'n_phi':shape[2],'n_theta':shape[1]}
    from xframe.library.mathLibrary import shtns
    return shtns.ShSmall(bandwidth,**opt)



    
def polar_spherical_dft_reciprocity_relation_radial_cutoffs(cutoff:float,n_points:int,reciprocity_coefficient: float =np.pi) -> float:
    '''
    Reciprocity relation between the real and reciprocal cutoff.
    Reciprocity coefficient of Pi corresponds to the usual FFT relation for [2pi/Angstrom] units in reciprocal and [Angstrom] in reals space.
    '''
    other_cutoff = reciprocity_coefficient*n_points/cutoff    
    return other_cutoff


@dataclass
class HankelWeightStruct:
    ''' Data class that specifies input variables for the generation of Hankel transform weights '''
    supported_dimensions : ClassVar[tuple[int]] = (2,3)
    supported_hankel_types : ClassVar[tuple[str]] = ['trapz','zernike_trapz','sincos_trapz','midpoint','zernike_midpoint','sincos_midpoint']

    dimension: int = 3
    n_radial_points: tuple[int,int] = (64,64)
    angular_bandwidth: int = 32
    hankel_type: str = 'midpoint'
    n_processes_for_weight_generation: int|None =  None
    reciprocity_coefficient: float = np.pi
    other: dict = field(default_factory=dict)

    @property
    def weight_name(self):
        s = self
        return f'{s.dimension}D_NrNq{s.n_radial_points}_bandwith{s.angular_bandwidth}_{s.hankel_type}_rc{s.reciprocity_coefficient}'
        
class HankelTransformWeights:
    ht_modes = HankelWeightStruct.supported_hankel_types
    @staticmethod
    def _reciprocity_relation(Nq,reciprocity_coefficient,Q_or_R):
        R_or_Q =  reciprocity_coefficient*Nq/Q_or_R
        return R_or_Q
    @staticmethod
    def _read_n_points(n_points):
        if isinstance(n_points,(list,tuple)):
            Nr,Nq = n_points
        else:
            Nr = Nq = n_points
        return (Nr,Nq)
    @staticmethod
    def _read_weights(weights):
        if isinstance(weights,(list,tuple)):
            weights_forward,weights_inverse = weights
        else:
            weights_forward = weights_inverse = weights
        return (weights_forward,weights_inverse)
    @staticmethod
    def _apply_prefactors_and_reshape_weights(weights,prefactors):
        w_forward=np.moveaxis(weights[0],0,2)
        w_inverse=np.moveaxis(weights[1],0,2)
        forward_weights=w_forward*prefactors[0]
        inverse_weights=w_inverse*prefactors[1]
        return (forward_weights,inverse_weights)
    ### Direct Midpoint ###
    @classmethod
    def midpoint(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        xprint(f'calculating weights for bandwidth = {angular_bandwidth}')
        if dimensions==3:
            orders = np.arange(angular_bandwidth)
        elif dimensions ==2:
            orders = np.arange(angular_bandwidth+1)
        worker_by_dimensions= {2:cls.midpoint_polar_worker,3:cls.midpoint_spherical_worker}
        worker = worker_by_dimensions[dimensions]
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_processes_for_weight_generation)
        forward_weights = np.concatenate([part[0] for part in weights],axis =0 )
        inverse_weights = np.concatenate([part[1] for part in weights],axis = 0)
        return (forward_weights,inverse_weights)
    @classmethod
    def midpoint_polar_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates polar midpoint rule weights.
        '''
        Nr,Nq = cls._read_n_points(n_points)
        ps=np.arange(Nr)+1/2
        ks=np.arange(Nq)+1/2
        ms = orders
        Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],Nr,axis=1),Nq,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/Nr)        
        weights_forward = ps[None,:,None]*Jmpk
        weights_inverse = ks[None,:,None]*np.swapaxes(Jmpk,-1,-2)
        return (weights_forward,weights_inverse)

    @classmethod
    def midpoint_spherical_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates spherical midpoint rule weights.
        '''
        Nr,Nq = cls._read_n_points(n_points)
        ps=np.arange(Nr)+1/2
        ks=np.arange(Nq)+1/2
        ls = orders
        jmpk = bessel_spherical_jnu(np.repeat(np.repeat(ls[:,None,None],Nr,axis=1),Nq,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/Nr)
        weights_forward = ps[None,:,None]**2*jmpk
        weights_inverse = ks[None,:,None]**2*np.swapaxes(jmpk,-1,-2)
        return (weights_forward,weights_inverse)

    @classmethod
    def assemble_midpoint(cls,weights,bandwidth,r_max,reciprocity_coefficient,dimensions=3):
        '''
        Generates weights for the forward and inverse transform by multiplieing with the propper constants.
        And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
        (summation_radial_coordinate,new_radial_coordinate,order)
        '''
        w_forward,w_inverse= cls._read_weights(weights)
        Nr,Nq = w_forward.shape[1:]        
        q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,Nq,reciprocity_coefficient = reciprocity_coefficient)
        #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
        #xprint(f'alignment dimensions = {dimensions}')
        if dimensions == 2:
            orders = np.arange(bandwidth+1)
            all_orders = np.concatenate((orders,-orders[-2:0:-1]))
            #log.info(all_orders)
            forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/Nr)**2#*np.sqrt(2)
            inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/Nq)**2#*np.sqrt(2)
            # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
            #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
            w_forward = np.concatenate((w_forward,(-1)**orders[-2:0:-1,None,None]*w_forward[-2:0:-1]),axis = 0)
            w_inverse = np.concatenate((w_inverse,(-1)**orders[-2:0:-1,None,None]*w_inverse[-2:0:-1]),axis = 0)
        elif dimensions == 3:
            #print(f'Nr = Nq = {n_radial_points} r_max = {r_max} q_max = {q_max}')
            orders = np.arange(bandwidth)
            forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/Nr)**3*np.sqrt(2/np.pi)
            inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/Nq)**3*np.sqrt(2/np.pi)
            
        forward_weights,inverse_weights = cls._apply_prefactors_and_reshape_weights((w_forward,w_inverse),(forward_prefactor,inverse_prefactor))
        #log.info('weights shape = {}'.format(forward_weights.shape))
        real_points = (np.arange(Nr)+0.5)*r_max/Nr
        reciprocal_points = (np.arange(Nq)+0.5)*q_max/Nq
        #print(f'R = {real_points.max()}')
        return {'forward':forward_weights,'inverse':inverse_weights,'real_points':real_points,'reciprocal_points':reciprocal_points}

    ### Direct Chebyshev, bad approximation dont use this ###
    @staticmethod
    def chebyshev_nodes(n_points,start=-1,end=1):
        N = n_points
        a = start
        b = end
        return (a+b)/2 + (a-b)/2 * np.cos((np.arange(N,dtype=float)+0.5)*np.pi/N)
    @classmethod
    def chebyshev(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        if dimensions==3:
            orders = np.arange(angular_bandwidth)
        elif dimensions ==2:
            orders = np.arange(angular_bandwidth+1)
        worker_by_dimensions= {2:cls.chebyshev_polar_worker,3:cls.chebyshev_spherical_worker}
        worker = worker_by_dimensions[dimensions]
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_processes_for_weight_generation)
        forward_weights = np.concatenate([part[0] for part in weights],axis =0 )
        inverse_weights = np.concatenate([part[1] for part in weights],axis = 0)
        return (forward_weights,inverse_weights)
    @classmethod
    def chebyshev_polar_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates polar chebyshev rule weights.
        '''
        Nr,Nq = cls._read_n_points(n_points)
        ps = cls.chebyshev_nodes(Nr,start=0)
        ks = cls.chebyshev_nodes(Nq,start=0)
        ls = orders
        Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],Nr,axis=1),Nq,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*Nq)
        weights_forward = ps[None,:,None]**2*np.sin((np.arange(Nr)+0.5)*np.pi/Nr)[None,:,None]*Jmpk*np.pi/2
        weights_inverse = ks[None,:,None]**2*np.sin((np.arange(Nq)+0.5)*np.pi/Nq)[None,:,None]*np.swapaxes(Jmpk,-1,-2)*np.pi/2
        return (weights_forward,weights_inverse)
    @classmethod
    def chebyshev_spherical_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates spherical chebyshev rule weights.
        '''
        Nr,Nq = cls._read_n_points(n_points)
        ps = cls.chebyshev_nodes(Nr,start=0)
        ks = cls.chebyshev_nodes(Nq,start=0)
        ls = orders
        jmpk = bessel_spherical_jnu(np.repeat(np.repeat(ls[:,None,None],Nr,axis=1),Nq,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*Nq)
        weights_forward = ps[None,:,None]**2*np.sin((np.arange(Nr)+0.5)*np.pi/Nr)[None,:,None]*jmpk*np.pi/2
        weights_inverse = ks[None,:,None]**2*np.sin((np.arange(Nq)+0.5)*np.pi/Nq)[None,:,None]*np.swapaxes(jmpk,-1,-2)*np.pi/2
        return (weights_forward,weights_inverse)
    @classmethod
    def assemble_chebyshev(cls,weights,bandwidth,r_max,reciprocity_coefficient,dimensions=3):
        '''
        Generates weights for the forward and inverse transform by multiplieing with the propper constants.
        And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
        (summation_radial_coordinate,new_radial_coordinate,order)
        '''
        w_forward,w_inverse= cls._read_weights(weights)
        Nr,Nq = w_forward.shape[1:]
        q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,Nq,reciprocity_coefficient = reciprocity_coefficient)
        #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
        #xprint(f'alignment dimensions = {dimensions}')
        if dimensions == 2:
            orders = np.arange(bandwidth+1)
            all_orders = np.concatenate((orders,-orders[-2:0:-1]))
            #log.info(all_orders)
            forward_prefactor = (-1.j)**(all_orders[None,None,:])*r_max**2/Nr
            inverse_prefactor = (1.j)**(all_orders[None,None,:])*q_max**2/Nq
            w_forward = np.concatenate((w_forward,(-1)**orders[-2:0:-1,None,None]*w_forward[-2:0:-1]),axis = 0)
            w_inverse = np.concatenate((w_inverse,(-1)**orders[-2:0:-1,None,None]*w_inverse[-2:0:-1]),axis = 0)
        elif dimensions == 3:
            orders = np.arange(bandwidth)
            forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max)**3*np.sqrt(2/np.pi)/Nr
            inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max)**3*np.sqrt(2/np.pi)/Nq
            
        forward_weights,inverse_weights = cls._apply_prefactors_and_reshape_weights((w_forward,w_inverse),(forward_prefactor,inverse_prefactor))
        real_points = chebyshev_nodes(Nr,start=0,end=r_max)        
        reciprocal_points = chebyshev_nodes(Nq,start=0,end=q_max)
        #log.info('weights shape = {}'.format(forward_weights.shape))
        return {'forward':forward_weights,'inverse':inverse_weights,'real_points':real_points,'reciprocal_points':reciprocal_points}


    ### Direct Trapezoidal ###
    @classmethod
    def trapz(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        pass
    ### Zernike Midpoint ###
    @classmethod
    def zernike_midpoint(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        pass
    ### Zernike Trapezoidal ###
    @classmethod
    def zernike_trapz(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        pass
    ### SinCos Midpoint ###
    @classmethod
    def sincos_midpoint(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        pass
    ### SinCos Trapezoidal ###
    @classmethod
    def sincos_trapz(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        pass


    @classmethod
    def get_weights_dict(cls,struct: HankelWeightStruct = HankelWeightStruct()):
        temp = struct.__dict__        
        weight_generator = getattr(cls,struct.hankel_type)
        weights = weight_generator(struct.n_radial_points,struct.angular_bandwidth,struct.dimension,struct.reciprocity_coefficient,struct.n_processes_for_weight_generation,**struct.other)
        temp = {"weights":weights}
        return temp

    @classmethod
    def get_weights(cls,struct: HankelWeightStruct = HankelWeightStruct()):
        temp = struct.__dict__        
        weight_generator = getattr(cls,struct.hankel_type)
        weights = weight_generator(struct.n_radial_points,struct.angular_bandwidth,struct.dimension,struct.reciprocity_coefficient,struct.n_processes_for_weight_generation,**struct.other)
        return weights

    
@dataclass
class HankelTransformStruct(HankelWeightStruct):
    ''' Data class that specifies input variables of HankelTransform class'''
    max_q: float = None # default float value set in post_init 
    max_nonzero_r: float = None # default float value set in post_init 
    use_gpu:bool = True
    def __post_init__(self):
        if self.max_q is None:
            self.max_q = polar_spherical_dft_reciprocity_relation_radial_cutoffs(1.0,self.n_radial_points[1])
        if self.max_nonzero_r is None:
            self.max_nonzero_r = polar_spherical_dft_reciprocity_relation_radial_cutoffs(self.max_q,self.n_radial_points[1])
    @property
    def max_r(self) -> float:
        return polar_spherical_dft_reciprocity_relation_radial_cutoffs(self.max_q,self.n_radial_points[1])
    
class HankelTransform:
    ht_modes = ['trapz','zernike_trapz','sincos_trapz','midpoint','zernike_midpoint','sincos_midpoint']

    def __init__(self,struct: HankelTransformStruct  = HankelTransformStruct(),weights: np.ndarray|list|tuple|None = None):
        self.struct = struct
        #xprint(f'struct2 = {struct.__dict__}')
        self.use_gpu = struct.use_gpu
        self.dimensions = struct.dimension
        self.mode = struct.hankel_type
        self.Nr,self.Nq = HankelTransformWeights._read_n_points(struct.n_radial_points)
        self.n_points = struct.n_radial_points
        self.bandwidth = struct.angular_bandwidth
        self.q_max = struct.max_q
        self.r_max = struct.max_r
        self.r_support = struct.max_nonzero_r
        #xprint(f'current process id = {Multiprocessing.get_process_name()}')
        if struct.use_gpu and Multiprocessing.get_process_name()==0:
            comm = Multiprocessing.comm_module
            if not comm.are_control_workers_running():
                settings.general.n_control_workers = 1
                comm.restart_control_worker()
        if not isinstance(weights,(np.ndarray,list,tuple)):            
            weight_generator = getattr(HankelTransformWeights,self.mode)
            weights = weight_generator(self.n_points,self.bandwidth,self.dimensions,struct.reciprocity_coefficient,struct.n_processes_for_weight_generation,**struct.other)
        self.assembled_weights = getattr(HankelTransformWeights,'assemble_'+self.mode)(weights,self.bandwidth,self.r_max,np.pi,dimensions=self.dimensions)
        self.grids = {'real':self.assembled_weights.pop('real_points'),'reciprocal':self.assembled_weights.pop('reciprocal_points')}
        if self.r_support<self.r_max:
            max_id = np.argmin(self.grids['real']<self.r_support)
            self.Nr = max_id
            self.grids['real']=self.grids['real'][:max_id]
            fw = self.assembled_weights['forward'][:max_id]
            iw = self.assembled_weights['inverse'][:,:max_id]
            self.assembled_weights['forward'] = fw
            self.assembled_weights['inverse'] = iw
        if 'pinv' in struct.other:
            #fw = self.assembled_weights['forward']            
            #self.assembled_weights['inverse'] = np.moveaxis(np.array([np.linalg.pinv(fw[...,i]) for i in range(fw.shape[-1])]),0,-1)
            iw = self.assembled_weights['inverse']
            self.assembled_weights['forward'] = np.moveaxis(np.array([np.linalg.pinv(iw[...,i]) for i in range(iw.shape[-1])]),0,-1)
            #print(f'iw shape = {iw.shape}')
        
        self._forward_coeff,self._inverse_coeff = self._generate_coeff_arrays()
        self.forward_cmplx,self.inverse_cmplx = self._generate_ht()

    @classmethod
    def from_kwargs(cls,**kwargs):
        struct = HankelTransformStruct(**kwargs)
        return cls(struct)
    def _generate_coeff_arrays(self):
        if self.dimensions==2:
            coeffs = self._generate_polar_coeff_arrays()
        elif self.dimensions==3:
            coeffs = self._generate_spherical_coeff_arrays()
        return coeffs
    def _generate_spherical_coeff_arrays(self):
        from xframe.library.mathLibrary import shtns
        coeff_shape_f = (self.Nq,self.bandwidth**2)
        coeff_shape_i = (self.Nr,self.bandwidth**2)
        forward_array = np.zeros(coeff_shape_f,dtype=complex)
        inverse_array = np.zeros(coeff_shape_i,dtype=complex)
        forward_coeff = shtns.ShCoeff.from_bandwith_complex(forward_array,self.bandwidth)
        inverse_coeff = shtns.ShCoeff.from_bandwith_complex(inverse_array,self.bandwidth)
        return forward_coeff,inverse_coeff
    def _generate_polar_coeff_arrays(self):
        coeff_shape_f = (self.Nq,2*self.bandwidth)
        coeff_shape_i = (self.Nr,2*self.bandwidth)
        forward_array = np.zeros(coeff_shape_f,dtype=complex)
        inverse_array = np.zeros(coeff_shape_i,dtype=complex)
        return forward_array,inverse_array

    def _generate_ht(self):
        dimensions = self.dimensions
        use_gpu = self.use_gpu
        spherical_gpu_version = (dimensions == 3) and use_gpu
        spherical_version = (dimensions == 3) and (not use_gpu)
        polar_version = (dimensions == 2)  and (not use_gpu)
        polar_gpu_version = (dimensions == 2) and use_gpu
        
        if spherical_gpu_version:
            zht,izht = self._generate_spherical_ht_gpu()
        elif spherical_version:
            zht,izht = self._generate_spherical_ht()
        elif polar_version:
            zht,izht = self._generate_polar_ht()
        elif polar_gpu_version:
            zht,izht = self._generate_polar_ht_gpu()
        return zht,izht
        
    
    def _generate_spherical_ht(self):
        from xframe.library.mathLibrary import shtns
        ShCoeff = shtns.ShCoeff
        fw= np.swapaxes(self.assembled_weights['forward'],0,1)
        iw= np.swapaxes(self.assembled_weights['inverse'],0,1)
        #fw= self.assembled_weights['forward']
        #iw= self.assembled_weights['inverse']
        l_orders = np.arange(self.bandwidth)
        l_max = self.bandwidth-1
        #m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
        matmul = np.matmul
        forward_coeff = self._forward_coeff
        inverse_coeff = self._inverse_coeff
        if 'trapz' in self.mode:
            def ht(harmonic_coeff):
                #return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
                for l in l_orders:
                    matmul(fw[:,:,l],harmonic_coeff.lm[l][1:],out = forward_coeff.lm[l])
                return forward_coeff
            def iht(harmonic_coeff):
                for l in l_orders:
                    matmul(iw[:,:,l],harmonic_coeff.lm[l][1:],out = inverse_coeff.lm[l])
                return inverse_coeff               
        else:
            def ht(harmonic_coeff):
                #print(f'in shape = {harmonic_coeff.shape}')
                #print(f'fw shape = {fw.shape}')
                #print(f'fw shape = {fw.shape}')
                for l in l_orders:
                    matmul(fw[:,:,l],harmonic_coeff.lm[l],out = forward_coeff.lm[l])
                return forward_coeff        
            def iht(harmonic_coeff):
                #print(f'in shape = {harmonic_coeff.shape}')
                #print(f'iw shape = {iw.shape}')
                #print(f'inverse_coeff shape = {inverse_coeff.shape}')
                for l in l_orders:
                    matmul(iw[:,:,l],harmonic_coeff.lm[l],out = inverse_coeff.lm[l])
                return inverse_coeff
        return ht,iht

    def _generate_spherical_ht_gpu(self):
        forward_weights = self.assembled_weights['forward']
        inverse_weights = self.assembled_weights['inverse']
        l_orders = np.arange(self.bandwidth)
        n_radial_points = forward_weights.shape[1]    
        l_max=l_orders.max()    
        nq = self.Nq
        nr = self.Nr
        nl = len(l_orders)
        nlm = self.bandwidth**2
        if 'trapz' in self.mode:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out,
            __global double2* w,
            __global double2* rho,
            long nr,long nq,long nlm, long nl)
            {
      
            long i = get_global_id(0);
            long j = get_global_id(1);
            long l = (long) sqrt((double)j);
    
    
            // value stores the element that is
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int r = 0; r < nr-1; ++r)
            {
            double2 wqql = w[r*nq*nl + i*nl + l];
            double2 rqlm = rho[(r+1)*nlm + j];
            value.x += wqql.x * rqlm.x - wqql.y * rqlm.y;
            value.y += wqql.x * rqlm.y + wqql.y * rqlm.x;
            }
        
            // Write the matrix to device memory each
            // thread writes one element
            out[i * nlm + j] = value;//w[nq*nl+i*nl + l];
            }
            """
        else:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out,
            __global double2* w,
            __global double2* rho,
            long nr,long nq,long nlm, long nl)
            {
      
            long i = get_global_id(0);
            long j = get_global_id(1);
            long l = (long) sqrt((double)j);
    
    
            // value stores the element that is
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int r = 0; r < nr; ++r)
            {
            double2 wqql = w[r*nq*nl + i*nl + l];
            double2 rqlm = rho[r*nlm + j];
            value.x += wqql.x * rqlm.x - wqql.y * rqlm.y;
            value.y += wqql.x * rqlm.y + wqql.y * rqlm.x;
            }
        
            // Write the matrix to device memory each
            // thread writes one element
            out[i * nlm + j] = value;//w[nq*nl+i*nl + l];
            }
            """
        
        kernel_dict_forward={
                'kernel': kernel_str,
                'name': 'forward_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64,np.int64),
                    'shapes' : ((nq,nlm),forward_weights.shape,(nr,nlm),None,None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input','const_input'),
                    'const_inputs' : (None,forward_weights,None,np.int64(nr),np.int64(nq),np.int64(nlm),np.int64(nl)),
                    'global_range' : (nq,nlm),
                    'local_range' : None,                    
                },)
            }
        
        kernel_dict_inverse={
                'kernel': kernel_str,
                'name': 'inverse_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64,np.int64),
                    'shapes' : ((nr,nlm),inverse_weights.shape,(nq,nlm),None,None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input','const_input'),
                    'const_inputs' : (None,inverse_weights,None,np.int64(nq),np.int64(nr),np.int64(nlm),np.int64(nl)),
                    'global_range' : (nr,nlm),
                    'local_range' : None
                },)
            }
        forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
        inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

        forward_coeff=(self._forward_coeff,)
        inverse_coeff=(self._inverse_coeff,)
        hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process,local_outputs=forward_coeff)
        inverse_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process,local_outputs=inverse_coeff)
            
        return hankel_transform,inverse_hankel_transform
    
    def _generate_polar_ht(self):
        '''
        Generates polar hankel transform using Zernike weights. $HT_m(f_m) = \sum_k f_m(k)*w_kpm$ for $m\geq 0$.
        '''
        fw= self.assembled_weights['forward']
        iw= self.assembled_weights['inverse']
        l_orders = np.arange(self.bandwidth)
        l_max = self.bandwidth-1
        #n_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
        matmul = np.matmul

        n_radial_points=fw.shape[0]
        full_shape = (n_radial_points,2*l_max)
        out_forward = self._forward_coeff
        out_inverse = self._inverse_coeff
        if 'trapz' in self.mode:
            def ht(harmonic_coeff):
                #log.info('harmxoonic shape = {}'.format(harmonic_coeff.shape))
                #log.info('out forward = {}'.format(out_forward.shape))
                #reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[1:,None,:],axis = 0,out=out_forward)
                np.sum(fw*harmonic_coeff[1:,None,:],axis=0,out=out_forward)
                return out_forward
            def iht(harmonic_coeff):
                np.sum(iw*harmonic_coeff[1:,None,:],axis=0,out = out_inverse) #iw@harmonic_coeff[1:]
                #harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[1:,None,:],axis = 0,out=out_inverse)
                #harmonic_coeff[:,unused_order_mask]=0
                return out_inverse
        else:
            def ht(harmonic_coeff):
                #out_forward[:]=fw@harmonic_coeff
                np.sum(fw*harmonic_coeff[:,None,:],axis=0,out=out_forward)
                #log.info('harmonic shape = {}'.format(harmonic_coeff.shape))
                #log.info('out forward = {}'.format(out_forward.shape))
                #reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[:,None,:],axis = 0,out=out_forward)
                #reciprocal_harmonic_coeff[:,unused_order_mask]=0
                return out_forward
            def iht(harmonic_coeff):
                #out_inverse[:]=iw@harmonic_coeff
                np.sum(iw*harmonic_coeff[:,None,:],axis=0,out = out_inverse)
                #harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[:,None,:],axis = 0,out=out_inverse)
                #harmonic_coeff[:,unused_order_mask]=0
                return out_inverse
        return ht,iht

    def _generate_polar_ht_gpu(self):
        forward_weights = self.assembled_weights['forward']
        inverse_weights = self.assembled_weights['inverse']
        n_radial_points = forward_weights.shape[1]

        m_max=self.bandwidth
        nq = self.Nq #n_radial_points
        nr = self.Nr
        nm = m_max*2
    
        if 'trapz' in self.mode:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out, 
            __global double2* w, 
            __global double2* rho, 
            long nr,long nq,long nm)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int r = 0; r < nr-1; ++r)
            {
            double2 wqqm = w[r*nq*nm + i*nm + j];
            double2 rqm = rho[(r+1)*nm + j];
            value.x += wqqm.x * rqm.x - wqqm.y * rqm.y;
            value.y += wqqm.x * rqm.y + wqqm.y * rqm.x;
            }
        
            // Write the matrix to device memory each 
            // thread writes one element
            out[i * nm + j] = value;//w[nq*nl+i*nl + l];
            }
            """
        else:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out, 
            __global double2* w, 
            __global double2* rho, 
            long nr,long nq,long nm)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int r = 0; r < nr; ++r)
            {
            double2 wqqm = w[r*nq*nm + i*nm + j];
            double2 rqm = rho[r*nm + j];
            value.x += wqqm.x * rqm.x - wqqm.y * rqm.y;
            value.y += wqqm.x * rqm.y + wqqm.y * rqm.x;
            }
        
            // Write the matrix to device memory each 
            // thread writes one element
            out[i * nm + j] = value;//w[nq*nl+i*nl + l];
            }
            """
    
        kernel_dict_forward={
                'kernel': kernel_str,
                'name': 'forward_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                    'shapes' : ((nq,nm),forward_weights.shape,(nr,nm),None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                    'const_inputs' : (None,forward_weights,None,np.int64(nr),np.int64(nq),np.int64(nm)),
                    'global_range' : (nq,nm),
                    'local_range' : None
                },)
            }
        
        kernel_dict_inverse={
                'kernel': kernel_str,
                'name': 'inverse_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                    'shapes' : ((nr,nm),inverse_weights.shape,(nq,nm),None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                    'const_inputs' : (None,inverse_weights,None,np.int64(nq),np.int64(nr),np.int64(nm)),
                    'global_range' : (nr,nm),
                    'local_range' : None
                },)
            }
    
        forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
        inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

        forward_coeff=(self._forward_coeff,)
        inverse_coeff=(self._inverse_coeff,)
        hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process,local_outputs=forward_coeff)
        inverse_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process,local_outputs=inverse_coeff)
        return hankel_transform,inverse_hankel_transform


    @property
    def real_radial_points(self):        
        return self.grids['real']
    @property
    def reziprocal_radial_points(self):
        return self.grids['reciprocal']




@dataclass
class SphericalFourierTransformStruct(HankelTransformStruct):
    ''' Data class that specifies input variables of SphericalFourierTransform class'''
    n_azimutal_angles: int = 0 #0 leaves the decission to shtns (azimutal only used for dimension = 3)
    n_polar_angles: int = 0 #0 leaves the decission to shtns

        
class SphericalFourierTransform:
    '''
    Class that provides Fourier transforms of complex valued functions in polar and spherical coordinates.
    '''
    def __init__(self,struct : SphericalFourierTransformStruct = SphericalFourierTransformStruct(), weights = None):
        self.struct = struct
        if weights is not None:
            self.ht = HankelTransform(struct,weights = weights)
        else:
            #xprint(f'struct = {struct.__dict__}')
            self.ht = HankelTransform(struct)
        self.harm = get_harmonic_transform(struct.angular_bandwidth,dimensions=struct.dimension,n_phi = struct.n_polar_angles,n_theta = struct.n_azimutal_angles,options=struct.other)
        #xprint(f'angular shape = {self.harm.angular_shape}')
        self._init_end()
        self.reciprocal_grid_in_cartesian_coords = None
        self._shift = None
        
    def _init_end(self):
        self.dimensions=self.ht.dimensions
        self.forward_cmplx,self.inverse_cmplx = self._generate_transforms()
        
    @classmethod
    def from_transforms(cls,hankel_transform,harmonic_transform):
        self.ht = hankel_transform
        self.harm = harmonic_transform
        self._init_end()
        return self
    @classmethod
    def from_kwargs(cls,**kwargs):
        struct = SphericalFourierTransformStruct(**kwargs)
        return cls(struct)

    
    def _generate_transforms(self):
        harm_forward_cmplx = self.harm.forward_cmplx
        harm_inverse_cmplx = self.harm.inverse_cmplx
        hankel_forward_cmplx = self.ht.forward_cmplx
        hankel_inverse_cmplx = self.ht.inverse_cmplx
        def forward_cmplx(data):
            coeff = harm_forward_cmplx(data)
            rcoeff = hankel_forward_cmplx(coeff)
            return harm_inverse_cmplx(rcoeff)
        def inverse_cmplx(data):
            rcoeff = harm_forward_cmplx(data)
            coeff = hankel_inverse_cmplx(rcoeff)
            #temp = harm_inverse_cmplx(coeff)
            return harm_inverse_cmplx(coeff)
        return forward_cmplx,inverse_cmplx
    def empty_density(self):
        angular_shape = self.harm.angular_shape
        radial_shape = (self.ht.Nr,)
        return np.zeros(radial_shape+angular_shape,dtype=complex)
    @property
    def real_grid(self):
        rs = self.ht.grids["real"]
        phis = self.harm.phis
        if self.dimensions==2:
            grid = np.stack(np.meshgrid(rs,phis,indexing='ij'),2)
        elif self.dimensions == 3:
            thetas = self.harm.thetas
            grid = np.stack(np.meshgrid(rs,thetas,phis,indexing='ij'),3)
        return grid
    @property
    def qs(self):
        return self.ht.grids["reciprocal"]
    @property
    def rs(self):
        return self.ht.grids["real"]
    @property
    def phis(self):
        return self.harm.phis
    @property
    def reciprocal_grid(self):
        qs = self.ht.grids["reciprocal"]
        phis = self.harm.phis
        if self.dimensions==2:
            grid = np.stack(np.meshgrid(qs,phis,indexing='ij'),2)
        elif self.dimensions == 3:
            thetas = self.harm.thetas
            grid = np.stack(np.meshgrid(qs,thetas,phis,indexing='ij'),3)
        return grid

    @property
    def shift(self):
        if self._shift is None:
            self._shift = self._generate_shift()
        return self._shift
    
    def _generate_shift(self):
        from xframe.library.mathLibrary import spherical_to_cartesian        
        if self.reciprocal_grid_in_cartesian_coords is None:
            self.reciprocal_grid_in_cartesian_coords = spherical_to_cartesian(self.reciprocal_grid)
        cart_grid = self.reciprocal_grid_in_cartesian_coords
        def shift(reciprocal_density,vector,
                  opposite_direction=False,
                  power = 1,
                  center_coord_sys:Literal['spherical','cartesian']='spherical'):
            prefactor = -1.j*(-1)**opposite_direction*power
            if center_coord_sys == 'spherical':
                cart_vect = spherical_to_cartesian(vector)
            else:
                cart_vect = vector
            phases = np.exp(prefactor*(cart_grid*cart_vect).sum(axis=-1))
            reciprocal_density*=phases
            return reciprocal_density
        return shift

@dataclass
class SphericalZernikeTransformStruct:
    bandwidth:int = 32
    n_points:int = 128
    max_r:float = 1.0
    grid_type:str = 'midpoint'
    
class SphericalZernikeTransform:
    def __init__(self,
                 struct:SphericalZernikeTransformStruct= None,
                 harmonic_transform:SphericalHarmonicTransformInterface = None):
        if struct is None:
            struct = SphericalZernikeTransformStruct()
        if harmonic_transform is None:
            harmonic_transform = get_harmonic_transform(32)            
        self.struct = struct
        self.harm = harmonic_transform
        self._dim = 3
        self.weights,self.rs = self.generate_weights()
        
    def generate_weights(self):
        Nr = self.struct.n_points
        max_r = self.struct.max_r
        angular_bw = self.harm.bandwidth
        bw = self.struct.bandwidth
        if self.struct.grid_type != 'midpoint':
            raise NotImplementedError(f'Wrong grid type {self.struct.grid_tpye}. Currently only grid_type = "midpoint" is supported.' )
        points = (np.arange(Nr)+0.5)/Nr
        rs = points*max_r
        weights = np.zeros((angular_bw,bw,Nr),dtype=float)
        
        for l in range(min(angular_bw,bw)):
            weights[l,l::2,:] = ND_zernike_polynomials(l,bw,points,self._dim)
        return weights,rs
    
    def forward(self,lm_coeff):
        out = self.harm.get_empty_coeff(pre_shape=(self.struct.bandwidth,),
                                        complex_data=lm_coeff.complex_data,
                                        real_harmonics= np.issubdtype(lm_coeff.dtype,np.floating))
        max_r = self.struct.max_r
        r2 = self.rs**(self._dim-1)
        delta_r = self.rs[1]-self.rs[0]
        bw = min(self.harm.bandwidth,self.struct.bandwidth)
        for l in range(bw):
            out_l = out.lm[l]
            for s in range(self.struct.bandwidth):
                const = ((2*s+self._dim)/max_r**self._dim)*delta_r
                out_l[s] = const*np.sum(lm_coeff.lm[l].T*r2[None,:]*self.weights[l,s][None,:],axis=-1)
            out.lm[l] = out_l
        return out
    
    def inverse(self,lms_coeff):
        out = self.harm.get_empty_coeff(pre_shape=(self.struct.n_points,),
                                        complex_data=lms_coeff.complex_data,
                                        real_harmonics= np.issubdtype(lms_coeff.dtype,np.floating))
        bw = min(self.harm.bandwidth,self.struct.bandwidth)
        for l in range(bw):
            out.lm[l] = np.sum(lms_coeff.lm[l][:,None,:]*self.weights[l,:,:,None],axis=0)
        return out

    def full_inverse_at(self, lms_coeff, new_points):
        """
        Evaluate the spherical-Zernike expansion at arbitrary points.

        Parameters
        ----------
        lms_coeff : ShCoeff
            Zernike coefficients with radial/Zernike index s as first axis.
        new_points : ndarray, shape (..., 3)
            Points in spherical coordinates: (r, theta, phi).
        """
        new_points = np.asarray(new_points)
        
        if new_points.shape[-1] != 3:
            raise ValueError("new_points must have shape (..., 3).")
        
        if lms_coeff.shape[0] != self.struct.bandwidth:
            raise ValueError("Unexpected radial/Zernike coefficient dimension.")
        
        if (not lms_coeff.complex_data
            and np.issubdtype(lms_coeff.dtype, np.floating)):
            raise NotImplementedError(
                "Full real-harmonic coefficient representation is not supported."
            )
        
        output_shape = new_points.shape[:-1]
        r, theta, phi = new_points.reshape(-1, 3).T
        
        if np.any(r < 0) or np.any(r > self.struct.max_r):
            raise ValueError("Points must satisfy 0 <= r <= max_r.")
        
        if np.any(theta < 0) or np.any(theta > np.pi):
            raise ValueError("theta must lie in [0, pi].")
        
        rho = r / self.struct.max_r
        phi = np.mod(phi, 2 * np.pi)
        cos_theta = np.cos(theta)
        # One SHTns angular-coefficient vector for every requested point.
        angular_coeffs = self.harm.get_empty_coeff(pre_shape=(len(rho),),complex_data = lms_coeff.complex_data)
        bw = min(self.harm.bandwidth, self.struct.bandwidth)
        #print('Zernike inverse')
        for l in range(bw):
            valid_s = np.arange(l, self.struct.bandwidth, 2)
            coeff_part = lms_coeff.lm[l][valid_s]
            # Shape: (n_points, n_valid_s)
            start = time()
            radial_basis = ND_zernike_polynomials(l,self.struct.bandwidth,rho,self._dim)
            #print(f'Basis creation time ={time()-start} basis shape = {radial_basis.shape} coeff_part.shape = {coeff_part.shape}')
            # Radial synthesis:
            # f_lm(r) = sum_s a_slm R_sl(r / max_r)
            start = time()
            angular_coeffs.lm[l] = radial_basis.T @ coeff_part
            #print(f'Matrix mult time ={time()-start}')
            
        if angular_coeffs.complex_data:
            eval_sh = self.harm._sh.SH_to_point_cplx
        else:
            eval_sh = self.harm._sh.SH_to_point
            
        #print('sh inverse')
        values = np.array([
            eval_sh(coeff, ct, p)
            for coeff, ct, p in zip(angular_coeffs, cos_theta, phi)
        ])
        return values.reshape(output_shape)

    def _get_cart_points(self,start,stop,points):
        '''
        computes cartesian grid points based on a 1d index
        and the list of points along one dimension.
        '''
        n = len(points)
        if stop>n**3:
            stop = n**3
        ids = np.arange(start,stop)
            
        out = np.zeros((len(ids),3),float)
        out[:,0] = points[(ids//n**2)%n]
        out[:,1] = points[(ids//n)%n]
        out[:,2] = points[ids%n]
        return out

    def _full_inverse_cartesian_single_process(self, lms_coeff, points,batch_size=8192,fill_value=0):
        """
        Evaluate the spherical-Zernike expansion at arbitrary points.

        Parameters
        ----------
        lms_coeff : ShCoeff
            Zernike coefficients with radial/Zernike index s as first axis.
        points : ndarray, shape (Np)
            Points in spherical coordinates: (r, theta, phi).
        """
        points = np.asarray(points)
        n = len(points)
        
        if lms_coeff.shape[0] != self.struct.bandwidth:
            raise ValueError("Unexpected radial/Zernike coefficient dimension.")
        
        if (not lms_coeff.complex_data
            and np.issubdtype(lms_coeff.dtype, np.floating)):
            raise NotImplementedError(
                "Full real-harmonic coefficient representation is not supported."
            )

        if lms_coeff.complex_data:
            output = np.full(n**3,fill_value,dtype=complex)
            eval_sh = self.harm._sh.SH_to_point_cplx
        else:
            output = np.full(n**3,fill_value,dtype=float)
            eval_sh = self.harm._sh.SH_to_point
            
            
        max_r = self.struct.max_r
        
        bw = min(self.harm.bandwidth, self.struct.bandwidth)

        n_batches = int(np.ceil((n**3)/(batch_size-1)))
        n_big_batches = (n**3)%n_batches
        batch_size = int((n**3)//n_batches)
        batch_split_ids = (
            tuple(i*(batch_size+1) for i in range(n_big_batches))
            +tuple(min(n_big_batches*(batch_size+1)+i*batch_size,n**3) for i in range(n_batches-n_big_batches+1))
        )
        
        coeff_workspace = self.harm.get_empty_coeff(
            pre_shape=(batch_size,),
            complex_data=lms_coeff.complex_data,
        )
        coeff_workspace2 = self.harm.get_empty_coeff(
            pre_shape=(batch_size+1,),
            complex_data=lms_coeff.complex_data,
        )
        #currently does not ignore values outside of data range....
        for batch_id in range(n_batches):
            start = time()
            start_id = batch_split_ids[batch_id]
            stop_id = batch_split_ids[batch_id+1]
            len_batch = stop_id-start_id
            batch_points = self._get_cart_points(start_id,stop_id,points)
            r,theta,phi = cartesian_to_spherical(batch_points).T
            
            
            rho = r / self.struct.max_r
            phi = np.mod(phi, 2 * np.pi)
            cos_theta = np.cos(theta)
            
            for l in range(bw):
                coeff_part = lms_coeff.lm[l][l:self.struct.bandwidth:2]
                radial_basis = ND_zernike_polynomials(l,self.struct.bandwidth,rho,self._dim).T
                if len_batch == batch_size:
                    tmp_workspace = coeff_workspace
                    tmp_workspace.lm[l] = radial_basis @ coeff_part
                else:
                    tmp_workspace = coeff_workspace2
                    tmp_workspace.lm[l] = radial_basis @ coeff_part
                
            values = np.array([
                eval_sh(coeff, ct, p)
                for coeff, ct, p in zip(tmp_workspace, cos_theta, phi)
            ])
            output[start_id:stop_id] = values
            print(f'batch {batch_id}/{n_batches} took {time()-start} seconds')
        return output.reshape((n,)*3)
    def _full_inverse_cartesian_multi_process(self, lms_coeff, points,batch_size=8192,fill_value=0,n_processes = 2):
        """
        Evaluate the spherical-Zernike expansion at arbitrary points.

        Parameters
        ----------
        lms_coeff : ShCoeff
            Zernike coefficients with radial/Zernike index s as first axis.
        points : ndarray, shape (Np)
            Points in spherical coordinates: (r, theta, phi).
        """
        points = np.asarray(points)
        n = len(points)
        
        if lms_coeff.shape[0] != self.struct.bandwidth:
            raise ValueError("Unexpected radial/Zernike coefficient dimension.")
        
        if (not lms_coeff.complex_data
            and np.issubdtype(lms_coeff.dtype, np.floating)):
            raise NotImplementedError(
                "Full real-harmonic coefficient representation is not supported."
            )

        if lms_coeff.complex_data:
            output = np.full(n**3,fill_value,dtype=complex)
            eval_sh = self.harm._sh.SH_to_point_cplx
        else:
            output = np.full(n**3,fill_value,dtype=float)
            eval_sh = self.harm._sh.SH_to_point
            
            
        max_r = self.struct.max_r
        
        bw = min(self.harm.bandwidth, self.struct.bandwidth)

        n_batches = int(np.ceil((n**3)/(batch_size-1)))
        n_big_batches = (n**3)%n_batches
        batch_size = int((n**3)//n_batches)
        batch_split_ids = (
            tuple(i*(batch_size+1) for i in range(n_big_batches))
            +tuple(min(n_big_batches*(batch_size+1)+i*batch_size,n**3) for i in range(n_batches-n_big_batches+1))
        )
        
        coeff_workspace = self.harm.get_empty_coeff(
            pre_shape=(batch_size,),
            complex_data=lms_coeff.complex_data,
        )
        coeff_workspace2 = self.harm.get_empty_coeff(
            pre_shape=(batch_size+1,),
            complex_data=lms_coeff.complex_data,
        )
        
        for batch_id in range(n_batches):
            start = time()
            start_id = batch_split_ids[batch_id]
            stop_id = batch_split_ids[batch_id+1]
            len_batch = stop_id-start_id
            batch_points = self._get_cart_points(start_id,stop_id,points)
            r,theta,phi = cartesian_to_spherical(batch_points).T
            
            
            rho = r / self.struct.max_r
            phi = np.mod(phi, 2 * np.pi)
            cos_theta = np.cos(theta)

            
            for l in range(bw):
                coeff_part = lms_coeff.lm[l][l:self.struct.bandwidth:2]
                radial_basis = ND_zernike_polynomials(l,self.struct.bandwidth,rho,self._dim).T
                if len_batch == batch_size:
                    tmp_workspace = coeff_workspace
                    tmp_workspace.lm[l] = radial_basis @ coeff_part
                else:
                    tmp_workspace = coeff_workspace2
                    tmp_workspace.lm[l] = radial_basis @ coeff_part
                
            values = np.array([
                eval_sh(coeff, ct, p)
                for coeff, ct, p in zip(tmp_workspace, cos_theta, phi)
            ])
            output[start_id:stop_id] = values
            print(f'batch {batch_id}/{n_batches} took {time()-start} seconds')
        return output.reshape((n,)*3)
    def full_inverse_cartesian(self, lms_coeff, points,batch_size=8192,fill_value=0,n_processes = 1):
        if n_processes <=1:
            return self._full_inverse_cartesian_single_process(lms_coeff, points,batch_size=8192,fill_value=0)
        else:
            return self._full_inverse_cartesian_multi_process(lms_coeff, points,batch_size=8192,fill_value=0,n_processes=n_processes)
        
            
#######################
## Experimental Area ##
## 
class GegenbauerTransform:
    def __init__(self,bandwidth,n_points,alpha,domain=(-1,1),mode = 'uniform'):
        self.bandwidth=bandwidth
        assert alpha>0, 'Gegenbauer Polynomials only exist for alpha > 0'
        start,stop = domain        
        self.ns = np.arange(bandwidth)
        self.alpha=alpha
        self.C_n_alpha_1 = np.exp(loggamma(2*alpha+self.ns)-(loggamma(2*alpha)+loggamma(self.ns+1)))
        self.transform_weights = 0
        if mode == 'uniform':
            self.gegenbauer_points = (np.arange(n_points)+0.5)*2/n_points-1
            self.integration_normalization=2/n_points
        elif mode == 'chebyshev':
            self.gegenbauer_points = np.cos((np.arange(n_points)+0.5)*np.pi/n_points)
            self.integration_normalization = np.pi/n_points
        self.normalization=np.pi*2**(1-2*self.alpha)*np.exp(loggamma(self.ns+2*self.alpha)-(2*loggamma(self.alpha)+loggamma(self.ns+1)))/(self.ns+self.alpha)
        self.func_points = start + (stop-start)*(1+self.gegenbauer_points)/2
        self.weights = (1-self.gegenbauer_points**2)**(self.alpha)
        self.gegenbauer_array = eval_gegenbauer(self.ns[:,None],self.alpha,self.gegenbauer_points[None,:])/np.sqrt(self.normalization[:,None])

    def forward(self,f):
        return (self.gegenbauer_array@(f*self.weights))*self.integration_normalization
    def inverse(self,F):
        return F@self.gegenbauer_array

class FreudGibbsTransform:
    '''
    doi: https://doi.org/10.1016/j.acha.2004.12.007
    '''
    def __init__(self,bandwidth,n_points,domain=(-1,1),epsilon=1e-23):
        self.bandwidth = bandwidth
        self.n_points = n_points
        self.N = n_points/2
        self.epsilon = epsilon
        self.poly_points = (np.arange(n_points)+0.5)*2/n_points-1
        start,stop=domain
        self.func_points = start + (stop-start)*(1+self.poly_points)/2
        self.weights = self._freud_gibbs_weights(self.N,epsilon,self.func_points,domain=domain)
        self.poly_values = self.generate_polynomial_values()
    @staticmethod
    def _recurrence_relation_even(pn,pn1,beta,points):
        return points*pn - beta*pn1
    @staticmethod
    def _integrate(f,g,weights):
        return np.sum(f*g*weights)
    @staticmethod
    def _freud_weights(c,n,points):
        return np.exp(-c*points**(2*n))
    def _freud_gibbs_weights(self,N,epsilon,points,domain=(-1,1)):
        n = int(np.sqrt(N*(domain[1]-domain[0])/2)-2*np.sqrt(2)+0.5)
        c = -np.log(epsilon)
        return self._freud_weights(c,n,points)        
    def _calc_beta(self,pn,pn1,weights):
        return self._integrate(pn,pn,weights)/self._integrate(pn1,pn1,weights)
    def generate_polynomial_values(self):
        M=self.bandwidth
        integrate=self._integrate
        calc_beta =self._calc_beta
        recurrence=self._recurrence_relation_even
        weights = self.weights
        points = self.poly_points
        polys = np.zeros((M+1,self.n_points),float)
        betas = np.zeros((M+1,self.n_points),float)
        normalizations = np.zeros(M+1,float)
        polys[0]=np.full(self.n_points,1)
        normalizations[0]=integrate(polys[0],polys[0],weights)
        #polys[0]/=np.sqrt(normalizations[0])
        polys[1]=self._recurrence_relation_even(polys[0],np.full(self.n_points,1),0,points)
        normalizations[1]= integrate(polys[1],polys[1],weights)
        #polys[1]/=np.sqrt(normalizations[1])
        for k in range(1,M):
            beta = calc_beta(polys[k],polys[k-1],weights)
            polys[k+1] = recurrence(polys[k],polys[k-1],beta,points)
            normalizations[k+1] = integrate(polys[k+1],polys[k+1],weights)
            #polys[k+1]/= np.sqrt(normalizations[k+1])
        polys/=np.sqrt(normalizations)[:,None]
        return polys
    
    def forward(self,f):
        return np.sum(self.poly_values*f[None,:]*self.weights[None,:],axis = -1)
    def inverse(self,F):
        return np.sum(self.poly_values*F[:,None],axis=0)
        
        
