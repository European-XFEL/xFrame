import numpy as np
import traceback
import logging
from scipy.special import jv as bessel_jnu
from scipy.special import spherical_jn as bessel_spherical_jnu

from xframe import Multiprocessing
from xframe import settings
from xframe.library.pythonLibrary import xprint

log = logging.getLogger('root')


class PolarHarmonicTransform:
    def __init__(self,max_order=32):
        self.max_order = max_order
        self.n_points = 2*max_order        
        self.forward_cmplx = self._generate_forward_cmplx()
        self.inverse_cmplx = self._generate_inverse_cmplx()
        self.forward_real = self._generate_forward_real()
        self.inverse_real = self._generate_inverse_cmplx()
        self.phis = np.arange(self.n_points)*2*np.pi/self.n_points
        self.angular_shape = (self.n_points,)

    def _generate_forward_cmplx(self):
        fft = np.fft.fft
        n_points = self.n_points
        def forward_cmplx(data):
            return fft(data,axis=-1)/n_points
        return forward_cmplx
    def _generate_inverse_cmplx(self):
        ifft = np.fft.ifft
        n_points = self.n_points
        def inverse_cmplx(data):
            return ifft(data*n_points,axis=-1)
        return inverse_cmplx
    def _generate_forward_real(self):
        fft = np.fft.rfft
        n_points = self.n_points
        def forward_real(data):
            return fft(data,axis=-1)/n_points
        return forward_real
    def _generate_inverse_real(self):
        ifft = np.fft.irfft
        n_points = self.n_points
        def inverse_real(data):
            return ifft(data*n_points,n_points,axis=-1)
        return inverse_real


def get_harmonic_transform(bandwidth:int,dimensions=3,options={}):
    if dimensions==2:
        return PolarHarmonicTransform(max_order=bandwidth)
    elif dimensions==3:
        keys = ['anti_aliazing_degree','n_phi','n_theta']
        shtns_opt = {key:options[key] for key in keys if key in options}
        from xframe.library.mathLibrary import shtns        
        return shtns.ShSmall(bandwidth,**shtns_opt)
        

class HankelTransformWeights:
    ### Direct Midpoint ###
    @classmethod
    def midpoint(cls,n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**kwargs):
        if dimensions==3:
            orders = np.arange(angular_bandwidth)
        elif dimensions ==2:
            orders = np.arange(angular_bandwidth+1)
        worker_by_dimensions= {2:cls.midpoint_polar_worker,3:cls.midpoint_spherical_worker}
        worker = worker_by_dimensions[dimensions]
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_processes_for_weight_generation)
        return weights
    @classmethod
    def midpoint_polar_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates polar midpoint rule weights.
        '''
        N = n_points
        ps=np.arange(N)+1/2
        ks=np.arange(N)+1/2
        ms = orders
        Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
        weights = ps[None,:,None]*Jmpk
        return weights

    @classmethod
    def midpoint_spherical_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates spherical midpoint rule weights.
        '''
        N = n_points
        ps=np.arange(N)+1/2
        ks=np.arange(N)+1/2
        ls = orders
        jmpk = bessel_spherical_jnu(np.repeat(np.repeat(ls[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
        weights = ps[None,:,None]**2*jmpk
        return weights

    @classmethod
    def assemble_midpoint(cls,weights,bandwidth,r_max,reciprocity_coefficient,dimensions=3):
        '''
        Generates weights for the forward and inverse transform by multiplieing with the propper constants.
        And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
        (summation_radial_coordinate,new_radial_coordinate,order)
        '''
        n_radial_points = weights.shape[-1]
        q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient = reciprocity_coefficient)
        #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
        #xprint(f'alignment dimensions = {dimensions}')
        if dimensions == 2:
            orders = np.arange(bandwidth+1)
            all_orders = np.concatenate((orders,-orders[-2:0:-1]))
            #log.info(all_orders)
            forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/n_radial_points)**2#*np.sqrt(2)
            inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/n_radial_points)**2#*np.sqrt(2)
            # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
            #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
            weights = np.concatenate((weights,(-1)**orders[-2:0:-1,None,None]*weights[-2:0:-1]),axis = 0)
        elif dimensions == 3:
            orders = np.arange(bandwidth)
            forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/n_radial_points)**3*np.sqrt(2/np.pi)
            inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/n_radial_points)**3*np.sqrt(2/np.pi)
            
        weights=np.moveaxis(weights,0,2)
        forward_weights=weights*forward_prefactor
        inverse_weights=weights*inverse_prefactor
        #log.info('weights shape = {}'.format(forward_weights.shape))
        return {'forward':forward_weights,'inverse':inverse_weights}

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
        return weights
    @classmethod
    def chebyshev_polar_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates polar chebyshev rule weights.
        '''
        N = n_points
        ps = cls.chebyshev_nodes(N,start=0)
        ks = ps.copy()
        ls = orders
        Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*N)
        weights = ps[None,:,None]**np.sqrt(1-ps)[None,:,None]*Jmpk*np.pi/2
        return weights
    @classmethod
    def chebyshev_spherical_worker(cls,orders,n_points,reciprocity_coefficient,**kwargs):
        '''
        Generates spherical chebyshev rule weights.
        '''
        N = n_points
        ps = cls.chebyshev_nodes(N,start=0)
        ks = ps.copy()
        ls = orders
        jmpk = bessel_spherical_jnu(np.repeat(np.repeat(ls[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*N)
        weights = ps[None,:,None]**2*np.sin((np.arange(N)+0.5)*np.pi/N)[None,:,None]*jmpk*np.pi/2
        return weights
    @classmethod
    def assemble_chebyshev(cls,weights,bandwidth,r_max,reciprocity_coefficient,dimensions=3):
        '''
        Generates weights for the forward and inverse transform by multiplieing with the propper constants.
        And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
        (summation_radial_coordinate,new_radial_coordinate,order)
        '''
        n_radial_points = weights.shape[-1]
        q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient = reciprocity_coefficient)
        #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
        #xprint(f'alignment dimensions = {dimensions}')
        if dimensions == 2:
            orders = np.arange(bandwidth+1)
            all_orders = np.concatenate((orders,-orders[-2:0:-1]))
            #log.info(all_orders)
            forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/n_radial_points)**2#*np.sqrt(2)
            inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/n_radial_points)**2#*np.sqrt(2)
            # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
            #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
            weights = np.concatenate((weights,(-1)**orders[-2:0:-1,None,None]*weights[-2:0:-1]),axis = 0)
        elif dimensions == 3:
            orders = np.arange(bandwidth)
            forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max)**3*np.sqrt(2/np.pi)/n_radial_points
            inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max)**3*np.sqrt(2/np.pi)/n_radial_points
            
        weights=np.moveaxis(weights,0,2)
        forward_weights=weights*forward_prefactor
        inverse_weights=weights*inverse_prefactor
        #log.info('weights shape = {}'.format(forward_weights.shape))
        return {'forward':forward_weights,'inverse':inverse_weights}


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

    
class HankelTransform:
    ht_modes = ['trapz','zernike_trapz','sincos_trapz','midpoint','zernike_midpoint','sincos_midpoint','chebyshev']
    def __init__(self,n_points=64,angular_bandwidth=32,r_max=1,mode='midpoint',dimensions = 3, weights = None, reciprocity_coefficient = np.pi, use_gpu = True, n_processes_for_weight_generation = None,other={}):
        self.mode = mode
        self.dimensions = dimensions
        self.n_points = n_points
        self.bandwidth = angular_bandwidth
        self.use_gpu=use_gpu
        self.reciprocity_coefficient = reciprocity_coefficient
        self.r_max = r_max
        if use_gpu:
            settings.general.n_control_workers = 1
            Multiprocessing.comm_module.restart_control_worker()
        if not isinstance(weights,np.ndarray):            
            weight_generator = getattr(HankelTransformWeights,mode)
            weights = weight_generator(n_points,angular_bandwidth,dimensions,reciprocity_coefficient,n_processes_for_weight_generation,**other)
        self.assembled_weights = getattr(HankelTransformWeights,'assemble_'+mode)(weights,self.bandwidth,r_max,self.reciprocity_coefficient,dimensions=self.dimensions)
        self._forward_coeff,self._inverse_coeff = self._generate_coeff_arrays()
        self.forward_cmplx,self.inverse_cmplx = self._generate_ht()

    def from_weight_dict(self,weights_dict,r_max=1,use_gpu=True):
        w = weights_dict
        weights = w['weights']
        n_points = weights.shape[1]
        bandwidth = self.weights.shape[-1]
        self.__init__(n_points=n_points,angular_bandwidth=bandwidth,r_max=r_max,mode = w['mode'],dimensions=w['dimensions'],weights=weights,reciprocity_coefficient=w['reciprocity_coefficient'],use_gpu=use_gpu)
        return self
    
    def _generate_coeff_arrays(self):
        if self.dimensions==2:
            coeffs = self._generate_polar_coeff_arrays()
        elif self.dimensions==3:
            coeffs = self._generate_spherical_coeff_arrays()
        return coeffs
    def _generate_spherical_coeff_arrays(self):
        from xframe.library.mathLibrary import shtns
        coeff_shape = (self.n_points,self.bandwidth**2)
        forward_array = np.zeros(coeff_shape,dtype=complex)
        inverse_array = np.zeros(coeff_shape,dtype=complex)
        ls = np.arange(self.bandwidth)
        ms = np.concatenate((-ls,ls[1:]))
        forward_coeff = shtns.ShCoeff(forward_array,ls,ms)
        inverse_coeff = shtns.ShCoeff(inverse_array,ls,ms)
        return forward_coeff,inverse_coeff
    def _generate_polar_coeff_arrays(self):
        coeff_shape = (self.n_points,2*self.bandwidth)
        forward_array = np.zeros(coeff_shape,dtype=complex)
        inverse_array = np.zeros(coeff_shape,dtype=complex)
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
        fw= np.swapaxes(self.assembled_weights['forward'],0,1)
        iw= np.swapaxes(self.assembled_weights['inverse'],0,1) 
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
                for l in l_orders:
                    matmul(fw[:,:,l],harmonic_coeff.lm[l],out = forward_coeff.lm[l])
                return forward_coeff        
            def iht(harmonic_coeff):
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
        nq = n_radial_points
        nl = len(l_orders)
        nlm = self.bandwidth**2
        if 'trapz' in self.mode:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out, 
            __global double2* w, 
            __global double2* rho, 
            long nq,long nlm, long nl)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
            long l = (long) sqrt((double)j);
    
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int q = 0; q < nq-1; ++q)
            {
            double2 wqql = w[q*nq*nl + i*nl + l];
            double2 rqlm = rho[(q+1)*nlm + j];
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
            long nq,long nlm, long nl)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
            long l = (long) sqrt((double)j);
    
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int q = 0; q < nq; ++q)
            {
            double2 wqql = w[q*nq*nl + i*nl + l];
            double2 rqlm = rho[q*nlm + j];
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
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                    'shapes' : ((nq,nlm),forward_weights.shape,(nq,nlm),None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                    'const_inputs' : (None,forward_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)),
                    'global_range' : (nq,nlm),
                    'local_range' : None,                    
                },)
            }
        
        kernel_dict_inverse={
                'kernel': kernel_str,
                'name': 'inverse_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                    'shapes' : ((nq,nlm),inverse_weights.shape,(nq,nlm),None,None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                    'const_inputs' : (None,inverse_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)),
                    'global_range' : (nq,nlm), 
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
        fw= np.swapaxes(self.assembled_weights['forward'],0,1)
        iw= np.swapaxes(self.assembled_weights['inverse'],0,1)
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
                out_forward[:]=fw@harmonic_coeff[1:]
                return out_forward
            def iht(harmonic_coeff):
                out_inverse[:]=iw@harmonic_coeff[1:]
                #harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[1:,None,:],axis = 0,out=out_inverse)
                #harmonic_coeff[:,unused_order_mask]=0
                return out_inverse
        else:
            def ht(harmonic_coeff):
                out_forward[:]=fw@harmonic_coeff
                #log.info('harmonic shape = {}'.format(harmonic_coeff.shape))
                #log.info('out forward = {}'.format(out_forward.shape))
                #reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[:,None,:],axis = 0,out=out_forward)
                #reciprocal_harmonic_coeff[:,unused_order_mask]=0
                return out_forward
            def iht(reciprocal_coeff):
                out_inverse[:]=iw@harmonic_coeff
                #harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[:,None,:],axis = 0,out=out_inverse)
                #harmonic_coeff[:,unused_order_mask]=0
                return out_inverse
        return ht,iht

    def _generate_polar_ht_gpu(self):
        forward_weights = self.assembled_weights['forward']
        inverse_weights = self.assembled_weights['inverse']
        n_radial_points = forward_weights.shape[1]

        m_max=self.bandwidth
        nq = n_radial_points
        nm = m_max*2
    
        if 'trapz' in self.mode:
            kernel_str = """
            __kernel void
            apply_weights(__global double2* out, 
            __global double2* w, 
            __global double2* rho, 
            long nq,long nm)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int q = 0; q < nq-1; ++q)
            {
            double2 wqqm = w[q*nq*nm + i*nm + j];
            double2 rqm = rho[(q+1)*nm + j];
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
            long nq,long nm)
            {
      
            long i = get_global_id(0); 
            long j = get_global_id(1);
     
            // value stores the element that is 
            // computed by the thread
            double2 value = 0;
            // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
            for (int q = 0; q < nq; ++q)
            {
            double2 wqqm = w[q*nq*nm + i*nm + j];
            double2 rqm = rho[q*nm + j];
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
                    'dtypes' : (complex,complex,complex,np.int64,np.int64),
                    'shapes' : ((nq,nm),forward_weights.shape,(nq,nm),None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input'),
                    'const_inputs' : (None,forward_weights,None,np.int64(nq),np.int64(nm)),
                    'global_range' : (nq,nm),
                    'local_range' : None
                },)
            }
        
        kernel_dict_inverse={
                'kernel': kernel_str,
                'name': 'inverse_hankel',
                'functions': ({
                    'name': 'apply_weights',
                    'dtypes' : (complex,complex,complex,np.int64,np.int64),
                    'shapes' : ((nq,nm),inverse_weights.shape,(nq,nm),None,None),
                    'arg_roles' : ('output','const_input','input','const_input','const_input'),
                    'const_inputs' : (None,inverse_weights,None,np.int64(nq),np.int64(nm)),
                    'global_range' : (nq,nm),
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
        if 'trapz' in self.mode:
            rs = np.arange(self.n_points)*self.r_max/(self.n_points-1)
        elif 'midpoint' in self.mode:
            rs = (np.arange(self.n_points)+0.5)*self.r_max/self.n_points
        elif 'chebyshev' in self.mode:
            rs = HankelTransformWeights.chebyshev_nodes(self.n_points,start=0)*self.r_max
        return rs
    @property
    def reziprocal_radial_points(self):
        q_max = polar_spherical_dft_reciprocity_relation_radial_cutoffs(self.r_max,self.n_points,reciprocity_coefficient = self.reciprocity_coefficient)
        if 'trapz' in self.mode:
            qs = np.arange(self.n_points)*q_max/(self.n_points-1)
        elif 'midpoint' in self.mode:
            qs = (np.arange(self.n_points)+0.5)*q_max/self.n_points
        elif 'chebyshev' in self.mode:
            qs = HankelTransformWeights.chebyshev_nodes(self.n_points,start=0)*q_max
        return qs

def polar_spherical_dft_reciprocity_relation_radial_cutoffs(cutoff:float,n_points:int,reciprocity_coefficient=np.pi):
    '''
    Reciprocity relation between the real and reciprocal cutoff.
    Reciprocity coefficient of Pi corresponds to the usual FFT relation for [2pi/Angstrom] units in reciprocal and [Angstrom] in reals space.
    '''
    other_cutoff = reciprocity_coefficient*n_points/cutoff    
    return other_cutoff


class SphericalFourierTransform:
    def __init__(self,n_radial_points=64,angular_bandwidth=32,r_max=1,mode='midpoint',dimensions = 3,weights = None,reciprocity_coefficient = np.pi, use_gpu = True,n_processes_for_weight_generation = None, other={}):
        self.ht = HankelTransform(n_points=n_radial_points,angular_bandwidth=angular_bandwidth,r_max=r_max,mode=mode,dimensions=dimensions,weights=weights,reciprocity_coefficient=reciprocity_coefficient,use_gpu=use_gpu,n_processes_for_weight_generation = n_processes_for_weight_generation,other=other)
        self.harm = get_harmonic_transform(angular_bandwidth,dimensions=dimensions,options=other)
        self._init_end()
    def _init_end(self):
        self.dimensions=self.ht.dimensions
        self.forward_cmplx,self.inverse_cmplx = self._generate_transforms()
    def from_weight_dict(self,weights_dict,r_max=1,use_gpu=True):
        w = weights_dict
        weights = w['weights']
        n_points = weights.shape[1]
        bandwidth = self.weights.shape[-1]
        self.__init__(n_readial_points=n_points,angular_bandwidth=bandwidth,r_max=r_max,mode = w['mode'],dimensions=w['dimensions'],weights=weights,reciprocity_coefficient=w['reciprocity_coefficient'],use_gpu=use_gpu,other={})
        return self

    def from_transforms(self,hankel_transform,harmonic_transform):
        self.ht = hankel_transform
        self.harm = harmonic_transform
        self._init_end()
        return self    
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
            return harm_inverse_cmplx(coeff)
        return forward_cmplx,inverse_cmplx
    def empty_density(self):
        angular_shape = self.harm.angular_shape
        radial_shape = (self.ht.n_points,)
        return np.zeros(radial_shape+angular_shape,dtype=complex)
    @property
    def real_grid(self):
        rs = self.ht.real_radial_points
        phis = self.harm.phis
        if self.dimensions==2:
            grid = np.stack(np.meshgrid(rs,phis,indexing='ij'),2)
        elif self.dimensions == 3:
            thetas = self.harm.thetas
            grid = np.stack(np.meshgrid(rs,thetas,phis,indexing='ij'),3)
        return grid
    @property
    def reciprocal_grid(self):
        qs = self.ht.reziprocal_radial_points
        phis = self.harm.phis
        if self.dimensions==2:
            grid = np.stack(np.meshgrid(qs,phis,indexing='ij'),2)
        elif self.dimensions == 3:
            thetas = self.harm.thetas
            grid = np.stack(np.meshgrid(rs,thetas,phis,indexing='ij'),3)
        return grid
        


class ZernikeTransform:
    '''class implementing Zernike Series expansion.'''
    def __init__(self,bandwidth=32,order=0,n_points=128,max_r=1,dimension=3,grid='uniform',bw_to_sampling_factor=4):
        from xframe.library.mathLibrary import eval_ND_zernike_polynomials
        step = max_r/n_points
        self.s = np.arange(order,bandwidth,2)

        if grid=='chebyshev':
            R = max_r
            N = n_points
            ks = np.arange(1,N+1)
            phis = np.pi/2-(ks-0.5)*np.pi/(2*N)
            self.points = p = (R/2*(1+np.cos((ks-1/2)/N*np.pi)))[::-1]
            #self.points = p = R*np.cos(phis)
            self.weight = np.pi/N #*np.sqrt(R*p-(p*R)**2)#*2
            #self.weight = np.pi/(N*2) #*np.sqrt(R*p-(p*R)**2)#*2
            self.izernike_values = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,self.points,dimension)[order]
            self.zernike_values = self.izernike_values*np.sqrt(1-p**2)[None,:]#np.sqrt(R*p-(p*R)**2)[None,:]
            #self.zernike_values = self.izernike_values*np.sqrt(R*p-(p*R)**2)[None,:]
            
            
            zernike_points_per_step = int(bandwidth*bw_to_sampling_factor/n_points)+1
            zN = N*zernike_points_per_step
            zks = np.arange(1,zN+1)
            zernike_points = zp = (R/2*(1+np.cos((zks-1/2)/zN*np.pi)))[::-1]
            zweights = 1/zernike_points_per_step*np.sqrt(R*zp-(zp*R)**2)
            zernike_values=np.zeros((len(self.s),n_points),dtype=float)
            izernike_values=np.zeros((len(self.s),n_points),dtype=float)
            for i in range(zernike_points_per_step):
                vals = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,zernike_points[i::zernike_points_per_step],dimension)[order]
                zernike_values+=vals*zweights[i::zernike_points_per_step]
                izernike_values+=vals/zernike_points_per_step#*zweights[i::zernike_points_per_step]                
            #self.zernike_values = zernike_values
            #self.izernike_values = izernike_values
        elif grid == 'cheby2':
            R = max_r
            N = n_points
            cks = np.cos((np.arange(N)+0.5)*np.pi/(2*N))
            self.points = cks
            self.weight = np.pi/(2*N)
            self.izernike_values = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,cks,dimension)[order]
            self.zernike_values = self.izernike_values*np.sqrt(1-cks**2)
        elif grid == 'cheby3':
            R = max_r
            N = n_points
            sks = np.sin((np.arange(N)+0.5)*np.pi/(2*N))
            self.points = sks
            self.weight = np.pi/(2*N)
            self.izernike_values = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,sks,dimension)[order]
            self.zernike_values = self.izernike_values*np.sqrt(1-sks**2)
        elif grid == 'cheby4':
            R = max_r
            N = n_points
            sks = np.sin((np.arange(N)+0.5)*np.pi/(2*N))
            self.points = sks
            self.weight = np.pi/(2*N)
            self.izernike_values = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,sks,dimension)[order]
            self.zernike_values = self.izernike_values*np.sqrt(1-sks**2)

            zernike_points_per_step = int(bandwidth*bw_to_sampling_factor/n_points)
            print(f'n_points = {n_points}, zernike_points_per_step = {zernike_points_per_step}')
            zN = N*zernike_points_per_step
            zks = np.arange(1,zN)                 
            zernike_points = zp =  np.sin((np.arange(zN)+0.5)*np.pi/(2*zN))
            zweights = np.sqrt(1-zp**2)/(zernike_points_per_step)
            zernike_values=np.zeros((len(self.s),n_points),dtype=float)
            izernike_values=np.zeros((len(self.s),n_points),dtype=float)
            for i in range(zernike_points_per_step):
                vals = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,zernike_points[i::zernike_points_per_step],dimension)[order]
                zernike_values+=vals*zweights[i::zernike_points_per_step]
                izernike_values+=vals/zernike_points_per_step#*zweights[i::zernike_points_per_step]
            self.izernike_values = izernike_values
            self.zernike_values = zernike_values
        else:
            self.points = (np.arange(n_points)+0.5)*step
            zernike_points_per_step = int(bandwidth*bw_to_sampling_factor/n_points)+1
            zernike_points = (np.arange(n_points*zernike_points_per_step)+0.5)*(step/zernike_points_per_step)
            self.weight= step
            zernike_values=np.zeros((len(self.s),n_points),dtype=float)
            for i in range(zernike_points_per_step):
                zernike_values+=eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,zernike_points[i::zernike_points_per_step],dimension)[order]/zernike_points_per_step
                
            self.zernike_values = zernike_values
            self.izernike_values = zernike_values
            
#        self.zernike_values = eval_ND_zernike_polynomials(np.array([order]),bandwidth-1,self.points,dimension)[order]

        self.bandwidth = bandwidth
        self.order = order
        self.dimension=dimension
        self.n_points = n_points
        self.C=np.sqrt(2*self.s+dimension)
    def forward(self,f):
        fs = (self.zernike_values*self.C[:,None]*self.points[None,:]**(self.dimension-1)*self.weight) @ f
        return fs
    def inverse(self,fs):
        return np.sum(self.izernike_values*self.C[:,None]*fs[:,None],axis=0)
        
