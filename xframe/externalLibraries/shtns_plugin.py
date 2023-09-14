import numpy as np
import logging

from xframe.library.gridLibrary import GridFactory
from xframe.library import pythonLibrary as pyLib
from xframe.library.interfaces import SphericalHarmonicTransformInterface
import shtns

log=logging.getLogger('root')

class sh(SphericalHarmonicTransformInterface):
    # want to use r,theta,phi type of data
    # l indexed coeff are f_{l,:} = coeff[l] 0<=l<=L_max of size: 2*l+1
    #   were coeff[l] cotains f_{l,m} starting with -l<=m<=l
    # m indexed coeff are f_{:,m} = coeff[m] -L_max<=m<=L_max of size: L_max-abs(m)
     
    
    def __init__(self,l_max,mode_flag='complex',output_order='l',anti_aliazing_degree = 2,n_phi = False,n_theta=False):
        #log.info('\n \n harmonic transform l_max = {} \n \n'.format(l_max)
        sh = shtns.sht(l_max)#,norm = shtns.sht_schmidt)        
        self._sh = sh
        self.l_max=l_max
        self.anti_aliazing_degree = anti_aliazing_degree        
        self.n_coeff = (l_max+1)**2
        #log.info(" sh created form n_phi= {},n_theta={},l_max ={}".format(n_phi,n_theta,l_max))
        thetas,phis,grid=self._generate_grid(n_phi=n_phi,n_theta=n_theta)
        #log.info(" sh created  grids n_phi= {},n_theta={}".format(len(phis),len(thetas)))
        self._theta=thetas
        self._phi=phis
        self._grid=grid
        self.mode=mode_flag
        self.m,self.l,self.cplx_m_indices,self.cplx_l_indices=self.generate_complex_lm_indices()
        self.cplx_m_indices_concat = np.concatenate(self.cplx_m_indices)
        self.cplx_l_split_indices = np.arange(1,l_max+1)**2
        self.cplx_m_split_indices = self.generate_cplx_m_split()
        self.real_split_indices=tuple((sh.l==l).nonzero()[0] for l in range(l_max+1))
        self.inv_real_split_indices=np.argsort(np.concatenate(self.real_split_indices))
        #log.info("alive!")
        if mode_flag=='real':
            self._forward_l=self.forward_transform_real_l
            self._inverse_l=self.inverse_transform_real_l
            self._forward_m=self.forward_transform_real_m
            self._inverse_m=self.forward_transform_real_m
            self._forward_direct = NotImplementedError()
            self._inverse_direct = NotImplementedError()
        elif mode_flag=='complex':
            self._forward_l=self.forward_transform_complex_l
            self._forward_m=self.forward_transform_complex_m
            self._inverse_l=self.inverse_transform_complex_l
            self._inverse_m=self.inverse_transform_complex_m
            self._forward_direct = self.generate_forward_transform_complex_direct()
            self._inverse_direct = self.generate_inverse_transform_complex_direct()
        else:
            raise AssertionError('harmonic transform mode "{}" is not known. Known modes are "real" and "complex".'.format(mode_flag))

    @property
    def phi(self):
        return self._phi
    @property
    def theta(self):
        return self._theta
    @property
    def grid(self):
        return self._grid

    @property
    def forward_l(self):
        return self._forward_l
    @property
    def forward_m(self):
        return self._forward_m
    @property
    def forward_d(self):
        return self._forward_direct
    @property
    def inverse_d(self):
        return self._inverse_direct
    @property
    def inverse_m(self):
        return self._inverse_m
    @property
    def inverse_l(self):
        return self._inverse_l

    def max_order_from_n_angular_steps(self,n_phi):
        '''
        max_order for highest power of 2 smaller than n_phi
        '''
        n_phi = 2**int(np.log2(n_phi))
        N = self.anti_aliazing_degree
        max_order = n_phi//(N+1)
        return max_order

    def n_angular_step_from_max_order(self,max_order):
        size_dict={}
        N = self.anti_aliazing_degree
        n_phi = 2**(int(np.log2((N+1)*max_order)) + 1)
        n_theta = n_phi//2
        size_dict['n_phi'] = n_phi
        size_dict['n_theta'] = n_theta
        return size_dict
    
    
    
    def generate_complex_lm_indices(self):
        l_max=self.l_max
        ls=np.arange(l_max+1,dtype=int)
        ms=np.concatenate((ls,-ls[:0:-1]))
        
        m_ordered_indices=[ls[np.abs(m):]*(ls[np.abs(m):]+1)+m for m in ms]

        l_ordered_indices=[slice(l**2,l**2+2*l+1) for l in  range(l_max+1)]                        

        return ms,ls,m_ordered_indices,l_ordered_indices


    def _generate_grid(self,n_phi=False,n_theta=False):
        sh=self._sh
        size_dict = self.n_angular_step_from_max_order(self.l_max)
        #log.info('n_theta = {}, n_phi = {}'.format(n_theta,n_phi))
        if (not isinstance(n_theta,int)) or isinstance(n_theta,bool):
            n_theta=size_dict['n_theta']
        if (not isinstance(n_phi,int)) or isinstance(n_phi,bool):
            n_phi = size_dict['n_phi']
            
        #with pyLib.stdout_redirected(): # external library has a frites directly to stdout this line suppresses it.
            #n_theta,n_phi = sh.set_grid(nlat = n_theta,nphi=n_phi,polar_opt=0,flags=shtns.sht_gauss)
        #log.info(f'nlat = {n_theta} nphi = {n_phi}')
        
        sh.set_grid(polar_opt=0,flags=shtns.sht_gauss) #extra call needed otherwise there are sometimes errors in shtns grid generation
        n_theta,n_phi = sh.set_grid(nlat = n_theta,nphi=n_phi,polar_opt=0,flags=shtns.sht_gauss)            
        phis=2*np.pi*np.arange(n_phi)/(n_phi*sh.mres)
        thetas=np.arccos(sh.cos_theta)
        grid=GridFactory.construct_grid('uniform',(thetas,phis))
        return thetas,phis,grid


    def _io_analysis_real_decorator(fun):        
        def new_function(self,data):
            sh=self._sh
            shape=data.shape[:-2]
            data=data.reshape(-1,sh.nlat,sh.nphi)
            #log.info('data shape ={} '.format(data.shape))
            val=fun(self,data)
            val=val.reshape(*shape,-1)
            #log.info('val_shape={}'.format(val.shape))
            val=tuple(val[:,indices] for indices in self.real_split_indices)
            return val
        return new_function
    
    def _analysis_decorator(fun):
        def new_function1(self,data):
            #log.info('forward shtns input shape= {}'.format(data.shape))
            #log.info('data pre shape = {}'.format(data.shape))
            sh=self._sh
            shape=data.shape[:-2]
            data=data.reshape(-1,*sh.spat_shape)
            val=fun(self,data)
            val=np.moveaxis(val.reshape(*shape,-1),-1,0)
            return val
        return new_function1
    
    @pyLib.optional_arg_decorator
    def _analysis_output_complex_decorator(fun,order='l'):
        if order=='l':            
            def new_function(self,data):
                val=fun(self,data)
                val=[ np.moveaxis(val[index],0,-1) for index in self.cplx_l_indices ]
                #log.info('l forward shtns output len= {}'.format(len(val)))
                return val
        elif order=='m':   
            def new_function(self,data):
                val=fun(self,data)
                val=[ np.moveaxis(val[index],0,-1) for index in self.cplx_m_indices ]
                #log.info('m forward shtns output len= {}'.format(len(val)))
                return val
        return new_function
    
    @pyLib.optional_arg_decorator
    def _synthesis_complex_decorator(fun,order='l'):
        if order=='l':
            def new_function(self,data):
                data=np.concatenate(data,axis=1)
                return fun(self,data)
        elif order=='m':
            def new_function(self,data):
                l_max=self.l_max
                lm_shape=(l_max**2+2*l_max+1,)
                r_shape=data[0].shape[:-1]                
                input_data=np.zeros(r_shape+lm_shape,dtype=complex)
                for m_id,index in enumerate(self.cplx_m_indices):
                    input_data[...,index]=data[m_id]
                return fun(self,input_data)
        return new_function

    def _io_synthesis_real_decorator(fun):
        def new_function(self,data):
            data=np.concatenate(data,axis=1)[...,self.inv_real_split_indices]
            return fun(self,data)
        return new_function


    @_io_analysis_real_decorator
    def forward_transform_real_l(self,data):
        return np.array(tuple(map(self._sh.analys,data)))
    @_io_synthesis_real_decorator
    def inverse_transform_real_l(self,data):
        return np.array(tuple(map(self._sh.synth,data)))


    def forward_transform_real_m(self,data):
        raise NotImplementedError()

    def inverse_transform_real_m(self,data):
        raise NotImplementedError()

    
    @_analysis_output_complex_decorator(order='l')
    @_analysis_decorator
    def forward_transform_complex_l(self,data):
        #data[0]=data[0].mean() # assumme r_0 = 0 
        return np.array(tuple(map(self._sh.analys_cplx,data)))
        

    @_analysis_output_complex_decorator(order='m')
    @_analysis_decorator
    def forward_transform_complex_m(self,data):
        #data[0]=data[0].mean() # assumme r_0 = 0 
        return np.array(tuple(map(self._sh.analys_cplx,data)))
    @_synthesis_complex_decorator(order='l')
    def inverse_transform_complex_l(self,data):
        #data[0,1:] = 0
        return np.array(tuple(map(self._sh.synth_cplx,data)))
    
    @_synthesis_complex_decorator(order='m')
    def inverse_transform_complex_m(self,data):
        #data[0,1:] = 0
        return np.array(tuple(map(self._sh.synth_cplx,data)))        


    def m_to_l_ordering(self,m_coeff):
        l_coeff = np.zeros_like(m_coeff)
        pos = 0
        for index in self.cplx_m_indices:
            n_parts = len(index)
            l_coeff[index] = m_coeff[pos:pos+n_parts]
            pos += n_parts
        return l_coeff

    def generate_forward_transform_complex_direct(self):
        analys_cplx = self._sh.analys_cplx
        def forward_transform_complex_direct(data):
            #data[0]=data[0].mean() # assumme r_0 = 0
            return np.array(tuple(analys_cplx(q_shell) for q_shell in data))
        return forward_transform_complex_direct

    def generate_inverse_transform_complex_direct(self):
        synth_cplx = self._sh.synth_cplx
        def inverse_transform_complex_direct(data):
            return np.array(tuple(synth_cplx(q_shell) for q_shell in data))
        return inverse_transform_complex_direct
    
    def _test_map(self,data):
        sh= self._sh
        return sh.synth_cplx(sh.analys_cplx(data))
    def test(self,data):
        return np.array(tuple(map(self._test_map,data+0.j)))        

    def generate_cplx_m_split(self):
        l_max = self.l_max
        lp1 = np.arange(l_max+2)
        index = (lp1*(lp1+1)/2).astype(int)
        split_ids = np.concatenate((index[-1] - index[-2::-1],index[-1]+index[1:-2]))
        return split_ids
