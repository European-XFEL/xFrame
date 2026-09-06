import numpy as np
import logging

from xframe.library.gridLibrary import GridFactory
from xframe.library import pythonLibrary as pyLib
from xframe.library.interfaces import SphericalHarmonicTransformInterface
import shtns

log=logging.getLogger('root')

def shape_change_decorator(item_shape,out_shape=tuple()):
    def decorator(func):
        def wrapper(ndarray):
            input_shape = ndarray.shape
            ndarray = ndarray.reshape(-1,*item_shape)

            results = func(ndarray)

            if input_shape==item_shape:
                return results[0]
            else:
                new_shape = input_shape[:-len(item_shape)]+out_shape
                results = results.reshape(new_shape)
            return results
        return wrapper
    return decorator            

def complex_l_slice(l):
    return slice(l**2,(l+1)**2)
def get_lm_id(l,m):
    return l*(l+1)+m
def get_ml_id_real(m,l,bw):
    return int((m*(2*bw-1-m))//2+l)
def get_m_ids(m,l0s):
    return l0s[abs(m):]+m
def coeff_shape_complex(bandwidth):
    return bandwidth**2




class ShCoeff(np.ndarray):
    @classmethod
    def from_bandwith_complex(cls,array,bandwidth):
        ls = np.arange(bandwidth)
        ms =  np.concatenate((np.arange(bandwidth,dtype=int),-np.arange(1,bandwidth,dtype=int)[::-1]))
        n_coeff = coeff_shape_complex(bandwidth)
        l_ids_complex = np.zeros(n_coeff,dtype = int)
        m_ids_complex = np.zeros(n_coeff,dtype = int)
        for l in ls:
            for m in np.arange(-l,l+1):
                i = get_lm_id(l,m)
                l_ids_complex[i]=l
                m_ids_complex[i]=m
        return cls(array,l_ids_complex,m_ids_complex,ls = ls,ms=ms)
    
    def __new__(cls,array,l_ids,m_ids,ls=None,ms=None,complex_data=True):
        coeff = array.view(cls)
        coeff.ls = ls
        coeff.ms = ms
        if ls is None:
            coeff.ls = np.unique(l_ids)
        if ms is None:
            coeff.ls = np.unique(m_ids)
        coeff.l_ids = l_ids
        coeff.m_ids = m_ids
        
        coeff.lm=ShCoeffView(coeff)
        coeff.complex_data = complex_data
        return coeff
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        
        self.ls = getattr(obj,'ls',None)
        self.ms = getattr(obj,'ms',None)
        self.l_ids = getattr(obj,'l_ids',None)
        self.m_ids = getattr(obj,'m_ids',None)
        self.lm = getattr(obj,'lm',None)
        self.complex_data= getattr(obj,'complex_data',None)
        
    def copy(self):
        return ShCoeff(np.array(self),
                       self.l_ids,
                       self.m_ids,
                       ls = self.ls,
                       ms = self.ms,
                       complex_data = self.complex_data)
    
    def conj(self,*args,**kwargs):
        return ShCoeff(super().conj(*args,**kwargs),
                       self.l_ids,
                       self.m_ids,
                       ls = self.ls,
                       ms = self.ms,
                       complex_data = self.complex_data)
    
    def point_inverse(self):
        out = self.copy()
        for l in self.ls:
            out.lm[l]=(-1)**l*self.lm[l]
        return out
    
class ShCoeffView:
    def __init__(self,coeff:ShCoeff,mode='complex'):
        self.coeff = coeff
        self.ls = coeff.ls
        self.bw = np.max(self.ls)+1
        self.ms = coeff.ms
        self.l_ids = coeff.l_ids
        self.m_ids = coeff.m_ids
        self.l0s = (self.l_ids==0)#coeff.ls*(coeff.ls+1)
    def get_l_mask(self,l_sel):
        selected_ls = self.ls[l_sel]
        mask = np.in1d(self.l_ids,selected_ls)
        return mask
    def get_m_mask(self,m_sel):
        selected_ms = self.ms[m_sel]
        mask = np.in1d(self.m_ids,selected_ms)
        return mask

    def _to_coeff_items_complex(self,items):
        if not isinstance(items,tuple):
            return (Ellipsis,complex_l_slice(items))
        elif len(items)==1:
            if isinstance(items[0],int):
                return (Ellipsis,complex_l_slice(items[0]))
            else:
                return (Ellipsis,self.get_l_mask(items[0]))
        else:
            if (items[0]==slice(None)) and isinstance(items[1],int):
                return (Ellipsis,get_m_ids(items[1],self.l0s))
            elif (isinstance(items[0],int)) and  (isinstance(items[1],int)):
                #l_mask = self.get_l_mask(items[0])
                #m_mask = self.get_m_mask(items[1])
                #print(np.sum(l_mask & m_mask))
                #return self.coeff[...,l_mask & m_mask]
                return (Ellipsis,get_lm_id(items[0],items[1]))
            else:
                l_mask = self.get_l_mask(items[0])
                m_mask = self.get_m_mask(items[1])               
                mask = l_mask & m_mask
                return (Ellipsis,mask)
            
    def _to_coeff_items_real(self,items):
        if not isinstance(items,tuple):
            items = (items,)
            
        if len(items) == 1:
            return (Ellipsis,self.get_l_mask(items[0]))
        else:
            l_mask = self.get_l_mask(items[0])
            m_mask = self.get_m_mask(items[1])               
            mask = l_mask & m_mask
            return (Ellipsis,mask)
        
    def _to_coeff_items(self,items):
        if self.coeff.complex_data:
            return self._to_coeff_items_complex(items)
        else:
            return self._to_coeff_items_real(items)
    def __getitem__(self,items):
        items = self._to_coeff_items(items)
        return self.coeff[*items]

            
    def __setitem__(self,items,value):
        coeff_items = self._to_coeff_items(items)
        if isinstance(items,tuple):
            if len(items) ==2:
                if self.coeff.complex_data:
                    self.coeff[...,get_lm_id(items[0],items[1])] = value
                else:
                    self.coeff[...,get_ml_id_real(items[1],items[0],self.bw)] = value
            else:
                self.coeff[*coeff_items] = value
        else:
            self.coeff[*coeff_items] = value
            
        
class ShSmall:
    def __init__(self,bandwidth,anti_aliazing_degree = 2,n_phi = 0,n_theta=0):
        #print(f'bandwidth = {bandwidth}')
        sh = shtns.sht(int(bandwidth-1))#,norm = shtns.sht_schmidt)
        self._sh = sh
        self.bandwidth = bandwidth
        self.max_order = bandwidth-1
        self.anti_aliazing_degree = anti_aliazing_degree
        self.n_coeff = (bandwidth)**2
        self.n_coeff_real_data_complex_coeff = bandwidth*(bandwidth+1)//2
        print('generating grid')
        #log.info(" sh trying to create grids with n_phi= {},n_theta={}".format(n_phi,n_theta))
        thetas,phis=self._generate_grid(n_phi=n_phi,n_theta=n_theta)
        #log.info(" sh created  grids n_phi= {},n_theta={}".format(len(phis),len(thetas)))
        self.thetas=thetas
        self.phis=phis
        self.n_thetas = len(thetas)
        self.n_phis = len(phis)
        self.angular_shape=(self.n_thetas,self.n_phis)
        self.ls=np.arange(bandwidth,dtype=int)
        #self.ms=np.arange(-bandwidth+1,bandwidth,dtype=int)
        self.ms = np.concatenate((np.arange(bandwidth,dtype=int),-np.arange(1,bandwidth,dtype=int)[::-1]))
        
        self.l_ids_complex = np.zeros(self.n_coeff,dtype = int)
        
        # array such that np.split(coeff,self.l_split_ids_complex) returns a list in which the l'th entry contains harmonic coefficients of degree l, i.e. (I^L_m) for L=l and |m|<=l.          
        self.m_ids_complex = np.zeros(self.n_coeff,dtype = int)
        for l in self.ls:
            for m in np.arange(-l,l+1):
                i = get_lm_id(l,m)
                self.l_ids_complex[i]=l
                self.m_ids_complex[i]=m
        self.l_split_ids_complex = np.nonzero(np.roll(np.diff(self.l_ids_complex),1))[0]
        
        self.shtns_real_zero_ms = (self._sh.m==0)
        self.l_ids_real = np.concatenate((self._sh.l,self._sh.l[~self.shtns_real_zero_ms]))
        self.m_ids_real = np.concatenate((self._sh.m,-self._sh.m[~self.shtns_real_zero_ms]))
        
        self.l_ids_real_data_complex_coeff = np.zeros(self.n_coeff_real_data_complex_coeff,dtype = int)
        self.m_ids_real_data_complex_coeff = np.zeros(self.n_coeff_real_data_complex_coeff,dtype = int)
        self.ms_real_data_complex_coeff = self.ls.copy()
        for m in range(bandwidth):
            for l in range(m,bandwidth):
                i = get_ml_id_real(m,l,bandwidth)
                self.l_ids_real_data_complex_coeff[i] = l
                self.m_ids_real_data_complex_coeff[i] = m
        
        self.forward_cmplx = self._generate_forward_cmplx()
        self.inverse_cmplx =  self._generate_inverse_cmplx()
        self.forward_real = self._generate_forward_real()
        self.inverse_real =  self._generate_inverse_real()
        self.forward_real_data_cmplx_coeff = self._generate_forward_real_data_cmplx_coeff()
        self.inverse_real_data_cmplx_coeff =  self._generate_inverse_real_data_cmplx_coeff()
        
    def _generate_grid(self,n_phi=False,n_theta=False):
        sh=self._sh
        size_dict = self.n_angular_step_from_max_order()
        #log.info('n_theta = {}, n_phi = {}'.format(n_theta,n_phi))
        if (not np.issubdtype(type(n_theta),np.integer)):
            n_theta=size_dict['n_theta']
        else:
            n_theta=int(n_theta)
        if (not np.issubdtype(type(n_phi),np.integer)):
            n_phi = size_dict['n_phi']
        else:
            n_phi = int(n_phi)
            
        #log.info(f'nlat = {n_theta} nphi = {n_phi}')
        sh.set_grid(polar_opt=0,flags=shtns.sht_gauss) #extra call needed otherwise there are sometimes errors in shtns grid generation
        n_theta,n_phi = sh.set_grid(nlat = n_theta,nphi=n_phi,polar_opt=0,flags=shtns.sht_gauss)            
        phis=2*np.pi*np.arange(n_phi)/(n_phi*sh.mres)
        thetas=np.arccos(sh.cos_theta)

        return thetas,phis
    
    def n_angular_step_from_max_order(self):
        max_order = self.bandwidth-1
        size_dict={}
        N = self.anti_aliazing_degree
        n_phi = 2**(int(np.log2((N+1)*max_order)) + 1)
        n_theta = n_phi//2
        size_dict['n_phi'] = n_phi
        size_dict['n_theta'] = n_theta
        return size_dict
    def _generate_forward_cmplx(self):
        analys_cplx = self._sh.analys_cplx
        ls = self.ls
        ms = self.ms
        def forward_cmplx_inner(data):            
            return np.array(tuple(analys_cplx(r_shell) for r_shell in data))
        cmplx_inner = shape_change_decorator(self.angular_shape,out_shape=(self.n_coeff,))(forward_cmplx_inner)
        def forward_cmplx(data):
            return ShCoeff(cmplx_inner(data),self.l_ids_complex,self.m_ids_complex,ls=ls,ms=ms,complex_data=True)
        return forward_cmplx
    def _generate_inverse_cmplx(self):
        synth_cplx = self._sh.synth_cplx
        def inverse_cmplx_inner(data):            
            return  np.array(tuple(synth_cplx(coeff) for coeff in data))
        inverse_cmplx = shape_change_decorator((self.n_coeff,),out_shape=self.angular_shape)(inverse_cmplx_inner)
        return inverse_cmplx
    def _generate_forward_real(self):
        r'''
        computes the real sphrical harmonic coefficients.
        The shtns only computes the complex harmonic coefficients $f_lm$ with $m>=0$ in real mode.
        The relal harmonic coefficients $F_lm$ with $l = 0,...,L$ and $|m|<=l$ are defined by
        $F_lm = \sqrt{2}Im(f_lm)$ for $m<0$
        $F_l0 = f_l0$ for $m=0$
        $F_lm = \sqrt{2}Re(f_lm)$ for $m>0$
        '''
        analys = self._sh.analys
        ls = self.ls
        ms = self.ms
        zero_m = self.shtns_real_zero_ms
        nonzero_m = ~self.shtns_real_zero_ms
        def forward_real_inner(data):
            temp = tuple(analys(r_shell.real) for r_shell in data)            
            return np.array(tuple(np.concatenate((coeff[zero_m].real,np.sqrt(2)*coeff.real[nonzero_m],np.sqrt(2)*coeff.imag[nonzero_m])) for coeff in temp))
        real_inner = shape_change_decorator(self.angular_shape,out_shape=(self.n_coeff,))(forward_real_inner)
        def forward_real(data):
            return ShCoeff(real_inner(data),self.l_ids_real,self.m_ids_real,ls=ls,ms=ms,complex_data=False)
        return forward_real
    def _generate_inverse_real(self):
        synth = self._sh.synth
        ls = self.ls
        ms = self.ms
        N = self._sh.nlm
        negzero_m = self.m_ids_real<=0
        def _real_to_complex(coeff):
            temp = coeff[:N] + 1.j*coeff[negzero_m]
            temp[self.shtns_real_zero_ms].imag =0
            temp[~self.shtns_real_zero_ms]/=np.sqrt(2)
            return temp
            
        def inverse_real_inner(data):            
            return  np.array(tuple(synth(_real_to_complex(coeff)) for coeff in data))
        inverse_real= shape_change_decorator((self.n_coeff,),out_shape=self.angular_shape)(inverse_real_inner)
        return inverse_real
    def _generate_forward_real_data_cmplx_coeff(self):
        r'''
        computes the complex sphrical harmonic coefficients for real input data.
        '''
        analys = self._sh.analys
        ls = self.ls
        ms = self.ms_real_data_complex_coeff
        def forward_real_data_cmplx_coeff_inner(data):            
            return np.array(tuple(analys(r_shell.real) for r_shell in data))
        real_inner = shape_change_decorator(self.angular_shape,
                                            out_shape=(self.n_coeff_real_data_complex_coeff,))(forward_real_data_cmplx_coeff_inner)
        def forward_real_data_cmplx_coeff(data):
            return ShCoeff(real_inner(data),
                           self.l_ids_real_data_complex_coeff,
                           self.m_ids_real_data_complex_coeff,
                           ls=ls,
                           ms=ms,
                           complex_data=False)
        return forward_real_data_cmplx_coeff
    def _generate_inverse_real_data_cmplx_coeff(self):
        synth = self._sh.synth
        def inverse_real_data_cmplx_coeff_inner(data):            
            return  np.array(tuple(synth(coeff) for coeff in data))
        inverse_cmplx = shape_change_decorator((self.n_coeff_real_data_complex_coeff,),
                                               out_shape=self.angular_shape)(inverse_real_data_cmplx_coeff_inner)
        return inverse_cmplx
    
    def forward(self,data):
        if np.iscomplexobj(data):
            return self.forward_cmplx(data)
        else:
            return self.forward_real_data_cmplx_coeff(data)
        
    def inverse(self,data):
        if data.complex_data:
            return self.inverse_cmplx(data)
        else:
            return self.inverse_real_data_cmplx_coeff(data)

    def get_empty_coeff(self,pre_shape=None,complex_data=True,real_harmonics=False):
        n_coeff = self.n_coeff
        dtype = complex
        if complex_data:
            l_ids = self.l_ids_complex
            m_ids = self.m_ids_complex
            complex_data=True
        else:
            complex_data=False
            if not real_harmonics:
                n_coeff = self.n_coeff_real_data_complex_coeff
                l_ids = self.l_ids_real_data_complex_coeff
                m_ids = self.m_ids_real_data_complex_coeff
            else:
                dtype=float
                l_ids = self.l_ids_real
                m_ids = self.m_ids_real
            
        if not isinstance(pre_shape,tuple):
            data = np.zeros(n_coeff,dtype=dtype)
            return ShCoeff(data,l_ids,m_ids,ls = self.ls,ms = self.ms,complex_data=complex_data)
        else:
            data = np.zeros(pre_shape+(n_coeff,),dtype=dtype)
            return ShCoeff(data,l_ids,m_ids,ls = self.ls,ms = self.ms,complex_data=complex_data)
        
    @property
    def grid(self):
        return GridFactory.construct_grid('uniform',(self.thetas,self.phis))

class sh(SphericalHarmonicTransformInterface):
    ShSmall = ShSmall
    ShCoeff = ShCoeff
    _sh = shtns
