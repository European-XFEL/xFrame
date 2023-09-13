import numpy as np
import logging
from xframe.library.mathLibrary import get_spherical_harmonic_transform_obj
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward
from xframe.library.mathLibrary import circularHarmonicTransform_complex_inverse
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward
from xframe.library.mathLibrary import circularHarmonicTransform_real_inverse
log=logging.getLogger('root')


class HarmonicTransform:    
    def __init__(self,data_type,opt):
        self.data_type=data_type
        self.opt=opt
        self.dim=opt['dimensions']
        ht,iht,grid_param,trf_by_indices = self.chose_transforms()
        self.transforms_by_indices = trf_by_indices
        self.forward = ht
        self.inverse = iht        
        self.grid_param = grid_param
        self.max_order = opt['max_order']
        
    @classmethod
    def from_data_array(cls,data_type,array):
        shape = array.shape
        dimension = array.ndim
        if dimension ==2:
            opt = {'dimensions':dimension,'max_order':False,'n_angular_points':shape[1]}
        else:
            opt = {'dimensions':dimension,'max_order':shape[1]-1,'n_phi':shape[2],'n_theta':shape[1]}
        return cls(data_type,opt)
                
    def chose_transforms(self):
        opt=self.opt
        data_type=self.data_type
        dim=self.dim
        
        if dim == 2:
            #size=opt['n_angular_points']
            #log.info(opt.keys())
            #size = opt['n_phi']
            max_order = opt.get('max_order',False)
            #log.info('n_phi afrom settings = {}'.format(size))
            if isinstance(max_order,bool):
                size = opt['n_angular_points']
            else:
                size = max_order*2+1
            #log.info(f'angular grid size = {size} mode {data_type}')
            if data_type=='complex':
                ht_forward = circularHarmonicTransform_complex_forward
                ht_inverse = circularHarmonicTransform_complex_inverse
                
            elif data_type == 'real':
                ht_forward = circularHarmonicTransform_real_forward            
                def ht_inverse(data):
                    #log.info('seize = {}')
                    return circularHarmonicTransform_real_inverse(data,size)
            #log.info(f'size {size}')
            trf_by_indices={'m':{'forward':ht_forward,'inverse':ht_inverse}}
            phis = np.arange(size)/size*2*np.pi
            grid_opt={'phis':phis}        
        elif dim == 3:
            l_max=int(opt['max_order'])
            anti_aliazing_degree = opt.get('anti_aliazing_degree',False)
            n_phi = opt.get('n_phi',0)
            n_theta = opt.get('n_theta',0)
            indices=opt.get('indices','lm')
            if data_type == 'complex':
                #log.info(f'input nphi = {n_phi}')
                if isinstance(anti_aliazing_degree,bool):
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='complex',n_phi=n_phi,n_theta=n_theta)
                else:
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='complex',anti_aliazing_degree = anti_aliazing_degree ,n_phi= n_phi,n_theta=n_theta)
                self.m_indices = sh.cplx_m_indices
                self.l_indices = sh.cplx_l_indices
                self.m = sh.m
                self.l = sh.l
                self.m_split_indices = sh.cplx_m_split_indices
                self.l_split_indices = sh.cplx_l_split_indices
                self.n_coeff = sh.n_coeff
            elif data_type == 'real':
                if isinstance(anti_aliazing_degree,bool):
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='real',n_phi=n_phi,n_theta=n_theta)
                else:
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='real',anti_aliazing_degree = anti_aliazing_degree,n_phi=n_phi,n_theta=n_theta)
                #raise NotImplementedError('real spherical harmonic transforms are not implemented right now.')
            self.test = sh.test
            self._sh = sh
            

            trf_by_indices={'lm':{'forward':sh.forward_l,'inverse':sh.inverse_l},'ml':{'forward':sh.forward_m,'inverse':sh.inverse_m},'direct':{'forward':sh.forward_d,'inverse':sh.inverse_d}}
            
            ht_forward=trf_by_indices[indices]['forward']
            ht_inverse=trf_by_indices[indices]['inverse']
            grid_opt={'phis':sh.phi,'thetas':sh.theta}
        return ht_forward,ht_inverse,grid_opt,trf_by_indices
