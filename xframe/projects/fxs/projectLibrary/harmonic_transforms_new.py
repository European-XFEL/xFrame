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

    def chose_transforms(self):
        opt=self.opt
        data_type=self.data_type
        dim=self.dim
        
        if dim == 2:
            #size=opt['n_angular_points']
            size = opt['n_phi']
            grid_opt={}        
            if data_type=='complex':
                ht_forward = circularHarmonicTransform_complex_forward
                ht_inverse = circularHarmonicTransform_complex_inverse
                
            elif data_type == 'real':
                ht_forward = circularHarmonicTransform_real_forward            
                def ht_inverse(data):
                    return circularHarmonicTransform_real_inverse(data,size)                
            trf_by_indices={'m':{'forward':ht_forward,'inverse':ht_inverse}}
            
        elif dim == 3:
            l_max=int(opt['max_order'])
            anti_aliazing_degree = opt.get('anti_aliazing_degree',False)
            n_phi = opt['n_phi']
            n_theta = opt['n_theta']
            indices=opt.get('indices','lm')
            if data_type == 'complex':
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
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='real')
                else:
                    sh=get_spherical_harmonic_transform_obj(l_max,mode='real',anti_aliazing_degree = anti_aliazing_degree)
                raise NotImplementedError('real spherical harmonic transforms are not implemented right now.')
            self.test = sh.test
            

            trf_by_indices={'lm':{'forward':sh.forward_l,'inverse':sh.inverse_l},'ml':{'forward':sh.forward_m,'inverse':sh.inverse_m},'direct':{'forward':sh.forward_d,'inverse':sh.inverse_d}}
            
            ht_forward=trf_by_indices[indices]['forward']
            ht_inverse=trf_by_indices[indices]['inverse']
            grid_opt={'phis':sh.phi,'thetas':sh.theta}
        return ht_forward,ht_inverse,grid_opt,trf_by_indices





