import numpy as np
import logging

import xframe.library.mathLibrary as mLib
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward
from xframe.library.mathLibrary import circularHarmonicTransform_complex_inverse
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward
from xframe.library.mathLibrary import circularHarmonicTransform_real_inverse
from xframe.presenters.matplolibPresenter import heatPolar2D
from xframe.library.pythonLibrary import uniformGridGetStepSizes

from .ft_grid_pairs import get_grid
pres=heatPolar2D()
log=logging.getLogger('root')

##############
###general stuff###
def generatePolarIndexList(dataLength):
#        log.info('input data ={}'.format(inputData))
    n=dataLength
    lengthIsEven= n%2==0
    halfPolarIndex=np.arange(np.floor(n/2)+1)
    if lengthIsEven:
        polarIndex=np.concatenate((halfPolarIndex[:-1],-1*halfPolarIndex[:0:-1]))
        #            log.info('polar index ={}'.format(polarIndex))
    else:
        polarIndex=np.concatenate((halfPolarIndex,-1*halfPolarIndex[:0:-1]))
        #            log.info('polar index ={}'.format(polarIndex))
    return polarIndex


def calculateHarmonicOrder(angularIndex,nAngularSteps):
    harmonicOrderIndex=nAngularSteps-angularIndex
    orderIsNegative=harmonicOrderIndex<=nAngularSteps/2
    if orderIsNegative:
        harmonicOrder=-harmonicOrderIndex
    else:
        harmonicOrder = angularIndex
    return harmonicOrder


### Harmonic Transforms
def generate_cht_for_bessel(real_grid):
    phis=real_grid[0,0,:,1]
    N=len(phis)
    kernel=np.exp(-1.j*np.arange(N)[:,None]*phis[None,:])
    inverse_kernel=kernel.conjugate()
    
    def cht_complex_forward(data):
        return 1/N*np.sum(data*kernel[:,None,:],axis=2)
    def cht_complex_inverse(data):
        return np.sum(data*inverse_kernel[:,None,:],axis=2)

    return cht_complex_forward,cht_complex_inverse
        
        
def generate_circular_harmonic_transform_pair():
    pass


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
            size=opt['n_angular_points']
            grid_opt={}        
            if data_type=='complex':
                def ht_forward(data):
                    return circularHarmonicTransform_complex_forward(data)                
                def ht_inverse(data):
                    return circularHarmonicTransform_complex_inverse(data)
                
            elif data_type == 'real':
                def ht_forward(data):
                    return circularHarmonicTransform_real_forward(data)            
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
                    sh=mLib.get_spherical_harmonic_transform_obj(l_max,mode='complex',n_phi=n_phi,n_theta=n_theta)
                else:
                    sh=mLib.get_spherical_harmonic_transform_obj(l_max,mode='complex',anti_aliazing_degree = anti_aliazing_degree ,n_phi= n_phi,n_theta=n_theta)
                self.m_indices = sh.cplx_m_indices
                self.l_indices = sh.cplx_l_indices
                self.m = sh.m
                self.l = sh.l
                self.m_split_indices = sh.cplx_m_split_indices
                self.l_split_indices = sh.cplx_l_split_indices
                self.n_coeff = sh.n_coeff
            elif data_type == 'real':
                if isinstance(anti_aliazing_degree,bool):
                    sh=mLib.get_spherical_harmonic_transform_obj(l_max,mode='real')
                else:
                    sh=mLib.get_spherical_harmonic_transform_obj(l_max,mode='real',anti_aliazing_degree = anti_aliazing_degree)
            self.test = sh.test
            

            trf_by_indices={'lm':{'forward':sh.forward_l,'inverse':sh.inverse_l},'ml':{'forward':sh.forward_m,'inverse':sh.inverse_m},'direct':{'forward':sh.forward_d,'inverse':sh.inverse_d}}
            
            ht_forward=trf_by_indices[indices]['forward']
            ht_inverse=trf_by_indices[indices]['inverse']
            grid_opt={'phis':sh.phi,'thetas':sh.theta}
        return ht_forward,ht_inverse,grid_opt,trf_by_indices


### Spherical Harmonic Transform



