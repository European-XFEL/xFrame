import numpy as np
import logging

log=logging.getLogger('root')

from xframe.library import mathLibrary as mLib
from xframe.library.mathLibrary import PolarIntegrator,SphericalIntegrator
from xframe.library.mathLibrary import gaussian_fourier_transformed_spherical
from scipy.optimize import minimize_scalar

class ShrinkWrapParts:
    def __init__(self,initial_support,real_grid,reciprocal_grid,threshold,gaussian_sigma,mode = 'threshold',mode_options = {}):
        dimension = real_grid[:].shape[-1]
        self.mode_routines = {'threshold':self.generate_get_new_mask_threshold,'fixed_volume':self.generate_get_new_mask_fixed_volume}
        self.mode_options
        self.mode = mode
        self.real_grid = real_grid[:]
        self.reciprocal_grid = reciprocal_grid[:]
        dimension = self.real_grid.shape[-1]
        log.info(f'SW dim = {dimension}')
        if dimension ==2:
            self.integrator = PolarIntegrator(self.real_grid)            
            self._default_gaussian_sigma = real_grid[1,0,0]-real_grid[0,0,0]
        elif dimension == 3:
            self.integrator = SphericalIntegrator(self.real_grid)
            self._default_gaussian_sigma = real_grid[1,0,0,0]-real_grid[0,0,0,0]
        self.initial_volume = self.integrator.integrate_normed(initial_support.astype(float))
        
        self._threshold = [threshold]
        
        self._gaussian_sigma = [gaussian_sigma]
        self.gaussian_values = gaussian_fourier_transformed_spherical(self.grid,self._gaussian_sigma[0])
        
        self.get_new_mask = self.mode_routines.get[self.mode]()
        self.multipy_with_ft_gaussian = self.generate_multiply_by_ft_gaussian()
    @property
    def threshold(self):
        return self._threshold[0]
    @threshold.setter
    def threshold(self,value):
        if value<=0:
            self._threshold[0]=0
            log.warning('Shrikwrap threshold has to lie in [0,1] but given value is {}. Projecting threshold to {}.'.format(value,self._threshold[0]))
        elif value>=1:
            self._threshold[0]=1
            log.warning('Shrikwrap threshold has to lie in [0,1] but given value is {}. Projecting threshold to {}.'.format(value,self._threshold[0]))
        else:
            self._threshold[0]=value


    @property
    def gaussian_sigma(self):
        return self._gaussian_sigma[0]
    @gaussian_sigma.setter
    def gaussian_sigma(self,value):
        if value>=0:
            self._gaussian_sigma[0]=value
        else:
            self._gaussian_sigma[0]=0
            log.warning('Gaussian sigma has to be grater than 0 but given value is {}. Projecting threshold to {}.'.format(value,self._gaussian_sigma[0]))
        self.gaussian_values[:] = gaussian_fourier_transformed_spherical(self.grid,self._gaussian_sigma[0])
        
    def generate_get_new_mask_threshold(self):
        threshold = self.threshold
        def get_new_mask(convolution_data):
            abs_array=convolution_data
            max_value= abs_array.max()
            min_value= abs_array.min()
            diff = max_value-min_value
            new_mask = abs_array >= min_value + threshold[0]*diff
            return new_mask
        return update_mask
    
    def generate_get_new_mask_fixed_volume(self):
        target_volume = self.initial_volume*self.mode_options['volume'] # number in [0,1] indication volume fraction relative to self.initial_volume
        integrate = self.integrator.integrate_normed
        def get_new_mask(convolution_data):
            abs_array=convolution_data
            max_value= abs_array.max()
            min_value= abs_array.min()
            diff = max_value-min_value
            def new_volume(threshold):                
                new_mask = abs_array >= min_value + threshold*diff
                new_volume = abs(integrate(new_mask.astype(float))-target_volume)
            opti_result = minimize_scalar(new_volume,bounds=[0,1])
            threshold = opti_result.x
            log.info('Optimization result = {}'.format(opti_result.message))
            new_mask = abs_array >= min_value + threshold*diff
            return new_mask
        return update_mask

    
    def generate_multiply_by_ft_gaussian(self):
        gaussian_values = self.gaussian_values
        def multiply_ft_gaussian(data):
            return data*gaussian_values
        return multipy_with_ft_gaussian


    
def generateSW_SupportMaskRoutine(threshold):
    def generateSupportMask(convolutionData):
        absArray=np.abs(convolutionData)
        maxValue=np.max(absArray)
#        log.info('max value convoluted data={} max data value={}'.format(maxValue,np.max(absArray)))
        supportMask=absArray>=threshold*maxValue
#        positivity_mask=real_grid>=0
#        log.info('support mask={}'.format(supportMask))
        return supportMask
    return generateSupportMask

def generateSW_multiply_ft_gaussian(sigma,grid):
    dim=grid.total_shape[-1]
    #reciprocal grid needed
    if dim == 2:        
        gaussianArray=gaussian_fourier_transformed_spherical(grid,sigma)
    elif dim == 3:
        gaussianArray=gaussian_fourier_transformed_spherical(grid,sigma)
        
    else:
        e=AssertionError('multiply_ft_gaussion currently only implemented for dim=2,3')
        log.error(e)
        raise e
    
    def multiply_ft_gaussian(data):
        #            pres.present(gaussianArray,grid=ft_grid_pair.realGrid,layout={'title':'gaussian to multiply'})
        return data*gaussianArray
    return multiply_ft_gaussian



def SW_initialSupportMask(fxsData,threshold,realGrid,constantMaxR=False):
    if not isinstance(constantMaxR,bool):
        log.info('constantMaxR={}'.format(constantMaxR))
        supportMask=np.where(realGrid[:][:,0]<constantMaxR,True,False).reshape(realGrid.shape)
    else:
        maxValue=np.max(fxsData.ACD)
        #log.info('max value convoluted data={} max data value={}'.format(maxValue,np.max(absArray)))
        supportMask=absArray>=threshold*maxValue
    log.info('support mask[:]={}'.format(supportMask))
#    pres=heatPolar2D()
#    pres.present(realGrid,supportMask)
    return supportMask
