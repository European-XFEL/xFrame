import time
import numpy as np
from scipy.interpolate import griddata
import logging
import traceback

log=logging.getLogger('root')

from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe.library.gridLibrary import GridFactory
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library import units

def find_center_by_point_sym_overlap(data,grid,mask = True):
    log.info('\n mask_data dtype 3 ={}'.format(mask.dtype))
    log.info('input nonzero = {}'.format((data!=0).any()))
    n_x_points = grid.shape[0]
    n_y_points = grid.shape[1]
    zero_index =np.array(np.nonzero((np.isclose(grid[...,0],0))&(np.isclose(grid[...,1],0)))).flatten()
    shift = -zero_index +[n_x_points//2,n_y_points//2]

    data = np.roll(np.roll(data,shift[0],axis = 0),shift[1],axis=1) # correct off center origin for point inversion
    data = np.pad(data,((n_x_points,0),(n_y_points,0)))
    r_data = np.fft.fft2(data)
    convolution =  np.fft.ifft2(r_data*r_data).real
    
    if isinstance(mask,np.ndarray):
        log.info(mask.dtype)
        mask = ~mask
        mask = mask.astype(float)
        mask = np.roll(np.roll(mask,shift[0],axis = 0),shift[1],axis=1) # correct off center origin for point inversion
        mask = np.pad(mask,((n_x_points,0),(n_y_points,0)))
        r_mask = np.fft.fft2(mask)
        mask_convolution =  np.fft.ifft2(r_mask*r_mask).real
        non_zero_mask = mask_convolution!=0
        convolution[non_zero_mask] /= mask_convolution[non_zero_mask]
    log.info('convolution shape ={}'.format(convolution.shape))

    x_len=grid[:,0,0].max()-grid[:,0,0].min()
    y_len=grid[0,:,1].max()-grid[0,:,1].min()
    x_step=np.abs(grid[1,0,0] - grid[0,0,0])
    y_step=np.abs(grid[0,1,1] - grid[0,0,1])
    log.info("len/step x: {} , y:{}".format(x_len/x_step,y_len/y_step))

    n_conv = np.floor((2*x_len/x_step+2 +0.5,2*y_len/y_step+2 +0.5))
    conv_x = -x_len-x_step + np.arange(n_conv[0])*x_step
    conv_y = -y_len-y_step + np.arange(n_conv[1])*y_step
    conv_grid = GridFactory.construct_grid('uniform',[conv_x,conv_y])
    log.info('conv_grid shape ={}'.format(conv_grid.shape))

    #log.info('argmax convolution ={}'.format(np.argmax(convolution)))
    max_index = np.unravel_index(np.argmax(convolution),convolution.shape)
    #log.info('max index ={}'.format(max_index))
    max_grid_point = conv_grid.array[max_index]
    log.info('center ={}'.format(max_grid_point))

    max_area_sigma_mask = (convolution > convolution.max()*np.exp(-.5))    
    sigma_grid_points = conv_grid[max_area_sigma_mask,:]-max_grid_point    
    log.info('sigma gridpoints shape = {}'.format(sigma_grid_points.shape))
    if sigma_grid_points.shape[0] != 0:
        one_sigma_deviation = np.array(((sigma_grid_points[...,0].min(),sigma_grid_points[...,0].max()),(sigma_grid_points[...,1].min(),sigma_grid_points[...,1].max())))
    else:
        one_sigma_deviation = np.array([[np.inf, np.inf],[np.inf,np.inf]])


    out_dict = {}
    out_dict['convolution'] = convolution
    out_dict['convolution_grid'] = conv_grid
    out_dict['max'] = max_grid_point
    out_dict['center'] = max_grid_point/2
    out_dict['one_sigma_mask'] = max_area_sigma_mask    
    out_dict['one_sigma_deviations'] = one_sigma_deviation/2
    return out_dict
    

def find_center_of_mass(data,points):
    dim = points.shape[-1]
    return np.mean(data.flatten()*points.reshape(-1,dim) )/np.sum(data)
