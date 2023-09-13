import numpy as np
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward as ht
from xframe.library.mathLibrary import circularHarmonicTransform_real_inverse as iht

def generate_calc_cross_correlation(data_shape,pixels_per_radial_ring,mask_type='q_dependend'):    
    n_radial_points = data_shape[0]
    n_angular_points = data_shape[1]
    n_orders = n_angular_points//2 + 1
    coeff_mask = np.ones((n_radial_points,n_orders),dtype=bool)
    inverse_coeff_mask = ~coeff_mask
    orders_per_radial_ring = pixels_per_radial_ring//2 + 1
    for mask_part,max_order in zip(coeff_mask, orders_per_radial_ring):
        mask_part[max_order + 1:]=False
        
    if mask_type == 'q1q2_dependend':        
        def masked_cc(data,mask):
            #assumes mask(n,q1,q2,phi) = mask(n,q2,q1,phi) => mask(n,q1,q2,phi) * mask(n,q2,q1,phi) = mask(n,q1,q2,phi)
            data_harm_coeff = np.zeros(mask.shape,dtype=complex)
            mask_harm_coeff = np.array(data_harm_coeff)
            data_harm_coeff[...,coeff_mask]= ht(data[:,None,...]*mask)[...,coeff_mask]
            data_harm_coeff[...,coeff_mask]= ht(mask)[...,coeff_mask]
            data_cc_coeff = data_harm_coeff*np.swapaxes(data_harm_coeff,1,2).conj()
            mask_cc_coeff = mask_harm_coeff*mask_harm_coeff.conj()
            data_cc = iht(data_cc_coeff,n_angular_points)
            mask_cc = iht(mask_cc_coeff,n_angular_points)
            cc = data_cc/mask_cc
            return cc 
    elif mask_type == 'q_dependend':
        def masked_cc(data,mask):
            data_harm_coeff = np.zeros(mask.shape,dtype=complex)
            mask_harm_coeff = np.array(data_harm_coeff)
            data_harm_coeff[...,coeff_mask]= ht(data*mask)[...,coeff_mask]
            data_harm_coeff[...,coeff_mask]= ht(mask)[...,coeff_mask]
            data_cc_coeff = data_harm_coeff[:,:,None,:]*(data_harm_coeff.conj())[:,None,:,:]
            mask_cc_coeff = mask_harm_coeff[:,:,None,:]*(mask_harm_coeff.conj())[:,None,:,:]
            data_cc = iht(data_cc_coeff,n_angular_points)
            mask_cc = iht(mask_cc_coeff,n_angular_points)
            cc = data_cc/mask_cc
            return cc 
    elif mask_type == 'unmasked':
        def masked_cc(data):
            data_harm_coeff = ht(data)
            data_harm_coeff[...,inverse_coeff_mask]=0
            data_cc_coeff = data_harm_coeff[...,None,:]*(data_harm_coeff.conj())[...,None,:,:]
            cc = iht(data_cc_coeff,n_angular_points)
            return cc 
    return masked_cc
