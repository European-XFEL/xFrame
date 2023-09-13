import numpy as np
import logging
log=logging.getLogger('root')

from xframe.library.mathLibrary import SphericalIntegrator,PolarIntegrator,RadialIntegrator


###### FSC/FRC #######
def FSC_bit_limit(target_SNR,grid):
    # calculates FRC/FSC limit (1/2 bit limit is given by target_SNR=1/2)
    # assumes uniform spherical/polar datagrid
    # I am following  DOI: 10.1016/j.jsb.2005.05.009
    # Title: Fourier shell correlation threshold criteria  Authors: Marin van Heel, Michael Schatz
    # See equations (14) and (17) for target_SNR = 1 and 1/2 cases

    points_per_shell = np.prod(grid.shape[1:])
    half_dataset_SNR = target_SNR/2
    x = half_dataset_SNR
    p = points_per_shell
    fsc_limit = (x+2*np.sqrt(x/p)+1/np.sqrt(p))/(1+x+2**np.sqrt(x/p))
    return fsc_limit

def _fsc(a1,a2):
    #Computes FSC/FRC assuming the input scattering_amplitudes are sampled on a uniform spherical/polar grid
    axes = tuple(range(1,a1.ndim))
    I1 = (a1*a1.conj()).real
    I2 = (a2*a2.conj()).real
    nominator = np.sum(a1*a2.conj(),axis = axes).real
    denominator = np.sqrt(np.sum(I1,axis=axes)*np.sum(I2,axis=axes)).real
    non_zero = denominator!=0
    fsc = np.ones(nominator.shape,dtype = float)
    fsc[non_zero]=nominator[non_zero]/denominator[non_zero]
    return fsc

def _fsc_single(a,b):
    #Computes FSC/FRC assuming the input scattering_amplitudes are sampled on a uniform spherical/polar grid
    axes = tuple(range(1,a1.ndim))
    nominator = np.sum(a*a.conj(),axis = axes).real
    denominator = np.sum(b*b.conj(),axis=axes).real
    non_zero = denominator!=0
    fsc = np.ones(a.shape,dtype = float)
    fsc[non_zero]=nominator[non_zero]/denominator[non_zero]
    return fsc

def FSC_two_data_halves(averaged_scattering_amplitudes):
    a1,a2 = averaged_scattering_amplitudes
    return _fsc(a1,a2)
def FSC_single_classical(averaged_scattering_amplitude,input_intensity):
    a = averaged_scattering_amplitude
    b = np.sqrt(input_intensity)
    return _fsc_single(a,b)
def FSC_single_fxs(averaged_scattering_amplitude,averaged_projected_scattering_amplitude):
    a = averaged_scattering_amplitude
    b = averaged_projected_scattering_amplitude
    return _fsc(a,b)


###### PRTF ######
# The classical definitions follows
# Paper: High-resolution ab initio three-dimensional x-ray diffraction microscopy (equation (23))
# Authors: Henry N. Chapman, et. al.
# DOI: https://doi.org/10.1364/JOSAA.23.001179
def PRTF(a1,a2,b1,b2,return_prtf_nd=False):
    axes = tuple(range(1,a1.ndim))
    prtf_nd = np.ones(a1.shape,dtype = complex)
    non_zero = (b1!=0) & (b2!=0)
    prtf_nd[non_zero] = (a1[non_zero]*a2[non_zero].conj())/(b1[non_zero]*b2[non_zero].conj())
    #log.info(np.average(prtf_nd,axis = axes ))
    prtf_nd[~non_zero & (a1!=0) & (a2!=0)]=0
    #log.info('after zero = {}'.format(np.average(prtf_nd,axis = axes )))
    prtf_nd = np.sqrt(prtf_nd)
    #log.info('after sqrt = {}'.format(np.average(prtf_nd,axis = axes )))
    prtf = np.average(prtf_nd,axis = axes )
    #log.info('prtf = {}'.format(prtf))
    prtf_std = np.std(prtf_nd,axis = axes )
    if return_prtf_nd:
        return prtf,prtf_std,prtf_nd
    else:
        return prtf,prtf_std

def PRTF_clasical(averaged_scattering_amplitude,input_intensity,return_prtf_nd=False):
    a = averaged_scattering_amplitude
    b = np.sqrt(input_intensity)
    return PRTF(a,a,b,b,return_prtf_nd=return_prtf_nd)
def PRTF_classical_two_data_halves(averaged_scattering_amplitudes,input_intensities,return_prtf_nd=False):
    a1,a2 = averaged_scattering_amplitudes
    b1,b2 = np.sqrt(input_intensity[0]),np.sqrt(input_intensity[1])
    return PRTF(a1,a2,b1,b2,return_prtf_nd=return_prtf_nd)

def PRTF_fxs(averaged_scattering_amplitude,averaged_intensity,averaged_projected_scattering_amplitude=False, averaged_projected_intensity=False,return_prtf_nd=False):
    if isinstance(averaged_projected_scattering_amplitude,np.ndarray) and isinstance(averaged_projected_intensity,np.ndarray):
        a1 = averaged_scattering_amplitude
        a2 = averaged_projected_scattering_amplitude
        b1 = np.sqrt(averaged_intensity)
        b2 = np.sqrt(averaged_projected_intensity)
        prtf_data = PRTF(a1,a2,b1,b2,return_prtf_nd=return_prtf_nd)
    else:
        a = averaged_scattering_amplitude
        b = np.sqrt(averaged_intensity)
        prtf_data = PRTF(a,a,b,b,return_prtf_nd=return_prtf_nd)
    return prtf_data

###### FQC ######
# The classical definitions follows (with a small change)
# Paper: Correlations in Scattered X-Ray Laser Pulses Reveal Nanoscale Structural Features of Viruses (equation (17) in Supplements)
# Authors: Ruslan P. Kurta, et. al.
# DOI: https://doi.org/10.1103/PhysRevLett.119.158102
# The change is that we leave out the zero's order harmonic coefficient in the summs


def FQC(cn1,cn2,return_cc=False,skip_odd_orders=False):
    #Calculates FQC based on harmonic coefficients of two input cross-correlations
    # Assumes format (q1,q1,harmonic_order)
    # Assumes that cn[...,0] contins the zero'th order coefficient
    if skip_odd_orders:        
        o_start=2
        o_step=2
    else:
        o_start=1
        o_step=1
        
    # symmetrize & select relevant orders
    c1 = ((cn1 + np.swapaxes(cn1,0,1))/2)[...,o_start::o_step]
    c2 = ((cn2 + np.swapaxes(cn2,0,1))/2)[...,o_start::o_step]

    # squares
    C1 = (c1*c1.conj()).real
    C2 = (c2*c2.conj()).real
    
    
    cc_nominator = np.sum(c1*c2.conj(),axis=-1).real
    cc_denominator = np.sqrt(np.sum(C1,axis = -1)*np.sum(C2,axis = -1)).real
    non_zero = cc_denominator != 0

    cc = np.ones(c1.shape[:2],dtype = float)
    cc[non_zero]=cc_nominator[non_zero]/cc_denominator[non_zero]

    qq_mask = np.tril(np.ones(cc.shape)).astype(bool)

    fqc = np.mean(cc,axis = 1,where=qq_mask)
    if return_cc:
        return fqc,cc
    else:
        return fqc

def FQCB_2D(bn1,bn2,return_2d_fqcb=False,skip_odd_orders=False,include_zero_order=False):
    #Calculates FQC based on harmonic coefficients of two input cross-correlations
    # Assumes format (q1,q1,harmonic_order)
    # Assumes that cn[...,0] contins the zero'th order coefficient
    
    if skip_odd_orders:        
        o_start=2
        o_step=2
    else:
        o_start=1
        o_step=1
    if include_zero_order:
        o_start = 0 
    o_stop=min(len(bn1),len(bn2))
        
    # symmetrize & select relevant orders
    b1 = ((bn1 + np.swapaxes(bn1,-1,-2))/2)[o_start:o_stop:o_step]
    b2 = ((bn2 + np.swapaxes(bn2,-1,-2))/2)[o_start:o_stop:o_step]
    #b1 = bn1[o_start:o_stop:o_step]
    #b2 = bn2[o_start:o_stop:o_step]
    
    # squares
    B1 = (b1*b1.conj()).real
    B2 = (b2*b2.conj()).real
    
    bb_nominator = np.sum(b1*b2.conj(),axis = 0).real
    bb_denominator = np.sqrt(np.sum(B1,axis = 0)*np.sum(B2,axis = 0)).real
    non_zero = bb_denominator != 0

    bb = np.ones(b1.shape[1:],dtype = float)
    #log.info('bb shape = {} nonzero shape {} nom shape ={} denom shape = {}'.format(bb.shape,non_zero.shape,bb_nominator.shape,bb_denominator.shape))
    bb[non_zero]=bb_nominator[non_zero]/bb_denominator[non_zero]
    bb = np.abs(bb)

    qq_mask = np.tril(np.ones(bb.shape)).astype(bool)

    fqc = np.mean(bb,axis = 1,where=qq_mask)
    std = np.std(bb,axis=1,where=qq_mask)
    if return_2d_fqcb:
        return fqc,std,bb
    else:
        return fqc,std
    
def FQCB_3D(bn1,bn2,return_2d_fqcb=False,skip_odd_orders=False,include_zero_order=False):
    #Calculates FQC based on harmonic coefficients of two input cross-correlations
    # Assumes format (q1,q1,harmonic_order)
    # Assumes that cn[...,0] contins the zero'th order coefficient
    raise NotImplementedError()
    if skip_odd_orders:        
        o_start=2
        o_step=2
    else:
        o_start=1
        o_step=1
    if include_zero_order:
        o_start = 0 
    o_stop=min(len(bn1),len(bn2))
        
    # symmetrize & select relevant orders
    b1 = ((bn1 + np.swapaxes(bn1,-1,-2))/2)[o_start:o_stop:o_step]
    b2 = ((bn2 + np.swapaxes(bn2,-1,-2))/2)[o_start:o_stop:o_step]
    #b1 = bn1[o_start:o_stop:o_step]
    #b2 = bn2[o_start:o_stop:o_step]
    
    # squares
    B1 = (b1*b1.conj()).real
    B2 = (b2*b2.conj()).real
    
    bb_nominator = np.sum(b1*b2.conj(),axis = 0).real
    bb_denominator = np.sqrt(np.sum(B1,axis = 0)*np.sum(B2,axis = 0)).real
    non_zero = bb_denominator != 0

    bb = np.ones(b1.shape[1:],dtype = float)
    #log.info('bb shape = {} nonzero shape {} nom shape ={} denom shape = {}'.format(bb.shape,non_zero.shape,bb_nominator.shape,bb_denominator.shape))
    bb[non_zero]=bb_nominator[non_zero]/bb_denominator[non_zero]
    bb = np.abs(bb)

    qq_mask = np.tril(np.ones(bb.shape)).astype(bool)

    fqc = np.mean(bb,axis = 1,where=qq_mask)
    std = np.std(bb,axis=1,where=qq_mask)
    if return_2d_fqcb:
        return fqc,std,bb
    else:
        return fqc,std
