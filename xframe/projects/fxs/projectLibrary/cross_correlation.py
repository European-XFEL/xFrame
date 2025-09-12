import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat
import time, math
import struct
import sys
import os
import scipy as sp
import scipy.stats as spst
from xframe.library.math_transforms import PolarHarmonicTransform
#from scipy.fft import irfft,rfft
#from numpy.fft import irfft,rfft
import logging
log = logging.getLogger("root")
Pi = math.pi

# cross-correlation analysis of the images
class ccf_analysis:        
    def __init__(self, n_q1, n_q2, n_phi, q1vals_pos, q2vals_pos):
        self.n_q1=n_q1 
        self.n_q2=n_q2 
        self.n_phi=n_phi 
        self.q1vals_pos=q1vals_pos    
        self.q2vals_pos=q2vals_pos
        self.fc_ccf_q1q2 = np.zeros((self.n_q1, self.n_q2, self.n_phi//2+1), dtype=np.complex128)
        self.cht = PolarHarmonicTransform(max_order=n_phi//2)
        self.rcht = self.cht.forward_real
        self.ircht = self.cht.inverse_real
    # calculate CCF and its FCs for a set of q1 != q2 rings. output is a (n_q1 * n_q * n_phi) matrix of CCF and FCs
    #
    def ccf_twopoint_q1_q2(self, data_polar1):
        #log.info(f'data contains Nan= {np.isnan(data_polar1).any()}')
        #fc_I1q = rfft(data_polar1)
        fc_I1q = self.rcht(data_polar1)
        np.multiply(np.conjugate(fc_I1q[self.q1vals_pos, None, :]), fc_I1q[None,self.q2vals_pos, :], out=self.fc_ccf_q1q2)
        #ccf_q1q2=irfft(self.fc_ccf_q1q2,self.n_phi) #3D array, first index - q1, second - q2, third - 'angle phi'
        ccf_q1q2=self.ircht(self.fc_ccf_q1q2) #3D array, first index - q1, second - q2, third - 'angle phi'
        #log.info(f'correlation contains Nans = {np.isnan(ccf_q1q2).any()}')
        return  ccf_q1q2 

    # correct the CCF of the data by the CCF of the mask  
    #      
    def ccf_mask_correction(self, ccf_data, ccf_mask):
        ccf_data=ccf_data.real
        ccf_mask=ccf_mask.real
        nonzero_mask = (ccf_mask!=0)
        np.divide(ccf_data, ccf_mask, out=ccf_data, where=nonzero_mask)
        return ccf_data,nonzero_mask

    
    # Fourier components of the CCF 
    #
    def ccf_fcs(self, ccf_data):
        ccf_fcs=self.rcht(ccf_data)
        return ccf_fcs
    
    
    # mask-corrected two-point ccf 
    # 
    def ccf_twopoint_q1_q2_mask_corrected(self, image_pol, mask_pol):
        #log.info(f'data contains Nan= {np.isnan(image_pol).any()}')
        #log.info(f'no values are masked= {mask_pol.any()}')
        ccf_data=self.ccf_twopoint_q1_q2(image_pol)
        ccf_mask=self.ccf_twopoint_q1_q2(mask_pol)
        ccfcorrected,correction_mask=self.ccf_mask_correction(ccf_data, ccf_mask)
        return ccfcorrected,correction_mask
               
                           
    # perform symmetrization of the two-point ccf; used to correct for noisy at delta=0 or delta=2pi in the case of flat Ewald sphere
    #                       
    def symmetrize_ccf(self, ccf, posPi2, posPi, pos3Pi2):
        ccf_symmetric=np.empty_like(ccf)
        ccf_symmetric[...]=ccf[...]
        Nq2=ccf.shape[0]
        Nq1=ccf.shape[1]
        Nphi=ccf.shape[2]
        for i1 in range(Nq2):
            for j1 in range(Nq1):   
                ccf_symmetric[i1,j1,0:posPi2]=ccf[i1,j1,posPi:posPi+posPi2]
                ccf_symmetric[i1,j1,pos3Pi2+1:Nphi]=ccf[i1,j1,pos3Pi2+1-posPi:Nphi-posPi]
    
        return ccf_symmetric                        



class ccfAnalysis:        
    def __init__(self, n_q, n_phi):
        self.n_q=n_q 
        self.n_phi=n_phi 
        self.cht = PolarHarmonicTransform(max_order=n_phi//2)
        self.rcht = self.cht.forward_real
        self.ircht = self.cht.inverse_real
    # calculate CCF and its FCs for a set of q1 != q2 rings. output is a (n_q1 * n_q * n_phi) matrix of CCF and FCs
    #
    def ccf_twopoint_q1_q2(self, data_polar1):
        fc_I1q = self.rcht(data_polar1)
        fc_ccf_q1q2 = np.multiply(np.conjugate(fc_I1q[ :,None, :]), fc_I1q[None,:, :])
        ccf_q1q2=self.ircht(fc_ccf_q1q2) #3D array, first index - q1, second - q2, third - 'angle phi'
        return  ccf_q1q2

    def ccf_twopoint_q1_q2_symmetry(self, data_polar1):
        '''
        Uses the following symmetry relation C(q1,q2,phi) = C(q2,q1,-phi) which is always true.
        '''
        fc_I1q = self.rcht(data_polar1)
        ccf = np.zeros((self.n_q,self.n_q,self.n_phi),dtype=float)
        for q1 in range(self.n_q):
            ccf[q1,q1:]=self.ircht(fc_I1q[q1,None,:].conjugate()*fc_I1q[q1:])
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
        return  ccf

    # correct the CCF of the data by the CCF of the mask  
    #      
    def ccf_mask_correction(self, ccf_data, ccf_mask):
        ccf_data=ccf_data.real
        ccf_mask=ccf_mask.real
        
        # ccf_mask shoud only contain multiples of 1/n_phis as values
        # make sure there are no values close to 0 like 1e-16 due to rounding errors.
        n_phis = ccf_mask.shape[-1]
        # Use 1/(2*n_phis) as threshold instead of 1/n_phi to be insensitve to rounding errors. 
        nonzero_mask = (ccf_mask>=1/(2*n_phis))
        np.divide(ccf_data, ccf_mask, out=ccf_data, where=nonzero_mask)
        ccf_data[~nonzero_mask]=0
        return ccf_data,nonzero_mask

    
    # Fourier components of the CCF 
    #
    def ccf_fcs(self, ccf_data):
        ccf_fcs=self.rcht(ccf_data)
        return ccf_fcs
    
    
    # mask-corrected two-point ccf 
    # 
    def ccf_twopoint_q1_q2_mask_corrected(self, image_pol, mask_pol,use_symmetry=True):
        #log.info(f'data contains Nan= {np.isnan(image_pol).any()}')
        #log.info(f'no values are masked= {mask_pol.any()}')
        if use_symmetry:
            ccf_routine = self.ccf_twopoint_q1_q2_symmetry
        else:
            ccf_routine = self.ccf_twopoint_q1_q2
        ccf_data=ccf_routine(image_pol)
        ccf_mask=ccf_routine(mask_pol)
        ccfcorrected,correction_mask=self.ccf_mask_correction(ccf_data, ccf_mask)
        return ccfcorrected,correction_mask


    def ccf_twopoint_q1_q2_mask_corrected_fast(self, image_pol, mask_pol):
        # Compute harmonic coefficients of image and mask
        fc_I = self.rcht(image_pol)
        fc_I_conj = fc_I.conjugate()
        fc_m = self.rcht(mask_pol.astype(float))
        fc_m_conj = fc_m.conjugate()
        
        # Allocate Memory needed for the computation
        temp_ccf_I = np.zeros((self.n_q,self.n_phi),dtype=float)
        temp_ccf_m = np.zeros((self.n_q,self.n_phi),dtype=float)
        temp_ccn_I = np.zeros((self.n_q,fc_I.shape[-1]),dtype=complex)
        temp_ccn_m = np.zeros((self.n_q,fc_m.shape[-1]),dtype=complex)
        ccf = np.zeros((self.n_q,self.n_q,self.n_phi),dtype=float)
        ccf_mask = np.zeros((self.n_q,self.n_q,self.n_phi),dtype=bool)

        #map numpy methods
        mult = np.multiply
        divide = np.divide
        inverse_harm_transform = self.ircht
        n_phi = self.n_phi
        # start loop over q1 of C(q1,q2,phi)
        for q1 in range(self.n_q):
            # Compute parts of the cross correlation of image and mask
            mult(fc_I_conj[q1,None,:],fc_I[q1:],temp_ccn_I[q1:])
            mult(fc_m_conj[q1,None,:],fc_m[q1:],temp_ccn_m[q1:])
            inverse_harm_transform(temp_ccn_I[q1:],n_points=n_phi,out = temp_ccf_I[q1:])
            inverse_harm_transform(temp_ccn_m[q1:],n_points=n_phi,out = temp_ccf_m[q1:])

            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            ccf_mask[q1,q1:]=temp_ccf_m[q1:]>1/n_phi
            # correct the computed image cross correlation by dividing out the mask correlation
            divide(temp_ccf_I[q1:],temp_ccf_m[q1:],where = ccf_mask[q1,q1:],out=ccf[q1,q1:])

            # Use the Symmetrie C(q1,q2,phi)=C(q2,q1,-phi)
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
            ccf_mask[q1:,q1,0] = ccf[q1,q1:,0]
            ccf_mask[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
        return ccf,ccf_mask


    def ccf_twopoint_q1_q2_mask_corrected_fast2(self, image_pol, mask_pol):
        # Compute harmonic coefficients of image and mask
        fc_I = self.rcht(image_pol)
        fc_I_conj = fc_I.conjugate()
        fc_m = self.rcht(mask_pol.astype(float))
        fc_m_conj = fc_m.conjugate()
        
        # Allocate Memory needed for the computation
        temp_ccf_I = np.zeros((self.n_q,self.n_phi),dtype=float)
        temp_ccf_m = np.zeros((self.n_q,self.n_phi),dtype=float)
        temp_ccn_I = np.zeros((self.n_q,fc_I.shape[-1]),dtype=complex)
        temp_ccn_m = np.zeros((self.n_q,fc_m.shape[-1]),dtype=complex)
        ccf = np.zeros((self.n_q,self.n_q,self.n_phi),dtype=float)
        ccf_mask = np.zeros((self.n_q,self.n_q,self.n_phi),dtype=bool)

        #map numpy methods
        mult = np.multiply
        divide = np.divide
        inverse_harm_transform = self.ircht
        n_phi = self.n_phi
        # start loop over q1 of C(q1,q2,phi)
        for q1 in range(self.n_q):
            # Compute parts of the cross correlation of image and mask
            mult(fc_I_conj[q1,None,:],fc_I[q1:],temp_ccn_I[q1:])
            mult(fc_m_conj[q1,None,:],fc_m[q1:],temp_ccn_m[q1:])
            inverse_harm_transform(temp_ccn_I[q1:],n_points=n_phi,out = temp_ccf_I[q1:])
            inverse_harm_transform(temp_ccn_m[q1:],n_points=n_phi,out = temp_ccf_m[q1:])

            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            ccf_mask[q1,q1:]=temp_ccf_m[q1:]>1/n_phi
            # correct the computed image cross correlation by dividing out the mask correlation
            divide(temp_ccf_I[q1:],temp_ccf_m[q1:],where = ccf_mask[q1,q1:],out=ccf[q1,q1:])

            # Use the Symmetrie C(q1,q2,phi)=C(q2,q1,-phi)
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
            ccf_mask[q1:,q1,0] = ccf[q1,q1:,0]
            ccf_mask[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
        return ccf,ccf_mask


    # perform symmetrization of the two-point ccf; used to correct for noisy at delta=0 or delta=2pi in the case of flat Ewald sphere
    #                       
    def symmetrize_ccf(self, ccf, posPi2, posPi, pos3Pi2):
        ccf_symmetric=np.empty_like(ccf)
        ccf_symmetric[...]=ccf[...]
        Nq2=ccf.shape[0]
        Nq1=ccf.shape[1]
        Nphi=ccf.shape[2]
        for i1 in range(Nq2):
            for j1 in range(Nq1):   
                ccf_symmetric[i1,j1,0:posPi2]=ccf[i1,j1,posPi:posPi+posPi2]
                ccf_symmetric[i1,j1,pos3Pi2+1:Nphi]=ccf[i1,j1,pos3Pi2+1-posPi:Nphi-posPi]
    
        return ccf_symmetric                        



