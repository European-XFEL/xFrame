import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat
import time, math
import struct
import sys
import os
import scipy as sp
import scipy.stats as spst
#from scipy.fft import irfft,rfft
from numpy.fft import irfft,rfft
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
                 
    # calculate CCF and its FCs for a set of q1 != q2 rings. output is a (n_q1 * n_q * n_phi) matrix of CCF and FCs
    #
    def ccf_twopoint_q1_q2(self, data_polar1):
        #log.info(f'data contains Nan= {np.isnan(data_polar1).any()}')
        fc_I1q = rfft(data_polar1)
        np.multiply(np.conjugate(fc_I1q[self.q1vals_pos, None, :]), fc_I1q[None,self.q2vals_pos, :], out=self.fc_ccf_q1q2)
        ccf_q1q2=irfft(self.fc_ccf_q1q2,self.n_phi) #3D array, first index - q1, second - q2, third - 'angle phi'
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
        ccf_fcs=np.fft.fft(ccf_data)
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
                                                              
