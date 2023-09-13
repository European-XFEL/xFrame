import numpy as np
import logging
log = logging.getLogger("root")
from xframe.library.interfaces import SoftInterface
import pysofft
from pysofft.make_wiegner import CosEvalPts,CosEvalPts2,SinEvalPts,SinEvalPts2,genWigTrans_L2
from pysofft.make_wiegner import genWigAll,genWigAllTrans,get_euler_angles
from pysofft.wignerTransform import wigNaiveSynthesis_fftw
from pysofft.wignerWeights import makeweights2
from pysofft.soft import Inverse_SO3_Naive_fft_pc,Forward_SO3_Naive_fft_pc,coefLoc_so3,sampLoc_so3,totalCoeffs_so3
from pysofft.soft import sampLoc_so3, calc_mean_C_array
from pysofft.rotate import rotate_coeff_multi,rotate_coeff
from xframe.library.gridLibrary import GridFactory
from pysofft import soft as _soft


class Soft(SoftInterface):
    def __init__(self,bw):
        self.bw = bw
        self._soft = _soft
        tmp = self.generate_data()
        self.grid = self.make_SO3_grid()
        self.wigners = tmp[0]
        self.wigners_transposed = tmp[1] #small d matrices with m1,m2 swapped
        self.legendre_weights = tmp[2]
        self.n_coeffs = totalCoeffs_so3(bw)
        self.n_points = (2*bw)**3
        
    def generate_data(self):
        band_width = self.bw
        wigners = genWigAll(band_width)
        wigners_transposed = genWigAllTrans(band_width)
        legendre_weights = makeweights2(band_width)
        return wigners, wigners_transposed, legendre_weights
    
    def forward_cmplx(self,data):        
        if data.ndim>1:
            data=data.flatten()
        if (data.dtype != np.dtype(np.complex128)):
            log.warning('input dtype is {} but {} is required. Trying to change dtype.'.format(data.dtype,np.dtype(np.complex128)))
            data = data.astype(np.complex128)
        if (len(data) != self.n_points):
            raise AssertionError('input data needs to be of length {}. Given length = {}'.format(self.n_points,data.dtype,len(data)))        
        coeffs = Forward_SO3_Naive_fft_pc(self.bw,data,self.legendre_weights,self.wigners,True)        
        return coeffs
    
    def inverse_cmplx(self,coeff):
        if (coeff.dtype != np.dtype(np.complex128)):
            log.warning('input dtype is {} but {} is required. Trying to change dtype.'.format(coeff.dtype,np.dtype(np.complex128)))
            coeff = coeff.astype(np.complex128)
        if (len(coeff) != self.n_coeffs) or (coeff.dtype != np.dtype(np.complex128)):
            raise AssertionError('input coeffs need to be of type complex128 and of length {}. Given data dtype = {} length = {}'.format(self.n_coeffs,coeff.dtype,len(coeff)))        
        data = Inverse_SO3_Naive_fft_pc(self.bw,coeff,self.wigners_transposed,True)        
        return data

    def make_SO3_grid(self):
        alpha,beta,gamma = get_euler_angles(self.bw)
        grid = GridFactory.construct_grid('uniform',[alpha,beta,gamma])[:]
        return grid
    

    def get_empty_coeff(self):
        return np.zeros(self.n_coeffs,dtype = complex)
        


    #################################################
    # computes the rotated harmonic coefficients
    # f_{l,m} ---> \sum_n D^l_{n,m} f_{l,n}
    # assumes f_{l,m}=coeff to be a complex array of length bw**2
    # split_ids such that np.split(coeff,split_ids) gives a list of coefficients indexed by l.
    # euler_angles is an array of length 3 containing the euler angles (alpha,beta,gamma) in ZYZ format
    def rotate_coeff(self,coeff,split_ids,euler_angles):
        #log.info('coeff.shape before rotation = {}'.format(coeff[0].shape))
        rotated_coeff = rotate_coeff_multi(self.bw, coeff,split_ids, euler_angles)
        return rotated_coeff

    def rotate_coeff_single(self,coeff,split_ids,euler_angles):
        log.info('coeff.shape before rotation = {}'.format(coeff.shape))
        rotated_coeff = rotate_coeff(self.bw, coeff,split_ids, euler_angles)
        return rotated_coeff

    
    #######################################
    # let f,g be two square integrable functions on the 2 sphere 
    # Define C: SO(3) ---> R,   R |---> <f,g \circ R> = \int_{S^2} dx f(x)*\overline{g(Rx)}
    # This function calculates C(R) for all R defined in make_SO3_grid
    #
    # arguments :
    #   f_coeff: f_{l,m} spherical harmonic coefficients of f 
    #   g_coeff: g_{l,m} spherical harmonic coefficients of g 
    #            f_coeff, g_coeff are complex numpy arrays of shape bw*(bw+1)+bw+1
    #   split _ids: ids that split coefficients in 2*bw+1 sub arrays indexed by m
    #
    def calc_mean_C(self,f_coeff,g_coeff,r_split_ids,ml_split_ids):
        r_split_lower=r_split_ids[0]
        r_split_upper=r_split_ids[1]
        mean_C = calc_mean_C_array(self.bw,f_coeff,g_coeff,r_split_lower,r_split_upper,ml_split_ids,self.wigners_transposed,True)
        return mean_C


    #testing 
    def combine_coeffs(self,f_coeff,g_coeff,split_ids):
        return combine_harmonic_coeffs(self.bw,f_coeff,g_coeff,split_ids)
