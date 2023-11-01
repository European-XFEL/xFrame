import numpy as np
import traceback
import logging
log = logging.getLogger("root")
from xframe.library.interfaces import SoftInterface
import pysofft
from pysofft.make_wiegner import CosEvalPts,CosEvalPts2,SinEvalPts,SinEvalPts2,genWigTrans_L2
from pysofft.make_wiegner import genWigAll,genWigAllTrans,get_euler_angles,wig_little_d_lookup
from pysofft.wignerTransform import wigNaiveSynthesis_fftw
from pysofft.wignerWeights import makeweights2
from pysofft.soft import Inverse_SO3_Naive_fft_pc,Forward_SO3_Naive_fft_pc,coefLoc_so3,sampLoc_so3,totalCoeffs_so3
from pysofft.soft import sampLoc_so3, calc_mean_C_array,integrate_over_so3,integrate_over_so3_normalized,zeros_order_forward_so3,get_so3_probabily_weights,wigner_normalization_factors,wigner_normalization_factor
from pysofft.rotate import rotate_coeff_multi,rotate_coeff
from xframe.library.gridLibrary import GridFactory
from pysofft import soft as _soft


class WigSmallCoefView:
    def __init__(self,array,d_to_coeff_info,bw):        
        self.array = array
        self.lookup = d_to_coeff_info[1]
        self.n_beta = 2*bw
    def __getitem__(self,items):
        init_id,step,sign = self.lookup[items]
        stop_id = init_id+step*self.n_beta        
        return self.array[init_id:stop_id:step]*sign/wigner_normalization_factor(items[0])
        
class WigSmallAngleView:
    def __init__(self,small_d,d_to_coeff_info,bw):        
        self.array = small_d
        self.normalization_factors = wigner_normalization_factors(bw)
        self.d_to_coeff_ids = d_to_coeff_info[0]        
        self.n_beta = 2*bw
    def __getitem__(self,beta_selection):
        ids = self.d_to_coeff_ids
        if isinstance(beta_selection,tuple):
            beta_selection = beta_selection[0]
        beta_ids = np.arange(self.n_beta)[beta_selection]
        if isinstance(beta_ids,np.ndarray):
            d_ids = ids[:,0]+ids[:,1]*beta_ids[None,:]
            d_lnk = self.array[d_ids]*ids[:,2,None]
        else:
            d_ids = ids[:,0]+ids[:,1]*beta_ids
            d_lnk = self.array[d_ids]*ids[:,2]
        return d_lnk/self.normalization_factors
        
class WignerSmallD(np.ndarray):
    def __new__(cls,wigners,d_to_coeff_info,bw):
        wig = wigners.view(cls)
        wig.lnk = WigSmallCoefView(wigners,d_to_coeff_info,bw)        
        wig.b = WigSmallAngleView(wigners,d_to_coeff_info,bw)
        wig.coef_shape = int((4*(bw**3) - bw)/3)
        wig.betas = get_euler_angles(bw)[1]
        return wig


class WigAnglesView:
    def __init__(self,small_d,lnks,so3_grid):        
        self.small_d = small_d
        self.lnks  = lnks
        self.alphas = so3_grid[:,0,0,0]
        self.betas = self.small_d.betas
        self.gammas = so3_grid[0,0,:,2]
    def __getitem__(self,items):        
            alpha_id = items[0]
            beta_id = items[1]
            gamma_id = items[2]
            d_beta = self.small_d.b[beta_id]
            alpha_exp = np.exp(- 1.j*self.lnks[:,1]*self.alphas[alpha_id])
            gamma_exp = np.exp(- 1.j*self.lnks[:,2]*self.gammas[gamma_id])
            D_abg = alpha_exp*d_beta*gamma_exp 
            return D_abg
    
class WigCoefView:
    def __init__(self,small_d,so3_grid):
        self.small_d = small_d
        self.alphas = so3_grid[:,0,0,0]
        self.gammas = so3_grid[0,0,:,2]
    def __getitem__(self,items):
        d_lnk = self.small_d.lnk[items]
        alpha_exp = np.exp(-1.j*items[1]*self.alphas)
        gamma_exp = np.exp(-1.j*items[2]*self.gammas)
        return alpha_exp[:,None,None]*d_lnk[None,:,None]*gamma_exp[None,None,:]
    
class WignerD:
    def __init__(self,soft,small_d,so3_grid,bw):
        self.bw = bw
        self._soft = soft
        self.small_d = small_d
        self.coef_lnks = soft.coef_id_to_lnk(bw)        
        self.alpha = so3_grid[:,0,0,0]
        self.beta = so3_grid[0,:,0,1]
        self.gamma = so3_grid[0,0,:,2]
        self.abg = WigAnglesView(small_d,self.coef_lnks,so3_grid)
        self.lnk = WigCoefView(small_d,so3_grid)
        self.coef_shape = soft.totalCoeffs_so3(bw) 

        
class Soft(SoftInterface):
    def __init__(self,bw):
        self.bw = bw
        self._soft = _soft
        tmp = self.generate_data()
        self.grid = self.make_SO3_grid()
        self.alphas = self.grid[:,0,0,0]
        self.betas = self.grid[0,:,0,1]
        self.gammas = self.grid[0,0,:,2]        
        self._wigners = tmp[0]
        self._wigners000 = tmp[0][:2*bw]
        self._wigners_transposed = tmp[1] #small d matrices with m1,m2 swapped
        self.legendre_weights = tmp[2]
        self.n_coeffs = totalCoeffs_so3(bw)
        self.n_points = (2*bw)**3
        self._wigners_small = None
        self._wigners_big = None
        #self.small_d_to_coeff_info = self._soft.d_to_coeff_info(bw)
        #self.wigners_small = WignerSmallD(self._wigners,self.small_d_to_coeff_info,bw)

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
        coeffs = Forward_SO3_Naive_fft_pc(self.bw,data,self.legendre_weights,self._wigners,True)        
        return coeffs
    
    def inverse_cmplx(self,coeff):
        if (coeff.dtype != np.dtype(np.complex128)):
            log.warning('input dtype is {} but {} is required. Trying to change dtype.'.format(coeff.dtype,np.dtype(np.complex128)))
            coeff = coeff.astype(np.complex128)
        if (len(coeff) != self.n_coeffs) or (coeff.dtype != np.dtype(np.complex128)):
            raise AssertionError('input coeffs need to be of type complex128 and of length {}. Given data dtype = {} length = {}'.format(self.n_coeffs,coeff.dtype,len(coeff)))        
        data = Inverse_SO3_Naive_fft_pc(self.bw,coeff,self._wigners_transposed,True)        
        return data

    def make_SO3_grid(self):
        alpha,beta,gamma = get_euler_angles(self.bw)
        grid = GridFactory.construct_grid('uniform',[alpha,beta,gamma])[:]
        return grid

    def integrate_over_so3(self,signal):
        return integrate_over_so3(signal,self._wigners000,self.legendre_weights)
    def integrate_over_so3_normalized(self,signal):
        return integrate_over_so3_normalized(signal,self._wigners000,self.legendre_weights)
    def zeros_order_forward_so3(self,signal):
        return zeros_order_forward_so3(signal,self._wigners000,self.legendre_weights)

    @property
    def wigners_small(self):
        if not isinstance(self._wigners_small,WignerSmallD):
            small_d_to_coeff_info = self._soft.d_to_coeff_info(self.bw)
            self._wigners_small =  WignerSmallD(self._wigners,small_d_to_coeff_info,self.bw)
        return self._wigners_small
    
    @property
    def wigners(self):
        if not isinstance(self._wigners_big,WignerD):
            self._wigners_big = WignerD(self._soft,self.wigners_small,self.grid,self.bw)
        return self._wigners_big    

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
    def forward_coeff(self,coeff):
        ''' SOFT applied on spherical harmonic coefficient.
        Assumes that coeff = f_{l,m} is represented whose last axis is of size bw**2 with fl beeing at position l*(l+1)+m.'''
        soft_coeff = self.get_empty_coeff
         #    coeff_pos = 0
        for l in range(bw):
            wigNorm = np.sqrt(8.*np.pi/(2.*l+1.))
            f_l = coeff[l**2,(l+1)**2]
            m0_pos=l # position of the m = 0 component in arrays with components -l,...,l
            for m1 in range(-l,l+1):
                f_lm1 = f_l[m0_pos + m1] #should give f_{l-m1}           
                for m2 in range(-l,l+1):
                    # and save it in the so3 coefficient array */                
                    index = so3CoefLoc(m1,m2,l,bw)
                    coeff[index] = wigNorm * f_lm1
        return coeff
            
    
    
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
        mean_C = calc_mean_C_array(self.bw,f_coeff,g_coeff,r_split_lower,r_split_upper,ml_split_ids,self._wigners_transposed,True)
        return mean_C
    def calc_mean_C_weighted(self,f_coeff,g_coeff,r_split_ids,ml_split_ids):
        '''
        Same as calc_mean_C but multiplies each correlation array with its maximum before averaging to emphasize shells with good alignment.
        '''
        r_split_lower=r_split_ids[0]
        r_split_upper=r_split_ids[1]
        try:
            from pysofft.soft import calc_weighted_mean_C_array as method
        except ImportError:
            log.warning('Installed pysofft version does not support weighted mean correlation method, defaulting to unweighted routine.')
            method=calc_mean_C_array
        mean_C = method(self.bw,f_coeff,g_coeff,r_split_lower,r_split_upper,ml_split_ids,self._wigners_transposed,True)
        return mean_C

    
    #testing 
    def combine_coeffs(self,f_coeff,g_coeff,split_ids):
        return combine_harmonic_coeffs(self.bw,f_coeff,g_coeff,split_ids)

    
    def get_empty_coeff(self):
        return np.zeros(self.n_coeffs,dtype = complex)
    
    @property
    def lnks(self):
        return self._soft.coef_id_to_lnk(self.bw)

    @property
    def probability_weights(self):
        return get_so3_probabily_weights(self.legendre_weights)
