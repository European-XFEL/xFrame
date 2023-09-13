import time
import numpy as np
import scipy as sp
import logging
import traceback

from xframe import Multiprocessing
from xframe.library.physicsLibrary import ewald_sphere_theta_pi
from xframe.library.pythonLibrary import measureTime
from xframe.library.gridLibrary import GridFactory
from xframe.library.mathLibrary import eval_legendre,leg_trf,spherical_to_cartesian,cartesian_to_spherical,masked_mean
from xframe.library.mathLibrary import RadialIntegrator,solve_procrustes_problem,psd_back_substitution,back_substitution
from .harmonic_transforms import HarmonicTransform
import xframe.library.mathLibrary as mLib
from xframe.library.mathLibrary import nearest_positive_semidefinite_matrix
from scipy.signal import butter,sosfilt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.linalg import solve_triangular
from .harmonic_transforms import HarmonicTransform
log=logging.getLogger('root')

def ccd_associated_legendre_matrices(thetas,l_max,m_max):
    '''
    Calculates the upper triangular matrix $P^m_l(\cos(\theta_{q_1}))*P^m_l(\cos(\theta_{q_2}))*\frac{1}{\sqrt{2*l+1}}$ of schmidt semi-normalized associated spherical harmonics.
    '''
    q_matrices = np.zeros((len(thetas),m_max+1,l_max+1))
    values,ls,ms= mLib.gsl.legendre_sphPlm_array(l_max,m_max,np.cos(thetas),return_orders = True)    
    q_matrices[:,ms,ls] = values.T
    orders = np.arange(l_max+1)
    qq_matrices = q_matrices[None,:]*q_matrices[:,None]/(2*orders+1)[None,None,None,:]#*1/(2*np.pi) # might be missing  here
    #causes failing reconstructions !!!! if included why?
    return qq_matrices

def ccd_associated_legendre_matrices_q1q2(thetas1,thetas2,l_max,m_max):
    '''
    Calculates the upper triangular matrix $P^m_l(\cos(\theta_{q_1}))*P^m_l(\cos(\theta_{q_2}))*\frac{1}{\sqrt{2*l+1}}$ of schmidt semi-normalized associated spherical harmonics.
    '''
    q1_matrices = np.zeros((len(thetas1),m_max+1,l_max+1))
    q2_matrices = q1_matrices.copy()
    values1,ls1,ms1 = mLib.gsl.legendre_sphPlm_array(l_max,m_max,np.cos(thetas1),return_orders = True)
    values2,ls2,ms2 = mLib.gsl.legendre_sphPlm_array(l_max,m_max,np.cos(thetas2),return_orders = True)    
    q1_matrices[:,ms1,ls1] = values1.T
    q2_matrices[:,ms2,ls2] = values2.T
    orders = np.arange(l_max+1)
    q1q2_matrices = q1_matrices*q2_matrices/(2*orders+1)[None,None,:]#*1/(2*np.pi) # might be missing  here
    return q1q2_matrices
def ccd_associated_legendre_matrices_single_m(thetas,l_max,m):
    '''
    Calculates the elements of the upper triangular matrix $P^m_l(\cos(\theta_{q_1}))*P^m_l(\cos(\theta_{q_2}))*\frac{1}{\sqrt{2*l+1}}$ of schmidt semi-normalized associated spherical harmonics at a specific order $m$.
    '''
    q_matrices = np.zeros((len(thetas),l_max+1))
    values,ls,ms= mLib.gsl.legendre_sphPlm_array_single_m(l_max,m,np.cos(thetas),return_orders = True)    
    q_matrices[:,ls] = values.T
    orders = np.arange(l_max+1)
    qq_matrices = q_matrices[None,:]*q_matrices[:,None]/(2*orders+1)[None,None,:]
    #q_matrices[:,ms[1:],ls[1:]]# *= 1/np.sqrt(ls[1:]*(ls[1:]+1))[None,:]
    return qq_matrices

def ccd_associated_legendre_matrices_single_l(thetas,l_max,l):
    '''
    Calculates the elements of the upper triangular matrix $P^m_l(\cos(\theta_{q_1}))*P^m_l(\cos(\theta_{q_2}))*\frac{1}{\sqrt{2*l+1}}$ of schmidt semi-normalized associated spherical harmonics at a specific order $m$.
    '''
    q_matrices = np.zeros((len(thetas),l_max+1))
    values,ls,ms= mLib.gsl.legendre_sphPlm_array_single_l(l,l_max,np.cos(thetas),return_orders = True)
    #print(f"len thetas = {len(thetas)} shape values = {values.shape} q_matrix shape = { q_matrices[:,ms].shape}, l = {l} lmax = {l_max}")
    if l==0:
        q_matrices[:,0] = values
        
    else:
        q_matrices[:,ms] = values.T
    qq_matrices = q_matrices[None,:,:]*q_matrices[:,None,:]/((2*l+1))#*(2*np.pi))
    #q_matrices[:,ms[1:],ls[1:]]# *= 1/np.sqrt(ls[1:]*(ls[1:]+1))[None,:]
    return qq_matrices
    
def ccd_legendre_matrices(q_ids, qq_ids,phis,thetas,orders,**kwargs):
    '''
    Calculates the $F_l(q,qq,phi)$ coefficients in $CC=\sum F_l B_l$, where $CC$ is the cross-correlation and $B_l$ are the degree 2 invariants.
    The inputs q_ids and qq_ids have to be of tha same length since they are interpreted as pairs of ids,i.e. as zip(q_ids,qq_ids)
    :param q_ids: Id's of q values for which to calculate F_l matrices 
    :type ndarray: shape = (N). N can be of arbitrary length < len(thetas)**2
    :param qq_ids: Id's of qq values for which to calculate F_l matrices 
    :type ndarray: shape = (N)  N can be of arbitrary length < len(thetas)**2
    :param phis: phi values
    :type ndarray: shape = (n_phis)
    :param thetas: ewald's sphere thetas for given q values
    :type ndarray: shape = (n_phis)
    :param orders: the l's for which to compute F_l
    :type ndarray: shape = (n_orders)
    :return F_l:
    :rtype ndarray: shape: (N,n_phis,n_orders)
    '''
    q_ids = q_ids.astype(int)
    qq_ids = qq_ids.astype(int)
    leg_args=np.cos(thetas)[q_ids,None]*np.cos(thetas)[qq_ids,None]+np.sin(thetas)[q_ids,None]*np.sin(thetas)[qq_ids,None]*np.cos(phis)[None,:] #Q,phi
    Legendre_matrices=np.moveaxis(1/(4*np.pi)*eval_legendre(orders[:,None,None],leg_args),0,-1) #Q,phi,l
    return Legendre_matrices


def pixel_arc_cc_mask(data_grid,datadict):
    '''
    Calculates mask of cross-correlation data which filters points cc values calculated for points on the ewald sphere which are close to eachother.
    Distance is measured via the arc length beetween the two points on the sphere.
    It is possible to additionally mask the area around phi = pi, i.e. Points that are close if one of their phi coordinates is inverted (phi -> phi - pi).     
    :parameter data_grid: Dictionary containing grid values for the 3 spherical axes "qs","thetas","phis"
    :parameter datadict: Dictionary containing: "xray_wavelength":double in Angstrom, 'pixel_size':double realspace size of features to mask, 'mask_at_pi':bool wether or not to also mask around phi=pi 
    '''
    qs = data_grid['qs']
    thetas = data_grid['thetas']
    phis = data_grid['phis']
    wavelength = datadict['xray_wavelength']
    pixel_size = datadict['pixel_size']
    mask_at_pi = datadict['mask_at_pi']
    
    grid = GridFactory.construct_grid('uniform_dependent',[np.concatenate((qs[:,None],thetas[:,None]),axis=1),phis]).array
    cart_grid  = spherical_to_cartesian(grid)
    ewald_r = 2*np.pi/wavelength
    ewald_grid = cartesian_to_spherical(cart_grid-np.array([0,0,ewald_r]))

    ew_cos_theta = np.cos(ewald_grid[:,0,1])
    ew_sin_theta = np.sin(ewald_grid[:,0,1])
    ew_cos_phi =  np.cos(ewald_grid[1,:,2])
    ew_arc_cos = ew_cos_theta[:,None,None]*ew_cos_theta[None,:,None]+ew_sin_theta[:,None,None]*ew_sin_theta[None,:,None]*ew_cos_phi[None,None,:]
    ark_length = np.abs(ewald_r*np.arccos(ew_arc_cos))
    
    r_pixel_size = 2*np.pi/pixel_size
    if mask_at_pi:
        ew_cos_phi_pi =  np.cos(ewald_grid[1,:,2]-np.pi)
        ew_arc_cos_pi = ew_cos_theta[:,None,None]*ew_cos_theta[None,:,None]+ew_sin_theta[:,None,None]*ew_sin_theta[None,:,None]*ew_cos_phi_pi[None,None,:]
        ark_length_pi = np.abs(ewald_r*np.arccos(ew_arc_cos_pi))
        
        mask = (ark_length > r_pixel_size) & (ark_length_pi > r_pixel_size)
    else:
        mask = (ark_length > r_pixel_size)
    return mask
def true_cc_mask(data_grid):
    n_qs =  len(data_grid['qs'])
    n_phis =  len(data_grid['phis'])
    return np.ones((n_qs,n_qs,n_phis),dtype = bool)
def pixel_custom_cc_mask(data_grid,datadict):
    '''
    Calculates mask of cross-correlation data which filters points cc values calculated for points on the ewald sphere which are close to eachother.
    Distance is measured via the arc length beetween the two points on the sphere.
    It is possible to additionally mask the area around phi = pi, i.e. Points that are close if one of their phi coordinates is inverted (phi -> phi - pi).     
    :parameter data_grid: Dictionary containing grid values for the 3 spherical axes "qs","thetas","phis"
    :parameter datadict: Dictionary containing: "xray_wavelength":double in Angstrom, 'pixel_size':double realspace size of features to mask, 'mask_at_pi':bool wether or not to also mask around phi=pi 
    '''
    qs = data_grid['qs']
    thetas = data_grid['thetas']
    phis = data_grid['phis']
    n = int(len(phis)*datadict['n_masked_pixels_phi'])
    nq = int(len(qs)*datadict['n_masked_pixels_q'])
    mask_at_pi = datadict['mask_at_pi']
    n_phis = len(phis)
    n_qs = len(qs)
    pi_index = int(n_phis/2)
    #mask_ids = list(range(n))
    if mask_at_pi:
        mask_ids = list(range(n))+list(range(pi_index-(n-1),pi_index+(n-1),1))+list(range(n_phis-n,n_phis,1))
    else:
        mask_ids = list(range(n))+list(range(n_phis-n,n_phis,1))
    #mask_ids = list(range(pi_index-(n-1),pi_index+(n-1),1))
    cc_mask = np.full((n_qs,n_qs,n_phis),True)
    cc_mask[...,mask_ids]=False
    q_mask= np.full(n_qs,True)
    qq_mask = (q_mask[:,None] | q_mask[None,:])
    qq_mask_2 = np.abs(np.arange(len(q_mask))[:,None] - np.arange(len(q_mask))[None,:])>nq
    cc_mask[qq_mask_2]=True
    cc_mask[~qq_mask]=False
    
    return cc_mask
def pixel_flat_cc_mask(data_grid,datadict):
    '''
    Calculates mask of cross-correlation data which filters values around phi = 0,np.pi for which q_1\approx q_2.
    '''
    qs = data_grid['qs']
    thetas = data_grid['thetas']
    phis = data_grid['phis']
    wavelength = datadict['xray_wavelength']
    pixel_size = datadict['pixel_size']
    mask_at_pi = datadict['mask_at_pi']
    
    grid = GridFactory.construct_grid('uniform_dependent',[np.concatenate((qs[:,None],thetas[:,None]),axis=1),phis]).array
    r_pixel_size = 2*np.pi/pixel_size
    arc_pixel_fraction = (2*np.pi*qs) / r_pixel_size
    phi_min = 2*np.pi / arc_pixel_fraction
    phi_mask = (phis[None,:]>phi_min[:,None]) & (phis[None,:]< (2*np.pi - phi_min[:,None]))
    if mask_at_pi:
        phi_mask &= (phis[None,:]>np.pi+phi_min[:,None]) | (phis[None,:]<np.pi - phi_min[:,None])
    phi_mask = phi_mask[None,:,:] & phi_mask[:,None,:]
    
    radial_mask = np.abs(qs[None,:]-qs[:,None]) > r_pixel_size
    
    mask = radial_mask[:,:,None] | phi_mask
    return mask

def donatelli_cc_mask(data_grid,datadict):
    '''
    Calculates mask of cross-correlation data which filters values around phi = 0,np.pi for which q_1\approx q_2.
    Treshold standard value = 0.01 (filters roughly 20% of data), negative thresholds gives a true mask.
    Formula from Donatelli PNAS 2018 Supplements.
    :parameter data_grid: Dictionary containing grid values for the 3 spherical axes "qs","thetas","phis"
    :parameter datadict: Dictionary containing "threshold":int a free parameter higher values => more maskin. 
    '''
    qs = data_grid['qs']
    thetas = data_grid['thetas']
    phis = data_grid['phis']
    threshold = datadict.get('threshold',0.01)
    qs_squared = qs**2
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    a = qs_squared[:,None,None] + qs_squared[None,:,None]
    b = 2*qs[:,None,None]*qs[None,:,None]*(cos_theta[:,None,None]*cos_theta[None,:,None] + sin_theta[:,None,None]*sin_theta[None,:,None]*np.cos(phis)[None,None,:])
    metric_1= (a+b < threshold)
    metric_2= (a-b < threshold)
    cc_mask = ~(metric_1 | metric_2)
    return cc_mask
def custom_cc_mask(data_grid,datadict):
    return datadict['mask']

def cross_correlation_mask(data_grid,datadict):
    known_mask_types={'direct':custom_cc_mask,'none':true_cc_mask,'pixel_arc':pixel_arc_cc_mask,'pixel_flat':pixel_flat_cc_mask,'pixel_custom':pixel_custom_cc_mask}
    given_type = datadict['cc_mask']['type']
    generator=known_mask_types.get(given_type,False)
    if isinstance(generator,bool):
        e = AssertionError('Given Cross-Correlation mask type "{}" not known. Known types are {}'.format(given_type,known_mask_types.keys()))
        raise e
    elif given_type == 'none':
        mask = generator(data_grid)
    else:
        mask = generator(data_grid,{**datadict['cc_mask'][given_type],**datadict})
    return mask


def modify_cross_correlation(cc,cc_mask,phis,max_order,average_intensity=False,enforce_max_order=False, low_pass_order_in_q = False, enforce_zero_odd_harmonics = False,q1q2_symmetric=False, pi_periodicity = False, interpolate_masked=False,apply_binned_mean = False,subtract_average_intensity= False):
    '''
    Imposes several constraints on the cross_correlation:
    1) enforce_max_ordes: $C_n = 0$ for $n >$ max order, since by $C_n = sum_l=|m|^\infty B_l(q_1,q_2) \overline{P}^{|m|}_l(\theta(q_1))  \overline{P}^{|m|}_l(\theta(q_2))$ they dont contribute to B_l with l<=max_order.
    2) low_pass_order_in_q: butterworth low pass filter in q1 and q2 of cc
    3) enforce_zero_odd_harmonics: cc is pi symmetric in phi => odd circular harmonic coefficients should be zero
    4) q1q2_symmetric: enforces cc(q1,q2,\delta) = cc(q2,q1,-\delta).
    5) pi_periodicity: same as 3) but enforced directy on the CC values if number of phis is even.
    '''
    #log.info('enforce max order {}, low pass = {}, zero odd ={}, q1q2 {}, qhipi {}, interpolate {}, bunned_mean {}'.format(enforce_max_order, low_pass_order_in_q, enforce_zero_odd_harmonics, q1q2_transpose , phi_pi, interpolate_masked,apply_binned_mean))
    if subtract_average_intensity and isinstance(average_intensity,np.ndarray):
        I= average_intensity
        cc -= I[:,None,None]*I[None,:,None]
    if not isinstance(low_pass_order_in_q,bool):
        log.info('low passorder = {}'.format(low_pass_order_in_q))
        sos = butter(1, low_pass_order_in_q, 'lp', fs=len(cc), output='sos')
        cc = sosfilt(sos,cc,axis = 0)
        cc = sosfilt(sos,cc,axis = 1)
    if enforce_max_order or enforce_zero_odd_harmonics:
        ht = HarmonicTransform('real',{'dimensions':2,'n_angular_points':cc.shape[-1],'max_order':False})
        log.info(cc.shape)
        cc_coeff = ht.forward(cc)
        log.info(cc_coeff.shape)
        if enforce_max_order:
            cc_coeff[...,max_order+1:]=0
        if enforce_zero_odd_harmonics:
            cc_coeff[...,1::2]=0
        cc = ht.inverse(cc_coeff)
        log.info(cc.shape)
    if pi_periodicity:
        assert cc.shape[-1]%2 == 0, 'for odd number of phi symmetry enforcing is not possible since phi+pi is not an existing grid point.'
        assert cc.shape[-1] == len(phis), 'Cross correlation has {} angular datapoints but only {} angle values are given.'.format(cc.shape[-1],len(phis))
        bad_angles = (phis < np.pi/2) | (phis >= 3*np.pi/2)
        cc[...,bad_angles] = 0
        cc += np.roll(cc,len(phis)//2,axis=-1)
        cc_mask = cc_mask | np.roll(cc_mask,len(phis)//2,axis=-1)
    if q1q2_symmetric:
        #swap angles
        log.info('Enforce cross correlation symmetry q1q2\delta = q2q1-\delta')
        cc_angle_swaped = cc.copy()
        cc_angle_swaped[...,1:] = cc[...,1:][...,::-1]
        cc_mask_angle_swaped = cc_mask.copy()
        cc_mask_angle_swaped[...,1:] = cc_mask[...,1:][...,::-1]
        cc,mask = masked_mean([np.swapaxes(cc_angle_swaped,0,1),cc],[np.swapaxes(cc_mask_angle_swaped,0,1),cc_mask])
        cc_mask = mask.astype(bool)
    #log.info('binned_mean: {}'.format(apply_binned_mean))
    if apply_binned_mean:
        log.info('calculating binned mean')
        cc,cc_mask,phis = binned_mean(cc_mask,cc,max_order,phis)
    if interpolate_masked:
        log.info('interpolating masked cc areas')
        cc =  interpolate(cc,cc_mask,phis)
        cc_mask[...]=True
    
    return cc,cc_mask,phis
        
def ccd_remove_0_order(cc):
    ht = HarmonicTransform('real',{"dimensions":2,'n_angular_points':cc.shape[-1]})
    ccf = ht.forward(cc)
    #log.info('shape of ccf = {}'.format(ccf.shape))
    ccf[:,:,0]=0
    new_cc = ht.inverse(ccf)
    return new_cc,locals()
def cross_correlation_low_pass(cc,max_order):
    ht = HarmonicTransform('real',{"dimensions":2,'n_angular_points':cc.shape[-1]})
    ccf = ht.forward(cc)
    #log.info('shape of ccf = {}'.format(ccf.shape))
    
    ccf[:,:,max_order+1:-max_order-1]=0
    new_cc = ht.inverse(ccf)
    return new_cc,locals()
        

def binned_mean(cc_mask,cc,max_order,phis):
    # create bin ids
    step_size = np.pi/max_order
    n_bins = 2*max_order
    ids = (phis+step_size/2)//step_size
    n_roll= np.sum(ids == n_bins)
    ids[ids == n_bins] = 0

    # roll such that bin ids monotonically increase
    ccr = np.roll(np.array(cc),n_roll,axis = -1)
    cc_mask_r = np.roll(cc_mask,n_roll,axis = -1)
    idr = np.roll(ids,n_roll)

    # masked add
    ccr[~cc_mask_r] = 0.0
    split_ids = np.where(np.roll(idr,1)!=idr)[0]
    new_cc = np.add.reduceat(ccr,split_ids,axis = -1)

    # divide by number of unmasked values per bin
    n_unmasked_values = np.add.reduceat(cc_mask_r,split_ids,axis = -1)
    new_cc_mask = (n_unmasked_values!=0)
    new_cc[new_cc_mask]/=n_unmasked_values[new_cc_mask]
    new_phis = np.arange(n_bins)*2*np.pi/n_bins
    
    return new_cc,new_cc_mask,new_phis


def interpolate(cc,mask,phis):
    'interpolates masked cross correlation using scipy interp1d.'
    shape = cc.shape
    log.info('interpolating')
    n_phi = len(phis)
    points = np.arange(n_phi)
    cci = np.array(cc)
    cci = cci.reshape(-1,n_phi)
    mask = mask.reshape(-1,n_phi)
    for d,m in zip(cci,mask):
        #log.info('data shape {} mask shape {}'.format(d.shape,m.shape))
        unmasked_d = d[m]
        if len(unmasked_d)!=0:
            d[~m] = interp1d(phis[m],d[m])(phis[~m])
    cci = cci.reshape(shape)
    mask = mask.reshape(shape)
    return cci
    

########### deg2 invariants and cross correlations  ##################
def get_q_slice(q_mask):
    q_lims = [0,len(q_mask)]
    if not q_mask[0]:
        q_lims[0]=np.argmax(q_mask)
    if not q_mask[-1]:
        neg_q_max = np.argmax(q_mask[::-1])+1
        q_lims[1] = len(q_lims)-neg_q_max
    return slice(*q_lims)
    
def get_q1q2_slices(cc_mask):
    qq_maks = cc_mask.sum(axis = -1)
    q1_mask = qq_mask.sum(axis = -1)
    q2_mask = qq_mask.sum(axis = 0)

    q1_lim = get_q_slice(q1_mask)
    q2_lim = get_q_slice(q1_mask)
    return [q1_lim,q2_lim]


def cross_correlation_to_deg2_invariant(cc,dim,**metadata):
    '''
    Routine that distributes requests for deg2_invariants to the respective routines in the 2 and 3 dimensional case.
    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param dim: Dimension of MTIP algorithm for which deg2 invariants will be extracted (2 or 3)
    :type int: 2 or 3    
    '''
    #log.info(metadata.keys())
    cc_mask = cross_correlation_mask(metadata['data_grid'],metadata)
    orders =  metadata['orders']
    max_order = orders.max()
    n_angular_points=cc.shape[-1]
    try:        
        assert n_angular_points>=max_order*2,'given maximal order {} can not be resolved using {} angular points in the cross rorrelation reducing extraction to a maximal order of {}'.format(max_order,n_angular_points,(n_angular_points-1)/2)
    except AssertionError as e:
        max_order = (n_angular_points-1)/2
        orders = orders[orders>=max_order]
    data_grid = metadata['data_grid']
    average_intensity = metadata.get('average_intensity',False)
    cc,cc_mask,phis = modify_cross_correlation(cc,cc_mask,data_grid['phis'],max_order,average_intensity = average_intensity,**metadata['modify_cc'])
    data_grid['phis'] = phis
    order_mask = np.full(len(orders),True)
    if metadata['assume_zero_odd_orders']:
        log.info(f'assume zero odd orders = {metadata["assume_zero_odd_orders"]}')
        order_mask = ~np.array(orders%2,dtype = bool)

    
    if dim == 2:
        b_coeff = np.zeros((len(order_mask),)+cc_mask.shape[:2],dtype = complex)
        #b_coeff,q_mask = ccd_to_deg2_invariant_2d(cc,cc_mask,data_grid,max_order)
        b_coeff[order_mask],qq_mask = ccd_to_deg2_invariant_2d(cc,data_grid,orders[order_mask],cc_mask)
        #log.info('max eig 4 = {}'.format(np.linalg.eigh(np.fft.rfft(cc,axis=-1)[...,0])[0].max()))
        #log.info('max eig 5 bl = {}'.format(np.linalg.eigh(b_coeff[0])[0].max()))

    elif dim == 3:
        #b_coeff = np.zeros((len(order_mask),)+cc_mask.shape[:2],dtype = complex)
        xray_wavelength = metadata['xray_wavelength']
        mode_specified = metadata.get('mode',False)
        #log.info('start extraction in mode = {}'.format(mode_specified))
        if isinstance(mode_specified,bool):
            b_coeff,qq_mask = ccd_to_deg2_invariant_3d(cc,xray_wavelength,data_grid,orders[order_mask],cc_mask)            
        else:
            b_coeff = np.zeros(cc_mask.shape[:2]+(max_order+1,),dtype = complex)
            b_coeff[...,order_mask],qq_mask = ccd_to_deg2_invariant_3d(cc,xray_wavelength,data_grid,orders[order_mask],cc_mask,mode=mode_specified)
            b_coeff = np.moveaxis(b_coeff,-1,0)
    else:                
        pass    
    return b_coeff,qq_mask

def ccd_to_deg2_invariant_3d(cc,xray_wavelength,data_grid,orders,cc_mask,mode='back_substitution'):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta).
    Currently implemented method uses pseudo inverses to invert above linear system
    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    
    extraction_modes = {'lstsq':ccd_to_deg2_invariant_3d_least_squares,'legendre':ccd_to_deg2_invariant_3d_legendre,'back_substitution':ccd_to_deg2_invariant_3d_back_substitution,'back_substitution_psd':ccd_to_deg2_invariant_3d_back_substitution_psd,'back_substitution_qqsym':ccd_to_deg2_invariant_3d_back_substitution_qqsym,'back_substitution_memory_hungry':ccd_to_deg2_invariant_3d_back_substitution_memory_hungry}

    log.info(f'orders = {orders}')
    b_coeff = np.zeros(cc_mask.shape[:2]+(orders.max()+1,),dtype = complex)
    
    worker = extraction_modes.get(mode,False)
    assert not isinstance(worker,bool),'Given B_l extraction mode "{}" is unknown. Known modes are {}"'.format(mode,list(extraction_modes.keys()))    
    bl,qq_mask = worker(cc,xray_wavelength,data_grid,orders,cc_mask)
    #log.info(f'bl shape = {bl.shape}')
    log.info(f'bl shape = {bl.shape}')
    return bl,qq_mask

def ccd_to_deg2_invariant_3d_least_squares(cc,xray_wavelength,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta) using
    $CC=\sum F_l B_l$.
    Currently implemented method uses pseudo inverses to invert above linear system
    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    
    # Get Fl arguments 
    qs = data_grid['qs']
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)
    phis=data_grid['phis']


    # calc deg 2 invcariants
    q_ids = np.arange(len(qs))
    b_coeff = Multiprocessing.comm_module.request_mp_evaluation(bl_3d_least_squares_worker,
                                                                       input_arrays=[q_ids,q_ids],
                                                                       const_inputs=[cc,phis,thetas,orders,cc_mask],
                                                                       call_with_multiple_arguments=True)
    qq_mask = cc_mask.sum(axis = -1).astype(bool)
    #b_coeff = np.moveaxis(b_coeff,-1,0)
    return b_coeff,qq_mask
    
def bl_3d_least_squares_worker(q_ids,qq_ids,cc,phis,thetas,orders,cc_mask,**kwargs):
    '''
    Uses pseudoinverse of the legendre matrices (numpy internal uses a singular value decomposition)
    '''
    b_shape  = (len(q_ids),len(orders))
    b_coeff = np.zeros(b_shape,dtype=complex)        
    #even_order_mask = ~np.array(orders%2,dtype = bool)
    #even_orders = orders[even_order_mask]
    #log.info(orders)
    cc_mask_part = cc_mask[q_ids,qq_ids]
    if (cc_mask_part).all():
        #legendre_matrices = ccd_legendre_matrices(q_ids,qq_ids,phis,thetas,even_orders)
        legendre_matrices = ccd_legendre_matrices(q_ids,qq_ids,phis,thetas,orders)
        #l_inv = np.linalg.pinv(legendre_matrices)
        cc_part = cc[q_ids,qq_ids,:]
        b_coeff = np.array(tuple(np.linalg.lstsq(leg_matrix,cc_subpart,rcond=None)[0] for leg_matrix,cc_subpart in zip(legendre_matrices,cc_part)))
        #log.info('legendre_shape = {} inv_leg_shape = {} cc_shape = {}'.format(legendre_matrices.shape,l_inv.shape,cc_part.shape))
        #b_coeff = np.sum(l_inv*cc_part,axis = -1)
    else:
        #legendre_matrices = ccd_legendre_matrices(q_ids,qq_ids,phis,thetas,even_orders)
        legendre_matrices = ccd_legendre_matrices(q_ids,qq_ids,phis,thetas,orders)
        phi_masks = cc_mask_part
        for n,q_qq_id in enumerate(zip(q_ids,qq_ids)):
            q_id,qq_id = q_qq_id
            phi_mask = phi_masks[n]
            if phi_mask.any():
                legendre_matrix = legendre_matrices[n][phi_mask]
                cc_part = cc[q_id,qq_id,phi_mask]
                result = np.linalg.lstsq(legendre_matrix,cc_part,rcond=None)[0]
                b_coeff[n] = result  
            else:
                b_coeff[n] = 0                
    return b_coeff
            
def ccd_to_deg2_invariant_3d_back_substitution_memory_hungry(cc,xray_wavelength,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta).
    Uses 
    $ Cn = \sum_{l=|n|}^\infty B_l(q_1,q_2) \overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    to extract B_l from given fourier coefficients of the averaged cross_correlation Cn by inverting the upper triangular matrix:
    $\overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    where \overline{P}^{|n|}_l are the associated Legendre Polynomials that are used in the spherical harmonic coefficients.
    
    Only works for q1,q2 pairs for which the crosscorrelation is not masked at any angular value.

    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    # Get Pl arguments 
    qs = data_grid['qs']
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)
    l_max = orders.max()

    if not cc_mask.all():
        log.info(f'nmasked = {cc_mask.sum()} of {np.prod(cc_mask.shape)}')
        log.info('interpolating masked cross_correlation areas.')
        cc,cc_mask,phis = modify_cross_correlation(cc,cc_mask,data_grid['phis'],orders.max(),interpolate_masked=True)
    qq_mask=cc_mask.prod(axis = -1).astype(bool)

    # Check if there are any odd orders to compute
    odd_orders_present = (orders%2).any()
    if odd_orders_present:
        l_stride = 1
    else:
        l_stride = 2
        orders = orders//2
    log.info(f'l_stride = {l_stride}')
    # Calculate triangular PP matrices and harmoncic coefficents CCn of the cross-correlation
    #log.info('start')
    qq_matrices = ccd_associated_legendre_matrices(thetas,l_max,l_max)[:,:,::l_stride,::l_stride]
    ccn = np.zeros(qq_matrices.shape[:3],dtype = complex)
    ccn[qq_mask,:] = mLib.circularHarmonicTransform_real_forward(cc[qq_mask,:])[...,:l_max+1:l_stride]
    b_coeff = back_substitution(ccn,qq_matrices)
    #log.info('stop')

    #b_coeff = np.zeros(ccn.shape,dtype = complex)
    #q_ids = np.arange(len(qs))
    #b_coeff =  Multiprocessing.comm_module.request_mp_evaluation(bl_3d_back_substitution_worker,
    #                                                                   input_arrays=[q_ids,q_ids],
    #                                                                   const_inputs=[cc,thetas,qq_mask,l_max,l_stride],
    #                                                                   call_with_multiple_arguments=True)
    #log.info('stop')
    log.info(f"b coeff shape = {b_coeff.shape}")
    return b_coeff,qq_mask

def ccd_to_deg2_invariant_3d_back_substitution(cc,xray_wavelength,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta).
    Uses 
    $ Cn = \sum_{l=|n|}^\infty B_l(q_1,q_2) \overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    to extract B_l from given fourier coefficients of the averaged cross_correlation Cn by inverting the upper triangular matrix:
    $\overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    where \overline{P}^{|n|}_l are the associated Legendre Polynomials that are used in the spherical harmonic coefficients.
    
    Only works for q1,q2 pairs for which the crosscorrelation is not masked at any angular value.

    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    # Get Pl arguments 
    qs = data_grid['qs']
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)
    l_max = orders.max()

    if not cc_mask.all():
        log.info(f'nmasked = {cc_mask.sum()} of {np.prod(cc_mask.shape)}')
        log.info('interpolating masked cross_correlation areas.')
        cc,cc_mask,phis = modify_cross_correlation(cc,cc_mask,data_grid['phis'],orders.max(),interpolate_masked=True)
    qq_mask=cc_mask.prod(axis = -1).astype(bool)

    # Check if there are any odd orders to compute
    odd_orders_present = (orders%2).any()
    if odd_orders_present:
        l_stride = 1
    else:
        l_stride = 2
        #orders = orders//2
    log.info(f'l_stride = {l_stride}')
    # Calculate triangular PP matrices and harmoncic coefficents CCn of the cross-correlation
    #log.info('start')
    #qq_matrices = ccd_associated_legendre_matrices(thetas,l_max,l_max)[:,:,::l_stride,::l_stride]
    ccn = np.zeros((len(qs),len(qs),len(orders)),dtype = complex)
    ccn[qq_mask,:] = mLib.circularHarmonicTransform_real_forward(cc[qq_mask,:])[...,:l_max+1:l_stride]

    #lazy back substitution
    bl = np.zeros(ccn.shape,dtype=complex)
    # reversed orders shoud be decreasing L,L-1,... or L,L-2,...
    reversed_orders=orders[::-1]
    for l in reversed_orders:
        last_triangular_matrix_column = ccd_associated_legendre_matrices_single_l(thetas,l,l)[...,::l_stride]
        bl[...,l//l_stride]= ccn[...,-1]/last_triangular_matrix_column[...,-1]
        ccn = ccn[...,:-1]-bl[...,l//l_stride,None]*last_triangular_matrix_column[...,:-1]
    b_coeff = bl
    #b_coeff = back_substitution(ccn,qq_matrices)
    #log.info('stop')

    #b_coeff = np.zeros(ccn.shape,dtype = complex)
    #q_ids = np.arange(len(qs))
    #b_coeff =  Multiprocessing.comm_module.request_mp_evaluation(bl_3d_back_substitution_worker,
    #                                                                   input_arrays=[q_ids,q_ids],
    #                                                                   const_inputs=[cc,thetas,qq_mask,l_max,l_stride],
    #                                                                   call_with_multiple_arguments=True)
    #log.info('stop')
    log.info(f"b coeff shape = {b_coeff.shape}")
    return b_coeff,qq_mask
def ccd_to_deg2_invariant_3d_back_substitution_qqsym(cc,xray_wavelength,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta).
    Uses 
    $ Cn = \sum_{l=|n|}^\infty B_l(q_1,q_2) \overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    to extract B_l from given fourier coefficients of the averaged cross_correlation Cn by inverting the upper triangular matrix:
    $\overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    where \overline{P}^{|n|}_l are the associated Legendre Polynomials that are used in the spherical harmonic coefficients.
    
    Only works for q1,q2 pairs for which the crosscorrelation is not masked at any angular value.

    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    # Get Pl arguments 
    qs = data_grid['qs']
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)
    l_max = orders.max()

    qq_mask=cc_mask.prod(axis = -1).astype(bool)

    # Check if there are any odd orders to compute
    odd_orders_present = (orders%2).any()
    if odd_orders_present:
        l_stride = 1
    else:
        l_stride = 2
        orders = orders//2
    log.info(f'l_stride = {l_stride}')
    # Calculate triangular PP matrices and harmoncic coefficents CCn of the cross-correlation
    #log.info('start')
    qq_matrices = ccd_associated_legendre_matrices(thetas,l_max,l_max)[:,:,::l_stride,::l_stride]
    qq_matrices = (qq_matrices+np.swapaxes(qq_matrices,0,1))/2
    ccn = np.zeros(qq_matrices.shape[:3],dtype = complex)
    temp_ccn = mLib.circularHarmonicTransform_real_forward(cc)[...,:l_max+1:l_stride]
    temp_ccn = (temp_ccn+np.swapaxes(temp_ccn,0,1).conj())/2
    ccn[qq_mask,:] = temp_ccn[qq_mask,:]
    b_coeff = back_substitution(ccn,qq_matrices)
    #log.info('stop')
    log.info(f"b coeff shape = {b_coeff.shape}")
    return b_coeff,qq_mask

def bl_3d_back_substitution_worker(q1_ids,q2_ids,cc,thetas,qq_mask,l_max,stride,**kwargs):
    n_parts = len(q1_ids)
    thetas1 = thetas[q1_ids]
    thetas2 = thetas[q2_ids]
    qq_matrices = ccd_associated_legendre_matrices_q1q2(thetas1,thetas2,l_max,l_max)[:,::stride,::stride]
    ccn =  mLib.circularHarmonicTransform_real_forward(cc[q1_ids,q2_ids])[...,:l_max+1:stride]
    bl_parts = np.zeros((n_parts,ccn.shape[-1]),dtype = complex)
    q1q2_masks = qq_mask[q1_ids,q2_ids]
    for bl,ppm,ccn_val,mask in zip(bl_parts,qq_matrices,ccn,q1q2_masks):
        if mask: 
            bl[:] = solve_triangular(ppm,ccn_val)
    return bl_parts


def ccd_to_deg2_invariant_3d_back_substitution_psd(cc,xray_wavelength,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_l=\sum_l I^l_m*I^l_m^*$ from Cross-Correlation data CC(q,qq,delta).
    Uses 
    $ Cn = \sum_{l=|n|}^\infty B_l(q_1,q_2) \overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    to extract B_l from given fourier coefficients of the averaged cross_correlation Cn by inverting the upper triangular matrix:
    $\overline{P}^{|n|}_l(q_1) \overline{P}^{|n|}_l(q_2) $
    where \overline{P}^{|n|}_l are the associated Legendre Polynomials that are used in the spherical harmonic coefficients.
    
    Only works for q1,q2 pairs for which the crosscorrelation is not masked at any angular value.

    :param cc: cross-correlation data
    :type complex ndarray: shape $= (n_q(q),n_q(qq),n_\Delta)$
    :param xray_wavelength: wavelength at which the cross-correlations where calculated.
    :type float: unit eV
    :param data_grid: qs and $\Delta$'s at whih cc is calculated. (assumes uniform grid, i.e. for each q value there are cc coefficients for all $\Delta$'s)
    :type lyist((ndarray,ndarray)): shape $(n_q,n_\Delta)$
    :param orders: $l$'s for which to calculate the invariants $B_l$
    :type int ndarray: shape $= (n_orders)$
    :return: $B_l$ coeff.
    :rtype ndarray: complex, shape $= (n_orders,n_q,n_q)$
    '''
    # Get Pl arguments 
    qs = data_grid['qs']
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)
    l_max = orders.max()
    #lop.info(f'initial mask shape = {cc_mask.shape}')
    qq_mask=cc_mask.prod(axis = -1).astype(bool)
    #lop.info(f'final mask shape = {qq_mask.shape}')
    
    # Check if there are any odd orders to compute
    odd_orders_present = (orders%2).any()
    if odd_orders_present:
        l_stride = 1
    else:
        l_stride = 2

    # Calculate triangular PP matrices and harmoncic coefficents CCn of the cross-correlation
    #qq_matrices = ccd_associated_legendre_matrices(thetas,l_max,l_max)[:,:,::stride,::stride]
    #ccn = np.zeros(qq_matrices.shape[:3],dtype = complex)
    #ccn[qq_mask,:] = mLib.circularHarmonicTransform_real_forward(cc[qq_mask,:])[...,:l_max+1:stride]

    #b_coeff = np.zeros(ccn.shape,dtype = complex)
    q_ids = np.arange(len(qs))

    ccn =  mLib.circularHarmonicTransform_real_forward(cc)[...,:l_max+1]
    ccn = np.array([nearest_positive_semidefinite_matrix(ccn[...,o]) for o in range(ccn.shape[-1])])
    ccn = np.moveaxis(ccn,0,-1)
    ppm = ccd_associated_legendre_matrices(thetas,l_max,l_max)
    ccn[~qq_mask] = 0
    b_coeff = psd_back_substitution(ccn,ppm)[...,::l_stride]
    return b_coeff,qq_mask


def ccd_to_deg2_invariant_3d_legendre(cc,xray_wavelength,data_grid,orders,cc_mask):
    # Get Fl arguments 
    qs = data_grid['qs']
    phis=data_grid['phis']
    #log.info(f'phis = {phis}')

    # Legendre expansion coefficients only correspond to the deg2 invariants iw 
    # Only look at unsymmetric part of CC, i.e. \phi<\pi (since by definition C(q1,q2,phi) = C(q1,q2,2\pi-\phi)))
    #log.info(f'cc shape = {cc.shape}')
    cc,_,_ = modify_cross_correlation(cc,True,phis,orders.max(),pi_periodicity=True)
    
    #phi_pi_mask = (phis >= np.pi/2) & (phis < np.pi*3/2)
    #log.info(f'n phis < pi = {phi_pi_mask.sum()}')
    phis_reduced = phis # phis[phi_pi_mask].copy()
    cc_reduced = cc # cc[... , phi_pi_mask].copy()
    cc_mask_reduced =cc_mask # cc_mask[...,phi_pi_mask].copy()
    #log.info(f"cc_reduced shape = {cc_reduced.shape}")

    n_qs = len(qs)
    q_ids = np.arange(n_qs)
    qq_mask = np.prod(cc_mask_reduced,axis = -1).astype(bool)
    b_coeff = Multiprocessing.comm_module.request_mp_evaluation(bl_3d_legendre_worker,
                                                                       input_arrays=[q_ids,q_ids],
                                                                       const_inputs=[cc_reduced,phis_reduced,orders,qq_mask],
                                                                       call_with_multiple_arguments=True,n_processes=False)
    #b_coeff = np.moveaxis(b_coeff,-1,0)
    return b_coeff,qq_mask
def bl_3d_legendre_worker(q_ids,qq_ids,cc,phis,orders,qq_mask,**kwargs):
    '''
    Uses pseudoinverse of the legendre matrices (numpy internal uses a singular value decomposition)
    '''
    cc_parts=cc[q_ids,qq_ids,:]
    bl_part = np.zeros((len(q_ids),len(orders)),dtype=complex)
    q1q2_masks = qq_mask[q_ids,qq_ids]
    #log.info(orders)
    leg_forward = leg_trf.forward
    for n in range(len(q_ids)):
        cc_part = cc_parts[n]
        #log.info(f'cc_part.shape = {cc_part.shape}')        
        if q1q2_masks[n]:
            leg_coeff = leg_forward(cc_part,closed = True)
            #log.info(f'leg part shape = {leg_coeff.shape}')
            leg_coeff= leg_coeff[orders]*np.pi*4
            #log.info(f'leg part shape = {leg_coeff.shape}')
            #log.info(f'bl part shape = {bl_part.shape}')
            bl_part[n] = leg_coeff
    return bl_part


def ccd_to_deg2_invariant_2d(cc,data_grid,orders,cc_mask):
    '''
    Calculates the degree 2 invariant $B_m=I_m*I_m^*$ from Cross-Correlation data CC using
    $B_m = C_m$ where $C_m$ are the harmonic coefficients of CC.  
               
    :type kind: ndarray (shape = (n_q,n_q,n_delta))
    :return: B_m coeff.
    :rtype ndarray: complex, shape = ((n_orders,n_q,n_q)))
    '''
    try:
        assert cc_mask.all(), 'B_l extraction from fourier coefficients does not support masked data parts. Modify cc by interpolating (+ binned_mean).'
    except AssertionError as e:
        log.warning(e)
        cc,cc_mask,phis = modify_cross_correlation(cc,cc_mask,data_grid['phis'],max_order,interpolate_masked=True)        
    n_angular_points=cc.shape[-1]

    #log.info('n_angular_points={}'.format(n_angular_points))
    harm_trf = HarmonicTransform('real',{'dimensions':2,'n_phi':n_angular_points,'max_order':n_angular_points//2-1})
    #log.info('cc shape ={}'.format(cc.shape))
    #log.info('cc={}'.format(cc[1,:100,0]))
    b_coeff=np.moveaxis(harm_trf.forward(cc),2,0)[orders]
    #b_coeff=np.moveaxis(np.fft.rfft(cc,axis=-1),2,0)[orders]
    #log.info('b_coeff shape ={}'.format(b_coeff))
    #lop.info(f'initial mask shape = {cc_mask.shape}')
    qq_mask=cc_mask.prod(axis = -1).astype(bool)
    #lop.info(f'final mask shape = {qq_mask.shape}')
    return b_coeff,qq_mask


def modify_deg2_invariant(b_coeff,enforce_psd=True, psd_q_min_id=0):
    #It follows from the connection of the averaged crosscorrelation with the invariant B_l that:
    # cc[0,0,0] = F_0 * B_0  and F_0 = 1/(4pi)
    #if cc_mask[0,0,0]:
    #    b_coeff[0,0,0]=4*np.pi*cc[0,0,0]
    # assumes qs[0] = 0 => same value for all angular points => only zeros order has nonzero contribution. 
    #b_coeff[1:,0,0]=0

    #if not isinstance(n_particles,bool):
    #    #correct for multiparticle crosscorrelation
    #    log.info("\n\n\n Numer of particles = {}. Correcting Bl_coefficients(shape = {}) for that.\n\n\n".format(n_particles,b_coeff.shape))
    #    b_coeff[0]/=n_particles**2
    #    b_coeff[1:]/=n_particles

    if enforce_psd:
        qid = psd_q_min_id
        b_coeff[:,qid:,qid:] = nearest_positive_semidefinite_matrix(b_coeff[:,qid:,qid:])
    return b_coeff

def deg2_invariant_apply_precision_filter(bl,precision):
    """
    for each q,qq pair set values whos absolute valu is smaller than abs_max*precision to 0.
    """
    abs_bl = np.abs(bl)
    abs_max = np.max(abs_bl,axis=0)
    invalid_mask = abs_bl < abs_max[None,...]*precision
    bl[invalid_mask] = 0
    return bl


def intensity_to_deg2_invariant(intensity,intensity2=False,cht = False):
    if not isinstance(cht,HarmonicTransform):
        cht = HarmonicTransform.from_data_array('complex',intensity)
    dimensions = intensity.ndim
    harm_coeff = cht.forward(intensity.astype(complex))
    #print(f'cht datatype = {type(cht.data_type)}')
    if dimensions==2 and cht.data_type=='complex':
        #print(f'intensity shape = {intensity.shape}')
        n_orders = intensity.shape[-1]//2+1
        harm_coeff = harm_coeff[...,:n_orders]
        #print(f'harm shape = {harm_coeff.shape}')
    if isinstance(intensity2,bool):
        deg2_invariants = harmonic_coeff_to_deg2_invariants(dimensions,harm_coeff)
    else:
        harm_coeff2 = cht.forward(intensity2.astype(complex))
        deg2_invariants = harmonic_coeff_to_deg2_invariants(dimensions,harm_coeff,harm_coeff2 = harm_coeff2)
    return deg2_invariants
def density_to_deg2_invariants(density,fourier_transform,dimensions,density2 = False,cht = False):
    '''Assumes that the density is sampled over the same grid for which the fourier_transform is defined.'''
    ftd = fourier_transform(density)
    intensity = ftd*ftd.conj()
    if isinstance(density2,bool):
        return intensity_to_deg2_invariant(intensity,cht = cht)
    else:
        ftd2 = fourier_transform(density2)
        intensity2 = ftd2*ftd2.conj()
    return intensity_to_deg2_invariant(intensity,intensity2 = intensity2,cht = cht)
def harmonic_coeff_to_deg2_invariants(dimensions,harm_coeff,harm_coeff2 = False):
    if dimensions ==2:
        deg2_invariants = harmonic_coeff_to_deg2_invariants_2d(harm_coeff,Ims2=harm_coeff2)
    if dimensions == 3:
        deg2_invariants = harmonic_coeff_to_deg2_invariants_3d(harm_coeff,Ilm2=harm_coeff2)
    return deg2_invariants

def harmonic_coeff_to_deg2_invariants_2d(Ims,Ims2=False):
    '''
    Calculates the degree 2 invariants $B_m$ via $B_m = I_{m} I^*_{m}$.    
    '''
    if isinstance(Ims2,bool):
        Bm = np.array(tuple(Im[:,None]*Im[None,:].conj() for Im in Ims.T))
    else:
        Bm = np.array(tuple(Im1[:,None]*Im2[None,:].conj() for Im1,Im2 in zip(Ims.T,Ims2.T)))
    return Bm
def harmonic_coeff_to_deg2_invariants_3d(Ilm,Ilm2 = False):
    '''
    Calculates the degree 2 invariants $B_l$ via $B_l = \sum_l I_{lm} I^*_{lm}$. 
    '''
    if isinstance(Ilm2,bool):
        Bl = np.array(tuple(Il @ Il.T.conj() for Il in Ilm))
    else:
        Bl = np.array(tuple(Il1 @ (Il2.T.conj()) for Il1,Il2 in zip(Ilm,Ilm2)))
    return Bl


def intensity_to_cc_3d(intensity,xray_wavelength,qs,phis=False,cc_mode='Pl',cht = False,intensity2 = False):
    bl = intensity_to_deg2_invariant(intensity,intensity2=intensity2,cht = cht)
    if not isinstance(phis,np.ndarray):
        phis = np.linspace(0,2*np.pi,2*(len(bl)-1)+1)
    cc = deg2_invariant_to_cc_3d(bl,xray_wavelength,{'qs':qs,'phis':phis},mode=cc_mode)
    return cc


def deg2_invariant_to_cc_2d(bl,cht):
    #print(f'bl shape = {bl.shape}')
    bl = np.moveaxis(bl,0,-1)
    cc = mLib.circularHarmonicTransform_real_inverse(bl,2*(bl.shape[-1]-1))
    #print(f'cc shape = {cc.shape} type {cc.dtype}')
    return cc

def deg2_invariant_to_cc_3d(bl,xray_wavelength,data_grid,orders=False,mode='back_substitution',n_processes = False):
    '''
    Method to recompute the averaged cross correlation $\mathcal{C}(q_1,q_2,\Delta)$ from the degree 2 invariants $B_l$, via $\mathcal{C} = \sum_l F_l B_l$
    Assumes Phi grid is uniform ranging from [0 to 2*Pi) such that Pi is contained in the grid.
    '''
    extraction_modes = {'lstsq':cc_3d_Fl_worker,'back_substitution':cc_3d_Pl_worker,'legendre':cc_3d_legendre_worker}
    try:
        worker = extraction_modes[mode]
    except KeyError as e :
        log.error('Given cc creation mode "{}" is unknown. Known modes are {}"'.format(mode,list(extraction_modes.keys())))
    if not isinstance(orders,np.ndarray):
        orders = np.arange(len(bl))
    qs=data_grid['qs']
    q_ids=np.arange(len(qs),dtype=int)
    phis=data_grid['phis']


    # log.info('\n \n len phis = {} \n \n'.format(len(phis)))
    # log.info('original cc shape = {}'.format(cc.shape))
    # log.info('new cc shape = {}'.format(cc.shape))
    
    thetas = ewald_sphere_theta_pi(xray_wavelength,qs)    
    if mode == 'lstsq':
        cc = np.zeros((len(qs),len(qs),len(phis)),dtype=complex)
        phis = phis[phis <= np.pi]
        cc_no_sym = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[q_ids,q_ids], const_inputs=[bl,phis,thetas,orders], call_with_multiple_arguments=True,split_mode='modulus',n_processes=n_processes)
        #log.info('cc no sym shape = {}'.format(cc_no_sym.shape))
        #log.info('cc <= pi shape  = {}'.format(cc[...,data_grid['phis']<=np.pi].shape))
        #log.info('cc > pi shape  = {}'.format(cc[...,data_grid['phis']>np.pi].shape))
        cc[...,data_grid['phis']<=np.pi] = cc_no_sym
        cc[...,data_grid['phis']>np.pi] = cc_no_sym[...,1:-1][...,::-1]
    elif mode == 'legendre':
        cc_no_sym = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[q_ids,q_ids], const_inputs=[bl.real], call_with_multiple_arguments=True,split_mode='modulus')
        n_phis_leg_pi = cc_no_sym.shape[-1]
        n_phis = 2*n_phis_leg_pi - 2
        cc = np.zeros((len(qs),len(qs),n_phis),dtype=float)
        cc[...,:n_phis_leg_pi] = cc_no_sym
        cc[...,n_phis_leg_pi:] = cc_no_sym[...,1:-1][...,::-1]
    elif mode == 'back_substitution':
        l_max=orders.max()
        #qq_matrices = ccd_associated_legendre_matrices(thetas,l_max,l_max)
        #qq_matrices = np.moveaxis(qq_matrices,-1,0)
        #cns2 = np.sum(bl[:,:,:,None]*qq_matrices,axis = 0)
        cns = np.zeros((len(qs),len(qs),l_max+1),dtype = complex )
        for l in np.arange(l_max+1):
            qq_matrix = ccd_associated_legendre_matrices_single_l(thetas,l_max,l)
            cns += bl[l,...,None]*qq_matrix
        cc = mLib.circularHarmonicTransform_real_inverse(cns,2*(cns.shape[-1]-1))
        
    return cc

def cc_3d_Fl_worker(q_ids,qq_ids,bl,phis,thetas,orders,**kwargs):
    '''
    Uses  $\mathcal{C} = \sum_l F_l B_l$  to calculate the averaged cross-correlation $\mathcal{C}$.
    '''
    legendre_matrices = ccd_legendre_matrices(q_ids,qq_ids,phis,thetas,orders)
    bl_part = np.moveaxis(bl,0,-1)[q_ids,qq_ids,:]
    #log.info('bl_shape = {}, f_l shape = {} bl_part_shape = {}'.format(bl.shape,legendre_matrices.shape,bl_part.shape))
    #log.info('n_orders = {}'.format(len(orders)))
    cc=np.sum(legendre_matrices*bl_part[:,None,:],axis = -1)
    return cc

def cc_3d_Pl_worker(q_ids,qq_ids,bl,thetas,orders,**kwargs):    
    cn = deg2_invariant_to_cn_3d_worker(q_ids,qq_ids,bl,thetas,orders)
    #log.info(f'cn shape = {cn.shape} bl shape = {bl.shape}')
    cc = mLib.circularHarmonicTransform_real_inverse(cn,2*(cn.shape[-1]-1))
    return cc
def deg2_invariant_to_cn_3d_worker(q_ids,qq_ids,bl,thetas,orders):
    #log.info(f'bl shape = {bl.shape} qs shape = {qs.shape} thetas shape = {thetas.shape} orders = {orders}')
    max_order = np.max(orders)
    thetas1 = thetas[q_ids]
    thetas2 = thetas[qq_ids]
    qq_matrices= ccd_associated_legendre_matrices_q1q2(thetas1,thetas2,max_order,max_order)
    qq_matrices= qq_matrices[...,orders]
    qq_matrices = np.moveaxis(qq_matrices,-1,0)
    cns = np.sum(bl[:,q_ids,qq_ids,None]*qq_matrices,axis = 0)    
    #log.info('cns shape = {}'.format(cns.shape))
    return cns
def deg2_invariant_to_cn_3d(bl,radial_points,xray_wavelength):
    max_order = len(bl)-1
    thetas = ewald_sphere_theta_pi(xray_wavelength,radial_points)
    qq_matrices= ccd_associated_legendre_matrices(thetas,max_order,max_order)
    qq_matrices = np.moveaxis(qq_matrices,-1,0)
    cns = np.sum(bl[...,None]*qq_matrices,axis = 0)
    log.info('cns shape = {}'.format(cns.shape))
    return cns

def cc_3d_legendre_worker(q_ids,qq_ids,bl,**kwargs):
    leg_inverse = leg_trf.inverse
    cc = np.array(tuple(leg_inverse(bl[:,q1,q2]/(np.pi*4),closed=True) for q1,q2 in zip(q_ids,qq_ids)))
    return cc


def estimate_number_of_particles2(deg2_invariants,radial_points,search_space,average_intensity=False):
    '''
    Routine that estimates the number of particles based on the degree 2 invariant $B_l(q_1,q_2) = \sum_{m} I_{lm}(q_1)\overline{I}_{lm}(q_2)$.
    It uses the fact that the 0 order harmonic coefficient of the cross correlation needs to be positive and that:
    $C_0(q_1,q_2) = B_0(q_1,q_2) + \sum_{l>0} B_l(q_1,q_2) P_l(\cos(q_1))P_l(\cos(q_2))$
    We realized that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param deg2_invariants: $B_l$'s
    :type numpy.ndarray: of shape (orders,q_1,q_2)
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    max_order = len(deg2_invariants)- 1
    nq = len(radial_points)
    Us = [ np.zeros((min(nq, 2*o+1)),dtype=complex) for o in range(max_order+1)]
    for u in Us:        
        u[0]=1
    if isinstance(average_intensity,np.ndarray):
        log.info('use average intensity')
        log.info('a int shape = {}'.format(average_intensity.shape))
        I0 = -average_intensity[:,None]
    else:
        log.info('use V 0')
        I0 = projection_matrices[0]*Us[0]/(2*np.sqrt(np.pi))
        
    scales = np.linspace(*search_space)
    nq = len(radial_points)
    
    I_l0 = [projection_matrices[o].dot(Us[o])*np.sqrt((2*o+1)/(4*np.pi)) for o in range(max_order+1)]
    #log.info(I_l0)
    worker = estimate_number_of_particles_worker2
    neg_volumes = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[scales], const_args=[I_l0,I0,radial_points], callWithMultipleArguments=True,splitMode = 'modulus')
    grad = np.gradient(neg_volumes,scales[1]-scales[0])
    n_particles = scales[np.argmax(np.abs(grad))]**2        
    return n_particles,neg_volumes


########### deg2 invariants and reciprocal projection matrices ###########
def deg2_invariant_to_projection_matrices(dim,b_coeff,q_id_limits=False,sort_mode=0):
    '''
    Extracts projection matrices from the deg 2 invariants b_coeff.
    :param dim: Dimension
    :type int: 2 or 3
    :param b_coeff:
    :type real nd_array: shape= (n_orders,q_1,q_2)
    :param sort_mode: sort_mode for eigenvalu/eigenvector sorting of deg2_invariants see function deg2_invariant_eigenvalues for details deg2_invariant_eigenvalues
    :type int: 0 or 1
    :return proj_matrices:
    :type real nd_array: shape= (n_orders,q)
    '''    
    if isinstance(q_id_limits,bool):
        q_id_limits = np.zeros((b_coeff.shape[0],)+(2,2),dtype = int)
        q_id_limits[...,1]=b_coeff.shape[-1]
    try:
        #log.info(f'q_id limits shape = {q_id_limits.shape}')
        assert (q_id_limits[:,0,:]==q_id_limits[:,1,:]).all(),'Trying to compute projection matrices from non-square submatrix of deg2 invariants specified. That makes no sense since projection matrices are theoretically given by eig values of a positive semidefinite matrix. Continue using the q1_id_limits also for q2 to make the selection square.'
    except AssertionError as e:
        log.info(e)
        q_id_limits[:,1]=q_id_limits[:,0]
    orders = np.arange(len(b_coeff))
    if dim == 2:
        temp = Multiprocessing.comm_module.request_mp_evaluation(deg2_invariant_to_projection_matrices_2d,input_arrays=(b_coeff,q_id_limits,orders),const_inputs=[sort_mode],split_together=True,split_mode='modulus')
        proj_matrices = tuple(p for p,eig in temp)
        eigen_values = tuple(eig for p,eig in temp)
    elif dim == 3:        
        temp = Multiprocessing.comm_module.request_mp_evaluation(deg2_invariant_to_projection_matrices_3d,input_arrays=(b_coeff,q_id_limits,orders),const_inputs=[sort_mode],split_together = True,split_mode='modulus')
        proj_matrices = tuple(p for p,eig in temp)
        eigen_values = tuple(eig for p,eig in temp)
        bad_eigen_values = [e==0 for e in eigen_values]
        bad_eigen_values_detected = np.array([m.any() for m in bad_eigen_values])
        #if bad_eigen_values_detected.any():
        #    log.info('Eigenvalues lower than Noise floor estimate detected in orders {}. Threating them as 0.'.format(orders[bad_eigen_values_detected]))
    return  (proj_matrices,eigen_values)
def deg2_invariant_eigenvalues(b_matrix,sort_mode=0):
    # Calculates the eigenvalue/eignvector pairs and sorts them.
    # Assumes input matrix to be hermitian
    # sort_mode can be 0 or 1
    # case sort_mode == 0: eigenvectors and eigenvalues are sorted by eigenvalues
    # case sort_mode == 1: eigenvectors and eigenvalues are sorted by median of the product of sqrt(eigenvalue)*eigenvector

    # Comment In some 2d examples sorting purely by eigenvalue(sort_mode == 0) has failed due to a verry small associated eigenvetor (which was almost everywhere 0)
    b_matrix = (b_matrix + b_matrix.T.conj())/2
    is_zero = np.isclose(b_matrix,0).all() 
    if not is_zero:
        # at the time of writing this code comparing the eigen_decomposition of b_matrix with itself yielded lower errors when using scipy linalg.eigh with driver 'ev'. 
        eig_vals,eig_vect=sp.linalg.eigh(b_matrix,driver = 'ev') 
        #eig_vals,eig_vect=np.linalg.eigh(b_matrix)
    else:
        eig_vect = np.zeros(b_matrix.shape)
        eig_vals = np.zeros(b_matrix.shape[0])            
    #log.info(eig_vals)
    eig_val_signs = np.sign(eig_vals)
    if sort_mode == 0:
        sort_metric = eig_vals
        #log.info('yay')
    elif sort_mode == 1:
        sort_metric = np.median(np.abs(np.sqrt(np.abs(eig_vals[None,:]))*eig_vect),axis = 0)*eig_val_signs
    sorted_ids=np.argsort(sort_metric)[::-1]
    eig_vals = eig_vals[sorted_ids].real
    eig_vect = eig_vect[:,sorted_ids]
    return eig_vals,eig_vect

def deg2_invariant_to_projection_matrices_2d(b_coeff,q_id_limits,order,sort_mode,**kwargs):
    q_slice = slice(*q_id_limits[0])
    #log.info(f'q_slice = {q_slice}')
    eig_vals,eig_vect = deg2_invariant_eigenvalues(b_coeff[q_slice,q_slice],sort_mode=sort_mode)
    # rank of b_coeff should be 1 => only one eigenvalue
    #log.info(f'order {order} eig vals = {eig_vals[0]} \n')
    eig_val = eig_vals[0]
    eig_vect = eig_vect[:,0]
    #log.info("order = {} max_eig = {}".format(order,eig_val))
    # eig_value should be positive since B_coeff should be positive semidefinite
    #order = 1
    try:            
        assert eig_val>=0, 'Eigenvalues of B_l ar not real or positive!'.format(eig_vals)
    except AssertionError as e:            
        log.warning('Negative eigenvalues detected for order {}. Setting them to 0'.format(order))
        eig_val = 0
        eig_vect[:] = 0
        #eig_vect=eig_vect[:,real_and_positive_mask]

    full_eig_vect = np.zeros((len(b_coeff)),dtype = eig_vect.dtype)
    full_eig_vect[q_slice] = eig_vect
    V=full_eig_vect
    G=np.sqrt(eig_val)
    #            log.info('lambda={}'.format(G))
    #log.info("V shape = {} eig val = {}".format(V.shape,G))
    proj_matrix=V*G
    return proj_matrix,eig_val

def deg2_invariant_to_projection_matrices_3d(b_coeff,q_id_limits,order,sort_mode,**kwargs):
    q_slice = slice(*q_id_limits[0])
    #log.info(f'q_slice = {q_slice}')
    eig_vals,eig_vect = deg2_invariant_eigenvalues(b_coeff[q_slice,q_slice],sort_mode=sort_mode)
    # Eigenvalues should be positive since Bl should be positive semidefinite
    # Using the absolute value of the lowest negative eigenvalue to as threshold for the noese floor of positive eigenvalues
    #log.info(f'len eigvals  = {len(eig_vals)}')
    #log.info(f'b_coeff shape  = {b_coeff.shape}')
    if len(eig_vals)!=0:
        min_eigvals_threshold = np.abs(np.min(eig_vals))
        #log.info(np.min(eig_vals))
        # N is the matrix rank of b_coeff
        N=min(len(eig_vect),2*order+1)
        # There should be only N eigenvalues since N is the rank of b_coeff
        eig_vect = eig_vect[:,:N]
        eig_vals = eig_vals[:N]
   
        #noise_mask = eig_vals < min_eigvals_threshold
        #noise_mask[0]=False
        #eig_vals[noise_mask] = 0
        #eig_vect[:,noise_mask] = 0
        
        neg_mask = eig_vals < 0
        eig_vals[neg_mask] = 0
        eig_vect[:,neg_mask] = 0
        
    NN = min(len(b_coeff),2*order+1)
    full_eig_vect = np.zeros((len(b_coeff),NN),dtype = eig_vect.dtype)
    full_eig_vals = np.zeros(NN,dtype = eig_vals.dtype)
    if len(eig_vals!=0):
        full_eig_vect[q_slice,:N] = eig_vect
        full_eig_vals[:N]=eig_vals
    V=full_eig_vect # n_radial_points x N array
    G=np.diag(np.sqrt(full_eig_vals))
    proj_matrix=V @ G
    #log.info('proj matrix shape = {}'.format(proj_matrix.shape))
    return proj_matrix.astype(complex),full_eig_vals



def eig_to_projection_matrices_3d(eig_vals,eig_vect,order,**kwargs):
    # N is the matrix rank of b_coeff
    N=min(len(eig_vect),2*order+1)
    # There should be only N eigenvalues since N is the rank of b_coeff
    eig_vect = eig_vect[:,:N]
    eig_vals = eig_vals[:N]
    # Eigenvalues should be positive since Bl should be positive semidefinite
    negative_mask = eig_vals < 0
    try:            
        assert not (negative_mask).any(), 'Eigenvalues of B_l ar not real or positive!'.format(eig_vals)
    except AssertionError as e:            
        log.warning('Negative eigenvalues detected for order {}. Setting them to 0'.format(order))
        #log.info('eig_vals={}'.format(eig_vals))
        eig_vals[negative_mask] = 0
        #eig_vect[:,negative_mask] = 0
        #eig_vect=eig_vect[:,real_and_positive_mask]
    
    V=eig_vect # n_radial_points x N array
    G=np.diag(np.sqrt(eig_vals))
    proj_matrix=V @ G    
    return proj_matrix

def projection_matrices_to_deg2_invariant_3d(projection_matrices):
    '''
    Uses the $(N_q,min(2*l+1),N_q)$ shaped proj_matrices $V_l$ and computes the deg2 invariants $B_l$ by:
    $B_l= V_l V_l^\dagger$
    Where $N_q$ is the number of radial sampling points and $l$ is the spherical harmonic order.
    :param projection_matrices: list of V_l sorted by l
    :type list: 
    :return B_l: array of the deg2 invariants
    :rtype ndarray: shape: (L_max,N_p,N_p)
    '''
    return spherical_harmonic_coefficients_to_deg2_invariant(projection_matrices)

def spherical_harmonic_coefficients_to_deg2_invariant(I_coeff):
    '''
    Uses the $(N_q,2*l+1)$ shaped intensity harmonic coefficients $I_l$ and computes the deg2 invariants $B_l$ by:
    $B_l= I_l I_l^\dagger = \sum_m I_{lm} \overline{I}_{lm}$
    Where $N_q$ is the number of radial sampling points and $l$ is the spherical harmonic order.
    param I_coeff: list of I_l sorted by l
    :type list: 
    :return B_l: array of the deg2 invariants
    :rtype ndarray: shape: (L_max,N_p,N_p)
    '''
    B_l = np.array(tuple(I_l @ I_l.conj().T for I_l in I_coeff)).real
    return B_l


def calc_projection_matrix_error_estimate(deg2_invariant,proj_matrices):
    errors = np.full_like(deg2_invariant,-1)
    #log.info(f'projection_matrix_shape = {len(proj_matrices)}')
    for b,pr,e in zip(deg2_invariant,proj_matrices,errors):
        #log.info(f'projection_matrix_shape = {pr.shape}')
        if pr.ndim ==1:
            pr = pr[:,None]
        non_zero_mask = (b != 0)
        e[non_zero_mask] = np.abs(b[non_zero_mask]-(pr@pr.conj().T)[non_zero_mask])/np.abs(b[non_zero_mask])
    return errors

######## projection pmatrices ########
def enforce_spherical_harmonic_transform_constraint(proj_I1I1,iterations,spherical_harmonic_transform,rel_err_limit=1e-6):
    error_target_reached = False
    cht = spherical_harmonic_transform.forward
    icht = spherical_harmonic_transform.inverse
    P = proj_I1I1
    V=[p.copy() for p in proj_I1I1]
    #log.info(f'V shapes = {[v.shape for v in V]}')
    err_old = np.inf
    for i in range(iterations):
        if error_target_reached:
            break
        I = icht(V)
        I[I<0]=0
        I.imag[:]=0
        V = cht(I)    
        V = [p@solve_procrustes_problem(v,p) for v,p in zip(V,P)]        
        if i%10==0:
            Inew = icht(V)
            err = np.sum(np.abs(I-Inew)/np.abs(Inew))
            #log.info(f'Iteration {i+1} of {iterations} error = {err}.')
            if err_old!=np.inf:
                error_target_reached =  np.abs(err_old-err)/err_old < rel_err_limit
            err_old = err
            
    return V,error_target_reached
        
def calc_unknown_unitary_transform(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,b_coeff_I2I1,radial_grid_points,q_id_limits = False,method='procrustes'):
    methods = {'procrustes':calc_unknown_unitary_transform_procrustes,"direct":calc_unknown_unitary_transform_direct}
    method = methods.get(method,calc_unknown_unitary_transform_procrustes)
    log.info(f'Extracting W using method {method}')
    w,errors  = method(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,b_coeff_I2I1,radial_grid_points,q_id_limits = q_id_limits)
    return w,errors
def calc_unknown_unitary_transform_worker(o,P,Vl2,**kwargs):
    p = P[o]
    v2 = Vl2[o]
    #log.info(f'solving procrustes for p {p.shape} and v2 {v2.shape}')
    w = solve_procrustes_problem(p,v2)
    #log.info(f'result shape {w.shape}')
    return w
    
def calc_unknown_unitary_transform_procrustes(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,b_coeff_I2I1,radial_grid_points,q_id_limits = False):
    '''
    This caclulates a unitary matrix that transforms the unknowns coming from the deg two invariant of I with the ones of I^2
    '''
    # B = b_coeff_I2I1 is Nq,Nq matrix, harmonic coeffs I_l are Nq,N matrices then we have:
    # B = I_{l,2} I_{l}^\dagger
    # I_ls are given by the projection matrices V (Nq,N) and an unknown matrix U (N,N) as I_l = V_lU_l
    # B = I_{l,2} I_{l}^\dagger = V_{l,2}U_{l,2} U_l^\dagger V_l^\dagger
    # the projection matrices are orthogonal lets call d_l,2 = diag(V_{l,2} V_{l,2}^\dagger) and similar for V_l then
    #  B V_l/d_l = V_l,2 U_{l,2} U_l^\dagger = V_l,2 W
    # We now seek to find a unitary W such that the frobenus norm ||B V_l/d_l-V_l,2 W || becomes minimal 
    # This is a unitary procrustres problem whose unique solution can be found using singular value decomposition.
    if not isinstance(q_id_limits,np.ndarray):
        q_id_limits = np.zeros((b_coeff_I2I1.shape[0],)+(2,2),dtype = int)
        q_id_limits[...,1]=b_coeff_I2I1.shape[-1]

    
    D = radial_grid_points
    #D = np.ones_like(radial_grid_points)
    Bls = [b[slice(*lim[0]),slice(*lim[1])] for b,lim in zip(b_coeff_I2I1,q_id_limits)]
    Vldl = []
    orders = np.arange(len(proj_I1I1))
    for p,e,lim,o in zip(proj_I1I1,eig_I1I1,q_id_limits[:,1],orders):
        #log.info(f'I1I1 limits = {lim}')
        pe=p[slice(*lim),:].copy()
        N=min(len(pe),2*o+1)
        pe = pe[:,:N]
        e=e[:N]
        mask = (e<=0)
        pe[:,mask]=0
        pe[:,~mask]/=e[None,~mask]
        Vldl.append(pe)
    #log.info(f'bls shapes = {[b.shape for b in Bls]}')
    P = [(D[slice(*lim),None]*b) @ vd for b,vd,lim in zip(Bls,Vldl,q_id_limits[:,0])]
    Vl2 = []
    for p,lim,o in zip(proj_I2I2,q_id_limits[:,0],orders):
        p2=p[slice(*lim),:].copy()
        N=min(len(p2),2*o+1)
        p2 = p2[:,:N]
        Vl2.append(p2)
    W = [np.eye(2*o+1,dtype = complex) for o in orders]
    W = [w[:min(lim[0,1]-lim[0,0],2*o+1),:min(lim[1,1]-lim[1,0],2*o+1)] for w,lim,o in zip(W,q_id_limits,orders)]
    #log.info(f'P shapes = {[p.shape for p in P]}')
    #log.info(f'Vl2 shapes = {[p.shape for p in Vl2]}')
    W_even = Multiprocessing.comm_module.request_mp_evaluation(calc_unknown_unitary_transform_worker,input_arrays=[orders[::2],],const_inputs=[P,Vl2],split_mode='modulus',call_with_multiple_arguments = False)
    for o,w in zip(orders[::2],W_even):
        W[o][:]=w
    #log.info(f"w even shapes = {[w.shape for w in W_even]}")
    #log.info(f"w shapes = {[w.shape for w in W]}")
    errors = np.full_like(b_coeff_I2I1,-1)
    for w,b,pr1,pr2,lim,e,o in zip(W,Bls,proj_I1I1,proj_I2I2,q_id_limits,errors,orders):
        q1_slice = slice(*lim[1])
        q2_slice = slice(*lim[0])
        N1 = min(lim[1,1]-lim[1,0],2*o+1)
        N2 = min(lim[0,1]-lim[0,0],2*o+1)
        pr11 = pr1[q1_slice,:N1].copy()
        pr22 = pr2[q2_slice,:N2].copy()
        #log.info(f'b shape {b.shape} - {pr22.shape}{w.shape}{pr11.T.shape}')
        error = e.copy()
        non_zero_mask = (b!=0)
        error[non_zero_mask] = np.abs(b[non_zero_mask]-(pr22@w@(pr11.conj().T))[non_zero_mask])/np.abs(b[non_zero_mask])
        e[q2_slice,q1_slice]=error
    return W,errors

def calc_unknown_unitary_transform_direct(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,b_coeff_I2I1,radial_grid_points,q_id_limits = False):
    '''
    This caclulates a unitary matrix that transforms the unknowns coming from the deg two invariant of I with the ones of I^2
    '''
    # B = b_coeff_I2I1 is Nq,Nq matrix, harmonic coeffs I_l are Nq,N matrices then we have:
    # B = I_{l,2} I_{l}^\dagger
    # I_ls are given by the projection matrices V (Nq,N) and an unknown matrix U (N,N) as I_l = V_lU_l
    # B = I_{l,2} I_{l}^\dagger = V_{l,2}U_{l,2} U_l^\dagger V_l^\dagger
    # the projection matrices are orthogonal lets call d_l,2 = diag(V_{l,2} V_{l,2}^\dagger) and similar for V_l then
    #  B V_l/d_l = V_l,2 U_{l,2} U_l^\dagger = V_l,2 W
    # We now seek to find a unitary W such that the frobenus norm ||B V_l/d_l-V_l,2 W || becomes minimal 
    # This is a unitary procrustres problem whose unique solution can be found using singular value decomposition.
    if not isinstance(q_id_limits,np.ndarray):
        q_id_limits = np.zeros((b_coeff_I2I1.shape[0],)+(2,2),dtype = int)
        q_id_limits[...,1]=b_coeff_I2I1.shape[-1]

    
    D = radial_grid_points
    Bls = [b[slice(*lim[0]),slice(*lim[1])] for b,lim in zip(b_coeff_I2I1,q_id_limits)]
    Vldl = []
    Vl2dl = []
    orders = np.arange(len(proj_I1I1))
    for p,e,p2,e2,lim,o in zip(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,q_id_limits,orders):
        #log.info(f'I1I1 limits = {lim}')
        pe=p[slice(*lim[1]),:].copy()
        pe2=p2[slice(*lim[0]),:].copy()
        N=min(len(pe),2*o+1)
        N2=min(len(pe2),2*o+1)
        pe = pe[:,:N]
        pe2 = pe2[:,:N2]
        e=e[:N]
        e2=e2[:N2]
        mask = (e<=0)
        mask2 = (e2<=0)
        pe[:,mask]=0
        pe[:,~mask]/=e[None,~mask]
        pe2[:,mask2]=0
        pe2[:,~mask2]/=e2[None,~mask2]
        Vldl.append(pe)
        Vl2dl.append(pe2)

    W = [np.eye(2*o+1,dtype = complex) for o in orders]
    W = [w[:min(lim[0,1]-lim[0,0],2*o+1),:min(lim[1,1]-lim[1,0],2*o+1)] for w,lim,o in zip(W,q_id_limits,orders)]
    for o,b,v,v2 in zip(orders,Bls,Vldl,Vl2dl):
        #log.info(f'w shape {W[o].shape}, {v2.T.shape} {b.shape} {v.shape}')
        W[o][:]= v2.conj().T@b@v
    #log.info(f"w even shapes = {[w.shape for w in W_even]}")
    #log.info(f"w shapes = {[w.shape for w in W]}")
    errors = np.full_like(b_coeff_I2I1,-1)
    for w,b,pr1,pr2,lim,e,o in zip(W,Bls,proj_I1I1,proj_I2I2,q_id_limits,errors,orders):
        q1_slice = slice(*lim[1])
        q2_slice = slice(*lim[0])
        N1 = min(lim[1,1]-lim[1,0],2*o+1)
        N2 = min(lim[0,1]-lim[0,0],2*o+1)
        pr11 = pr1[q1_slice,:N1].copy()
        pr22 = pr2[q2_slice,:N2].copy()
        #log.info(f'b shape {b.shape} - {pr22.shape}{w.shape}{pr11.T.shape}')
        error = np.abs(b-pr22@w@(pr11.conj().T))/np.abs(b)
        e[q2_slice,q1_slice]=error
    return W,errors
    
    
def rank_projection_matrices(dimensions,projection_matrices,orders,radial_points,radial_high_pass=0.15):
    '''ranks the projection matrices by the radial part of the L2 norm.'''
    if dimensions == 2:
        ranked_ids,ranked_orders,metric = rank_projection_matrices_2d(projection_matrices,orders,radial_points,radial_high_pass=radial_high_pass)
    elif dimensions == 3:
        ranked_ids,ranked_orders,metric = rank_projection_matrices_3d(projection_matrices,orders,radial_points,radial_high_pass=radial_high_pass)
    return ranked_ids,ranked_orders,metric

def rank_projection_matrices_2d(projection_matrices,orders,radial_points,radial_high_pass=0.15):
    "ranks the projection vectors for different harmonic orders by the radial part of the L2 norm "
    radial_high_pass_index = int((len(radial_points)-1)*radial_high_pass)
    radial_points=radial_points[radial_high_pass_index:]
    integrator =  RadialIntegrator(radial_points,2)
        
    even_order_mask= orders%2 == 0
    non_zero_mask= orders != 0
    order_mask=even_order_mask*non_zero_mask
    relevant_orders=orders[order_mask]
    #relevant_order_ids = order_ids[order_mask]
    log.info("pr shape = {} order mask shape = {}".format(projection_matrices.shape,order_mask.shape))
    projection_vector=projection_matrices[order_mask,radial_high_pass_index:]

    metric = integrator.L2_norm(projection_vector,axis=-1)

    sorted_indices = np.argsort(metric)[::-1]
    #log.info(sorted_indice)
    SO_order_indices=order_mask.nonzero()[0][sorted_indices]
    SO_orders=orders[SO_order_indices]
    return SO_order_indices,SO_orders,metric[sorted_indices]

def rank_projection_matrices_3d(projection_matrices,orders,radial_points,radial_high_pass=0.15):
    radial_high_pass_index = int((len(radial_points)-1)*radial_high_pass)
    radial_points=radial_points[radial_high_pass_index:]
    integrator =  RadialIntegrator(radial_points,2)
        
    even_order_mask= orders%2 == 0
    non_zero_mask= orders != 0
    order_mask=even_order_mask*non_zero_mask
    relevant_orders=orders[order_mask]
    relevant_ids = np.nonzero(order_mask)[0]
    metrics = []
    bls = spherical_harmonic_coefficients_to_deg2_invariant(tuple(projection_matrices[_id] for _id in relevant_ids))
    for bl in bls:
        #log.info(bl.shape)
        metric = integrator.L2_norm(integrator.L2_norm(bl[radial_high_pass_index:,radial_high_pass_index:]))
        metrics.append(metric)
    metrics = np.array(metrics)
    sorted_indices = np.argsort(metrics)[::-1]
    SO_order_indices=order_mask.nonzero()[0][sorted_indices]
    SO_orders=orders[SO_order_indices]
    return SO_order_indices,SO_orders,metrics[sorted_indices]
#     "ranks the projection matrices for different major harmonic orders by radial integration of the corresponding Bl(q,q) coefficiets "
#    radial_high_pass=0
#    if not isinstance(radial_high_pass_index,bool):
#        radial_high_pass=radial_high_pass_index
#    radial_points=radial_points[radial_high_pass:]
#    integrator =  RadialIntegrator(radial_points,2)
#        
#    even_order_mask= orders%2 == 0
#    non_zero_mask= orders != 0
#    order_mask=even_order_mask*non_zero_mask
#    relevant_orders=orders[order_mask]
#    n_relevant_radial_points = len(radial_points[radial_high_pass:])
#    #relevant_order_ids = order_ids[order_mask]
#    log.info("pr shape = {} order mask shape = {}".format(projection_matrices.shape,order_mask.shape))
#
#    values = np.zeros(len(relevant_orders),n_relevant_radial_points)
#    for i,order in enumerate(relevant_orders):
#        log.info('proj_matrix shape = {}'.format(projection_matrices[order].shape))
#        projection_matrix = projection_matrices[order][:,radial_high_pass:]
#        values[i] = np.sum((projection_matrix*projection_matrix.conj()).real,axis=0)
#    metric = np.mean(integrator.integrate(values,axis=-1))
#
#    sorted_indices = np.argsort(metric)[::-1]
#    #log.info(sorted_indice)
#    SO_order_indices=order_mask.nonzero()[0][sorted_indices]
#    SO_orders=orders[SO_order_indices]
#    return SO_order_indices,SO_orders,sorted_indices,metric

    




###### Particle number estimations not working ######


#################### number of particles estimation not working ##############
def estimate_number_of_particles_old(projection_matrices,radial_points,search_space,average_intensity=False,n_orders=False):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    if projection_matrices[2].ndim>1:
        dimensions = 3
    else:
        dimensions = 2
    max_order = len(projection_matrices)-1
    h_dict = {'dimensions':dimensions,'max_order':max_order,'anti_aliazing_degree':2,'n_phi':False,'n_theta':False}
    cht = HarmonicTransform('complex',h_dict)
    r_grid = GridFactory.construct_grid('uniform',[radial_points,cht.grid_param['thetas'],cht.grid_param['phis']])
    integrator = mLib.SphericalIntegrator(r_grid[:])

    log.info('integrator grod shape = {}'.format(r_grid.shape))
    if isinstance(average_intensity,np.ndarray):
        log.info('use average intensity')
        log.info('a int shape = {}'.format(average_intensity.shape))
        I0 = average_intensity[:,None]*2*np.sqrt(np.pi)
    else:
        log.info('use V 0')
        I0 = projection_matrices[0]
    if (np.sum(I0)<0):
        I0*=-1
        
    scales = np.linspace(*search_space)
    nq = len(radial_points)
    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:projection_matrices[i].shape[-1]]=projection_matrices[i]

    worker = estimate_number_of_particles_worker
    log.info('I0 shape = {}'.format(I0.shape))
    neg_volumes = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[scales], const_args=[I_lm,I0,integrator,cht], callWithMultipleArguments=True,splitMode = 'modulus')
    neg_volumes = neg_volumes/integrator.integrate(np.ones(r_grid[:].shape[:-1]))
    grad = np.gradient(neg_volumes,scales[1]-scales[0])
    n_particles = scales[np.argmax(np.abs(grad))]**2        
    return n_particles,grad,neg_volumes

def estimate_number_of_particles_worker_old(scales,I_lm,I0,integrator,cht,**kwargs):
    return np.array(tuple( integrator.integrate(cht.inverse([I0/s]+I_lm[1:])<0)  for s in scales))

def estimate_number_of_particles(projection_matrices,radial_points,search_space,average_intensity=False,n_orders=False,radial_mask=True):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    if projection_matrices[2].ndim>1:
        dimensions = 3
    else:
        dimensions = 2
    max_order = len(projection_matrices)-1
    h_dict = {'dimensions':dimensions,'max_order':max_order,'anti_aliazing_degree':2,'n_phi':0,'n_theta':0}    
    cht = HarmonicTransform('complex',h_dict)
    r_grid = GridFactory.construct_grid('uniform',[radial_points,cht.grid_param['thetas'],cht.grid_param['phis']])
    integrator = mLib.SphericalIntegrator(r_grid[:])

    log.info('integrator grid shape = {}'.format(r_grid.shape))
    if isinstance(average_intensity,np.ndarray):
        log.info('use average intensity')
        log.info('a int shape = {}'.format(average_intensity.shape))
        I0 = average_intensity[:,None]*2*np.sqrt(np.pi)
    else:
        log.info('use V 0')
        I0 = np.abs(projection_matrices[0])        
    
    scales = np.linspace(*search_space)
    nq = len(radial_points)
    I_lm_zeros = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:projection_matrices[i].shape[-1]]=projection_matrices[i]

    if not isinstance(n_orders,(tuple,list,np.ndarray)):
        n_orders=np.atleast_1d([max_order])
    else:
        n_orders=np.atleast_1d(n_orders)

    Is = [cht.inverse([I0]+I_lm[1:o+1]+I_lm_zeros[o+1:])[radial_mask,...] for o in n_orders]
    Is=np.array(Is)        
    I00y00=(I0.flatten()/(2*np.sqrt(np.pi)))[radial_mask,...]   
    worker = estimate_number_of_particles_worker2
    neg_volumes = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[np.arange(len(Is)),scales], const_args=[Is,I00y00], callWithMultipleArguments=True,splitMode = 'modulus')
    #neg_volumes = neg_volumes/integrator.integrate(np.ones(r_grid[:].shape[:-1]))
    neg_volumes = neg_volumes/np.prod(Is[0].shape)
    grad = np.gradient(neg_volumes,scales[1]-scales[0],axis = 1)
    n_particles = scales[np.argmax(np.abs(grad),axis = 1)]**2
    #log.info('yay')
    return n_particles,grad,neg_volumes

def estimate_number_of_particles_I(I,I00,radial_points,search_space,projection_matrices):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    
    max_order = len(projection_matrices)-1
    nq = len(radial_points)
    h_dict = {'dimensions':3,'max_order':max_order,'anti_aliazing_degree':2,'n_phi':False,'n_theta':False}
    cht = HarmonicTransform('complex',h_dict)
    r_grid = GridFactory.construct_grid('uniform',[radial_points,cht.grid_param['thetas'],cht.grid_param['phis']])

    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:projection_matrices[i].shape[-1]]=projection_matrices[i]
    
    I2 = cht.inverse(I_lm).real
    
    scales = np.linspace(*search_space)
    I00y00=np.abs(I00.flatten()).real/(2*np.sqrt(np.pi))    
    worker = estimate_number_of_particles_worker2
    neg_volumes = estimate_number_of_particles_worker3(I2,scales,I00y00)
    #neg_volumes = neg_volumes/integrator.integrate(np.ones(r_grid[:].shape[:-1]))
    log.info('neg vol dhark = {}'.format(neg_volumes.shape))
    neg_volumes = neg_volumes/np.prod(I.shape)
    grad = np.gradient(neg_volumes,scales[1]-scales[0],axis = 0)
    n_particles = scales[np.argmax(np.abs(grad),axis = 0)]**2
    log.info('yay')
    return n_particles,grad,neg_volumes

def estimate_number_of_particles_worker(I_ids,scales,Is,I00y00,integrator,**kwargs):
    return np.array(tuple(integrator.integrate((Is[i]+(1/s-1)*I00y00[:,None,None])<0) for i,s in zip(I_ids,scales)))

def estimate_number_of_particles_worker2(I_ids,scales,Is,I00y00,**kwargs):
    return np.array(tuple(np.sum((Is[i]+(1/s-1)*I00y00[:,None,None])<0) for i,s in zip(I_ids,scales)))

def estimate_number_of_particles_worker3(I,scales,I00y00,**kwargs):
    return np.sum((I[None,...]+(1/scales[:,None,None,None]-1)*I00y00[None,:,None,None])<0,axis = (1,2,3))




def generate_estimate_number_of_particles_new(rp,I00,grid,N_space):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    radial_mask=rp.radial_mask
    proj_matrices = rp.projection_matrices
    I00 = np.abs(proj_matrices[0].flatten().real)
    I00y00=I00/(2*np.sqrt(np.pi))
    Ns_sqrt = np.linspace(*np.sqrt(N_space[:2]),N_space[2])
    Ns = np.linspace(*N_space)
    summands = (1/Ns_sqrt-1)[:,None]*I00y00[None,radial_mask]
    nsum=np.sum
    n_pixels=np.prod(grid[radial_mask].shape)

    #@measureTime
    def estimate_number_of_particles(I):
        #neg_fraction = integrate(I<0)/vol
        n_particles=False
        scaled_I = I[None,radial_mask,...]+summands[:,:,None,None]
        neg_fractions = nsum(scaled_I<0,axis=(1,2,3))
        grad = np.gradient(neg_fractions)        
        inflection_id = np.argmax(grad)
        n_particles = Ns[inflection_id]        
        return n_particles        
    return estimate_number_of_particles


def estimate_number_of_particles(I,summands,Ns,n_pixels,radial_mask=True):
    scaled_I = I[None,radial_mask,...]+summands[:,:,None,None]
    neg_fractions = np.sum(scaled_I<0,axis=(1,2,3))
    grad = np.gradient(neg_fractions)        
    inflection_id = np.argmax(grad)
    n_particles = Ns[inflection_id]
    new_I = scaled_I[inflection_id].copy()
    del(scaled_I)
    return n_particles,grads,new_I
def generate_number_of_particles_projection(rp,I00,grid,N_space):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    radial_mask=rp.radial_mask
    proj_matrices = rp.projection_matrices
    I00 = np.abs(proj_matrices[0].flatten().real)
    I00y00=I00/(2*np.sqrt(np.pi))
    Ns_sqrt = np.linspace(*np.sqrt(N_space[:2]),N_space[2])
    Ns = np.linspace(*N_space)
    summands = (1/Ns_sqrt-1)[:,None]*I00y00[None,radial_mask]
    nsum=np.sum
    n_pixels=np.prod(grid[radial_mask].shape)

    #@measureTime
    def estimate_number_of_particles(I):
        #neg_fraction = integrate(I<0)/vol
        n_particles=False
        scaled_I = I[None,radial_mask,...]+summands[:,:,None,None]
        neg_fractions = nsum(scaled_I<0,axis=(1,2,3))
        grad = np.gradient(neg_fractions)        
        inflection_id = np.argmax(grad)
        n_particles = Ns[inflection_id]        
        return n_particles        
    return estimate_number_of_particles



def generate_estimate_number_of_particles_new_2(rp,grid,N_space):
    '''
    Routine that estimates the number of particles based on the projection_matrices calculated from the deg2 invariants.
    It assumes that the unknowns are Identity matrices and computes the volume of negative Intensity valus as function of a scaling parameter $s$,
    which is applied to the 0'th intensity harmonic coefficient like $I_(0,0)/s$.
    We realiced that the inflection point in the plot of negative volume over scaling parameter approximates the square root of the number of particles, independent of the actual unknowns.
    :param projection_matrices: projection_matrices
    :type list: list of numpy.ndarray's of length max_order + 1
    :param radial_points: radial component of the recoprocal grid (momentumtransfer values)
    :type numpy.ndarray: dtype float
    :param search_space: [start,stop,number_of_points] for the scaling parameter $s$
    :type list: of length = 3
    :param use_average_intensity: average intensity or bool. If not bool use average intensity instead of 0'th prijection matrix.
    :type bool,numpy.ndarray: 
    :return n_particles: estimated number of particles
    :rtype float: 
    :return grad: Gradient of negative volume over scaling parameter (inflection point is a local extrema here)
    :rtype ndarray: 
    '''
    projection_matrices = rp.projection_matrices
    I00 = np.abs(projection_matrices[0]).real.flatten()
    integrate = mLib.SphericalIntegrator(grid[:]).integrate
    unravel_index = np.unravel_index
    np_min = np.min
    y00 = 1/(2*np.sqrt(np.pi))
    
    vol = integrate(np.ones(grid[:].shape[:-1]))
    find_root = root_scalar
    argsort = np.argsort
    n_angular_values = np.prod(grid[:].shape[1:-1])
    #target_id = int(n_angular_values*target_neg_fraction)
    shape = grid[:].shape[:-1]
    min_I00_q_id = np.argmin(I00.flatten())
    min_I00_scaled=I00[min_I00_q_id]*y00
    I00y00=I00*y00
    nsum = np.sum
    n_total_points = np.prod(shape)
    Ns = np.linspace(np.sqrt(N_space[0]),np.sqrt(N_space[1]),N_space[2])
    N_space_sqrt = [np.sqrt(N_space[0]),np.sqrt(N_space[1]),N_space[2]]
    N_step = Ns[1]-Ns[0]
    gradient = np.gradient
    summands = (1/np.sqrt(Ns[:,None,None,None])-1)*I00y00[None,:,None,None]
    radial_points = grid[:,0,0,0]#rp.radial_points
    radial_mask = rp.radial_mask
    from xframe.presenters.matplotlibPresenter import plot1D
    from xframe.database import analysis as db
    from xframe import settings
    import os

    def calc_gradient(I):
        neg_fractions = nsum((I[None,...]+summands)<0,axis = (1,2,3))/n_total_points 
        grad = gradient(neg_fractions,N_step)        
        return neg_fractions

    #@measureTime
    def estimate_number_of_particles_alt(Ilm):
        #n_particles,grad,neg_volumes = estimate_number_of_particles(projection_matrices,radial_points,N_space_sqrt,average_intensity=False,n_orders=np.array((20,50,99)),I=I)
        n_particles,grad,neg_volumes = estimate_number_of_particles(Ilm,radial_points,N_space_sqrt,average_intensity=False,n_orders=np.array((20,50,99)),radial_mask=radial_mask)        
        layout = {'title':'Gradient of the Volume of negative Intensity \n Number of particles $= {}$ '.format(n_particles),'x_label':'square root of the number of particles $\sqrt{x}$ as in $I_{0,0}/\sqrt{x}$', 'y_label':'Gradient of Volume of negative Intensity','text_size':10}
        fig = plot1D.get_fig(grad,grid = np.linspace(*N_space),x_scale='lin',layout = layout)
        ax = fig.get_axes()[0]
        #ax.vlines(np.sqrt(n_particles),grads.min(),grads.max())
        path = db.get_path('reciprocal_proj_data',path_modifiers={'name':settings.analysis.name})
        new_path = os.path.dirname(path)+'/' + os.path.basename(path).split('.')[0] + '_n_particles.matplotlib'
        db.save(new_path,fig,dpi=400)
        return n_particles

    def estimate_number_of_particles_(Ilm):
        try:
            n2 = estimate_number_of_particles_alt(Ilm)
        except Exception as e:
            traceback.print_exc()
            log.error(e)
        
    return estimate_number_of_particles_
