import numpy as np
import numpy.ma as mp
import scipy.integrate as spIntegrate
from scipy.special import jv as bessel_jv
from scipy.special import roots_legendre
import logging
from itertools import repeat

log=logging.getLogger('root')

from xframe.library.mathLibrary import eval_ND_zernike_polynomials
from xframe.library.mathLibrary import bessel_jnu
from xframe.library.mathLibrary import spherical_bessel_jnu
from xframe.library.mathLibrary import polar_spherical_dft_reciprocity_relation_radial_cutoffs
from xframe import Multiprocessing
#######################################################
###    polar/spherical HT via Zernike expansion     ###

def generate_weightDict_zernike(max_order, n_radial_points,expansion_limit = -1,pi_in_q = True,dimensions=3,n_cpus=False):
    '''
    Calculates the weights for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
    Strangly the last sentence does not seem to be right, the transform gets better for higher limits thats why, quite arbitrarily, the default below corresponds to 2*(2*n_radial_points-1)
    '''
    orders = np.arange(max_order+1)
    assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)
    
    if expansion_limit < 0:
        expansion_limit = 2*(2*n_radial_points - 1)
        log.info('Default zernike expansion limit = {}'.format(expansion_limit))
    expansion_limit = max(expansion_limit,max_order)

    worker_by_dimensions= {3:calc_spherical_zernike_weights,2:calc_polar_zernike_weights}
    worker = worker_by_dimensions[dimensions]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,argArrays=[orders],const_args=[n_radial_points,expansion_limit,pi_in_q],callWithMultipleArguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,argArrays=[orders],const_args=[n_radial_points,expansion_limit,pi_in_q],callWithMultipleArguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'Zerinike_{}D'.format(dimensions)
    return weightDict


def generate_weightDict_trapz(max_order, n_radial_points,pi_in_q = True,dimensions=3,n_cpus=False):
    '''
    Calculates the weights for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
    Strangly the last sentence does not seem to be right, the transform gets better for higher limits thats why, quite arbitrarily, the default below corresponds to 2*(2*n_radial_points-1)
    '''
    orders = np.arange(max_order+1)
    assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)
    

    worker_by_dimensions= {2:calc_polar_trapz_weights,3:calc_spherical_trapz_weights}
    worker = worker_by_dimensions[dimensions]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,argArrays=[orders],const_args=[n_radial_points,pi_in_q],callWithMultipleArguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,argArrays=[orders],const_args=[n_radial_points,pi_in_q],callWithMultipleArguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'Zerinike_{}D'.format(dimensions)
    return weightDict

# this is equivalent to trapz for expansion_limit -> infinity -.- so no good
def calc_spherical_zernike_weights(orders,n_radial_points,expansion_limit,pi_in_q,**kwargs):
    '''
    Calculates the weights $w_{l,p,k}$ for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
    $l$ indexes the order, $p$ the new_radial_coordinate and  $k$ the summation radial coordinate.
    '''    
    radial_points=n_radial_points    
    #radial_points=opt['n_radial_points']+1
    
    ps=np.arange(1,radial_points)
    n_p = radial_points-1 
    ks=np.arange(radial_points)
    if pi_in_q:
        j_factor=np.pi        
    else:
        j_factor=1/2
    #    log.info('ps={}'.format(ps/2))
    n_k = radial_points
    #weights=np.zeros((len(orders),n_p,n_k))
    #zernike_dict=eval_ND_zernike_polynomials(orders,expansion_limit,ks/radial_points,3)

    def weights_for_fixed_order(l):
        zernike_dict=eval_ND_zernike_polynomials(np.array([l]),expansion_limit,ps/radial_points,3)
        #log.info('finished_zernike dict')
        s=np.arange(l,expansion_limit+1,2)
        len_s=len(s)
        prefactor=(-1)**( (s-l)/2 )*(2*s+3)
        jp=spherical_bessel_jnu(np.repeat((s+1)[:,None],radial_points-1,axis=1),ks[1:]*j_factor)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        log.info('making summands for order = {}'.format(l))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        if l == 0:
            summands[0,:,0] = j_factor
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,1:]=np.square(ps)[:,None]/(ks[None,1:])
    c_kp[:,0]=np.square(ps)
    weights=weights*c_kp[None,:,:]
    return weights

def calc_spherical_trapz_weights(orders,n_radial_points,pi_in_q,**kwargs):
    '''
    Generates weights for trapezooidal rule
    '''
    N = n_radial_points
    ps=np.arange(1,N)
    ks=np.arange(N)
    if pi_in_q:
        j_factor=np.pi        
    else:
        j_factor=1/2

    ls = orders
    jmpk = spherical_bessel_jnu(np.repeat(np.repeat(ls[:,None,None],N-1,axis=1),N,axis=2),ks[None,:]*ps[:,None]*j_factor/N)
    weights = ps[None,:,None]**2*jmpk
    return weights


# this is equivalent to trapz for expansion_limit -> infinity -.- so no good
def calc_polar_zernike_weights(orders,n_radial_points,expansion_limit,pi_in_q,**kwargs):
    '''
     Calculates the weights $w_{l,p,k}$ for the Zernike version of the approximated hankel transform.
     There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
     $l$ indexes the order, $p$ the new_radial_coordinate and  $k$ the summation radial coordinate.
    '''    

    radial_points=n_radial_points    
    #radial_points=opt['n_radial_points']+1
    
    ps=np.arange(1,radial_points)
    n_p = radial_points-1 
    ks=np.arange(radial_points)
    if pi_in_q:
        J_factor=np.pi        
    else:
        J_factor=1/2
    #    log.info('ps={}'.format(ps/2))
    n_k = radial_points
    #weights=np.zeros((len(orders),n_p,n_k))
    #zernike_dict=eval_ND_zernike_polynomials(orders,expansion_limit,ks/radial_points,3)

    def weights_for_fixed_order(m):
        zernike_dict=eval_ND_zernike_polynomials(np.array([m]),expansion_limit,ps/radial_points,2)
        #log.info('finished_zernike dict')
        s=np.arange(m,expansion_limit+1,2)
        len_s=len(s)
        prefactor=(-1)**( (s-m)/2 )*(2*s+2)
        Jk = bessel_jnu(np.repeat((s+1)[:,None],radial_points-1,axis=1),ks[1:]*J_factor)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zp=zernike_dict[m]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        log.info('making summands for order = {}'.format(m))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,1:] = prefactor[:,None,None]*Zp[:,:,None]*Jk[:,None,:]
        if m == 0:
            summands[0,:,0] = J_factor
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,1:]=ps[:,None]/(ks[None,1:])
    c_kp[:,0]=ps
    weights=weights*c_kp[None,:,:]
    return weights

def calc_polar_trapz_weights(orders,n_radial_points,pi_in_q,**kwargs):
    '''
    Generates weights for trapezoidal rule.
    '''
    N = n_radial_points
    ps=np.arange(1,N)
    ks=np.arange(N)
    if pi_in_q:
        J_factor=np.pi**2        
    else:
        J_factor=1/2**2
        #    log.info('ps={}'.format(ps/2))
    ms = orders
    Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N-1,axis=1),N,axis=2),ks[None,:]*ps[:,None]*np.pi/N)
    weights = ps[None,:,None]*Jmpk
    return weights



def assemble_weights_zernike(weights,orders,r_max,pi_in_q,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,pi_in_q = pi_in_q)

    if dimensions == 3:
        forward_prefactor = (-1.j)**(orders[None,None,:])*(np.pi**2/q_max**3)*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(np.pi**2/r_max**3)*np.sqrt(2/np.pi)
    elif dimensions == 2:
        all_orders = np.concatenate((orders,orders[:0:-1]))
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(np.pi/q_max**2)
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(np.pi/r_max**2)
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)   
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}

def assemble_weights_trapz(weights,orders,r_max,pi_in_q,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,pi_in_q = pi_in_q)

    if dimensions == 2:
        all_orders = np.concatenate((orders,orders[:0:-1]))
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(np.pi/q_max)**2
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(np.pi/r_max)**2
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)
    elif dimensions == 3:
        forward_prefactor = (-1.j)**(orders[None,None,:])*(np.pi/q_max)**3*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(np.pi/r_max)**3*np.sqrt(2/np.pi)
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}


def generate_ht(weights,orders,r_max,pi_in_q=False,dimensions=3,use_gpu=False,mode = 'trapz'):
    '''
    Routine that selects the construciton routine for 2D/3D Hankel transforms via zernike expansion. 
    '''
    if mode == 'trapz':
        w = assemble_weights_trapz(weights,orders,r_max,pi_in_q,dimensions=dimensions)
    elif mode == 'zernike':
        w = assemble_weights_zernike(weights,orders,r_max,pi_in_q,dimensions=dimensions)

    spherical_gpu_version = (dimensions == 3) and use_gpu
    spherical_version = (dimensions == 3) and (not use_gpu)
    polar_version = (dimensions == 2)
    if spherical_gpu_version:
        zht,izht = generate_spherical_zernike_ht_gpu(w,orders,r_max,pi_in_q=pi_in_q)
    elif spherical_version:
        zht,izht = generate_spherical_zernike_ht(w,orders,r_max,pi_in_q=pi_in_q)
    elif polar_version:
        zht,izht = generate_polar_zernike_ht(w,orders,r_max,pi_in_q=pi_in_q)
    return zht,izht

def generate_polar_zernike_ht(w,pos_orders,r_max,pi_in_q=False):
    '''
    Generates polar hankel transform using Zernike weights. $HT_m(f_m) = \sum_k f_m(k)*w_kpm$ for $m\geq 0$.
    '''
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    n_radial_steps = forward_weights.shape[1]
    n_orders = len(pos_orders)
    full_shape = (n_radial_steps,2*n_orders-1)
    out_forward = np.zeros(full_shape,dtype = complex)
    out_inverse = np.zeros(full_shape,dtype = complex)
    def zht(harmonic_coeff):
        reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[1:,None,:],axis = 0,out=out_forward)
        return reciprocal_harmonic_coeff
    def izht(reciprocal_coeff):
        harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[1:,None,:],axis = 0,out=out_inverse)
        return harmonic_coeff
    return zht,izht

def generate_spherical_zernike_ht(w,l_orders,r_max,pi_in_q=False):
    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_gpu(w,l_orders,r_max,pi_in_q = False):
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    n_radial_points = forward_weights.shape[1]

    l_max=l_orders.max()

    nq = n_radial_points
    nl = len(l_orders)
    nlm = l_max*(l_max+2)+1

    kernel_str = """
    __kernel void
    apply_weights(__global double2* out, 
    __global double2* w, 
    __global double2* rho, 
    long nq,long nlm, long nl)
    {
  
    long i = get_global_id(0); 
    long j = get_global_id(1);
    long l = (long) sqrt((double)j);

 
    // value stores the element that is 
    // computed by the thread
    double2 value = 0;
    // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
    for (int q = 0; q < nq-1; ++q)
    {
    double2 wqql = w[q*nq*nl + i*nl + l];
    double2 rqlm = rho[(q+1)*nlm + j];
    value.x += wqql.x * rqlm.x - wqql.y * rqlm.y;
    value.y += wqql.x * rqlm.y + wqql.y * rqlm.x;
    }
    
    // Write the matrix to device memory each 
    // thread writes one element
    out[i * nlm + j] = value;//w[nq*nl+i*nl + l];
    }
    """

    kernel_dict_forward={
            'kernel': kernel_str,
            'name': 'forward_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [np.complex,np.complex,np.complex,np.int64,np.int64,np.int64],
                'shapes' : [(nq,nlm),forward_weights.shape,(nq,nlm),None,None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input','const_input'],
                'const_inputs' : [None,forward_weights,None,np.long(nq),np.long(nlm),np.long(nl)],
                'global_range' : (nq,nlm),
                'local_range' : None
            }]
        }
    
    kernel_dict_inverse={
            'kernel': kernel_str,
            'name': 'inverse_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [np.complex,np.complex,np.complex,np.int64,np.int64,np.int64],
                'shapes' : [(nq,nlm),inverse_weights.shape,(nq,nlm),None,None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input','const_input'],
                'const_inputs' : [None,inverse_weights,None,np.long(nq),np.long(nlm),np.long(nl)],
                'global_range' : (nq,nlm),
                'local_range' : None
            }]
        }

    forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
    inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

    zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process)
    inverse_zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process)
    return zernike_hankel_transform,inverse_zernike_hankel_transform


