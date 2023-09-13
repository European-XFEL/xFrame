import numpy as np
import numpy.ma as mp
import logging
from itertools import repeat

log=logging.getLogger('root')

from xframe.library.mathLibrary import eval_ND_zernike_polynomials
from xframe.library.mathLibrary import bessel_jnu
from xframe.library.mathLibrary import spherical_bessel_jnu
from xframe.library.mathLibrary import polar_spherical_dft_reciprocity_relation_radial_cutoffs
from xframe.library.mathLibrary import gauss_legendre
from xframe import Multiprocessing

ht_modes = ['trapz','Zernike','midpoint','gauss',]

###legacy imports###
from xframe.library.pythonLibrary import plugVariableIntoFunction
from xframe.library.mathLibrary import spherical_bessel_jnu
import scipy.integrate as spIntegrate

def generate_weightDict(max_order, n_radial_points,reciprocity_coefficient = np.pi,dimensions=3,n_cpus=False,mode=ht_modes[0],**kwargs):
    if mode == ht_modes[0]:
        wd = generate_weightDict_trapz(max_order, n_radial_points,reciprocity_coefficient,dimensions=dimensions,n_cpus=n_cpus,**kwargs)
    elif mode == ht_modes[1]:
        wd = generate_weightDict_zernike(max_order, n_radial_points,reciprocity_coefficient,dimensions=dimensions,n_cpus=n_cpus,**kwargs)
    elif mode == ht_modes[2]:
        wd = generate_weightDict_mid(max_order, n_radial_points,reciprocity_coefficient,dimensions=dimensions,n_cpus=n_cpus,**kwargs)
    elif mode == ht_modes[3]:
        wd = generate_weightDict_gauss(max_order, n_radial_points,reciprocity_coefficient,dimensions=dimensions,n_cpus=n_cpus,**kwargs)
    else:
        raise AssertionError("Hankel transform mode {} not known. Known modes are {}".format(mode,ht_modes))
    wd['mode']=mode
    return wd
def assemble_weights(weights,pos_orders,r_max,reciprocity_coefficient=np.pi,dimensions=3,mode = ht_modes[0]):
    if mode == ht_modes[0]:
        pos_orders = np.arange(weights.shape[0])
        w = assemble_weights_trapz(weights,pos_orders,r_max,reciprocity_coefficient,dimensions=dimensions)
    elif mode == ht_modes[1]:
        w = assemble_weights_zernike(weights,pos_orders,r_max,reciprocity_coefficient,dimensions=dimensions)
    elif mode == ht_modes[2]:
        w = assemble_weights_mid(weights,pos_orders,r_max,reciprocity_coefficient,dimensions=dimensions)
    elif mode == ht_modes[3]:
        w = assemble_weights_gauss(weights,pos_orders,r_max,reciprocity_coefficient,dimensions=dimensions)
    else:
        raise AssertionError("Hankel transform mode {} not known. Known modes are {}".format(mode,ht_modes))
    w['mode']=mode
    return w

###############################################################
###    polar/spherical HT weights via Zernike expansion     ###
def generate_weightDict_zernike(max_order, n_radial_points,expansion_limit = -1,reciprocity_coefficient = np.pi,dimensions=3,n_cpus=False,approximation_type = 'trapz'):
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

    worker_by_dimensions= {
        3:{
            'trapz':calc_spherical_zernike_weights,
            'midpoint':calc_spherical_zernike_weights_midpoint
        },
        2:{
            'trapz':calc_polar_zernike_weights,
            'midpoint':calc_polar_zernike_weights_midpoint
        }
    }
    worker = worker_by_dimensions[dimensions][approximation_type]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,expansion_limit,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,expansion_limit,reciprocity_coefficient],call_with_multiple_arguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'Zerinike_{}D_{}'.format(dimensions,approximation_type)
    return weightDict

# this is equivalent to trapz for expansion_limit -> infinity -.- so no good
def calc_spherical_zernike_weights(orders,n_radial_points,expansion_limit,reciprocity_coefficient,**kwargs):
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
        jp=spherical_bessel_jnu(np.repeat((s+1)[:,None],radial_points-1,axis=1),ks[1:]*reciprocity_coefficient)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        #log.info('making summands for order = {}'.format(l))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        if l == 0:
            summands[0,:,0] = reciprocity_coefficient ## this might be trouble ? previousely it was pi or 1/2
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,1:]=np.square(ps)[:,None]/(ks[None,1:])
    c_kp[:,0]=np.square(ps)
    weights=weights*c_kp[None,:,:]
    return weights

# this is equivalent to trapz for expansion_limit -> infinity -.- so no good
def calc_polar_zernike_weights(orders,n_radial_points,expansion_limit,reciprocity_coefficient,**kwargs):
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
        Jk = bessel_jnu(np.repeat((s+1)[:,None],radial_points-1,axis=1),ks[1:]*reciprocity_coefficient)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zp=zernike_dict[m]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        log.info('making summands for order = {}'.format(m))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,1:] = prefactor[:,None,None]*Zp[:,:,None]*Jk[:,None,:]
        if m == 0:
            summands[0,:,0] = reciprocity_coefficient
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,1:]=ps[:,None]/(ks[None,1:])
    c_kp[:,0]=ps
    weights=weights*c_kp[None,:,:]
    return weights


# this is equivalent to midpoint for expansion_limit -> infinity -.- so no good
def calc_spherical_zernike_weights_midpoint(orders,n_radial_points,expansion_limit,reciprocity_coefficient,**kwargs):
    '''
    Calculates the weights $w_{l,p,k}$ for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
    $l$ indexes the order, $p$ the new_radial_coordinate and  $k$ the summation radial coordinate.
    '''    
    radial_points=n_radial_points    
    #radial_points=opt['n_radial_points']+1
    
    ps=np.arange(radial_points)+1/2
    n_p = radial_points
    ks=np.arange(radial_points)+1/2
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
        jp=spherical_bessel_jnu(np.repeat((s+1)[:,None],radial_points,axis=1),ks*reciprocity_coefficient)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        #log.info('making summands for order = {}'.format(l))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        #if l == 0:
        #    summands[0,:,0] = reciprocity_coefficient ## this might be trouble ? previousely it was pi or 1/2
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,:]=np.square(ps)[:,None]/(ks[None,:])
    #c_kp[:,0]=np.square(ps)
    weights=weights*c_kp[None,:,:]
    return weights

# this is equivalent to midpoint for expansion_limit -> infinity -.- so no good
def calc_polar_zernike_weights_midpoint(orders,n_radial_points,expansion_limit,reciprocity_coefficient,**kwargs):
    '''
     Calculates the weights $w_{l,p,k}$ for the Zernike version of the approximated hankel transform.
     There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
     $l$ indexes the order, $p$ the new_radial_coordinate and  $k$ the summation radial coordinate.
    '''    

    radial_points=n_radial_points    
    #radial_points=opt['n_radial_points']+1
    
    ps=np.arange(radial_points)+1/2
    n_p = radial_points
    ks=np.arange(radial_points)+1/2
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
        Jk = bessel_jnu(np.repeat((s+1)[:,None],radial_points,axis=1),ks*reciprocity_coefficient)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zp=zernike_dict[m]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        log.info('making summands for order = {}'.format(m))
        summands = np.zeros((len_s,n_p,n_k))
        summands[:,:,:] = prefactor[:,None,None]*Zp[:,:,None]*Jk[:,None,:]
        #if m == 0:
        #    summands[0,:,0] = reciprocity_coefficient
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_p,n_k))
    c_kp[:,:]=ps[:,None]/(ks[None,:])
    #c_kp[:,0]=ps
    weights=weights*c_kp[None,:,:]
    return weights

def assemble_weights_zernike(weights,orders,r_max,reciprocity_coefficient,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient=reciprocity_coefficient)

    if dimensions == 3:
        #forward_prefactor = (-1.j)**(orders[None,None,:])*(np.pi**2/q_max**3)*np.sqrt(2/np.pi)
        forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/n_radial_points)**3*np.sqrt(2/np.pi**3)
        #inverse_prefactor = (1.j)**(orders[None,None,:])*(np.pi**2/r_max**3)*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/n_radial_points)**3*np.sqrt(2/np.pi**3)
    elif dimensions == 2:
        all_orders = np.concatenate((orders,orders[:0:-1]))
        #forward_prefactor = (-1.j)**(all_orders[None,None,:])*(np.pi/q_max**2)
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/n_radial_points)**2/np.pi
        #inverse_prefactor = (1.j)**(all_orders[None,None,:])*(np.pi/r_max**2)
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/n_radial_points)**2/np.pi
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)   
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}


#############################################################
###    polar/spherical HT weights via trapezoidal rule    ###
def generate_weightDict_trapz(max_order, n_radial_points,reciprocity_coefficient = np.pi,dimensions=3,n_cpus=False):
    '''    
    '''
    orders = np.arange(max_order+1)
    #assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)
    

    worker_by_dimensions= {2:calc_polar_trapz_weights,3:calc_spherical_trapz_weights}
    worker = worker_by_dimensions[dimensions]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'trapz'
    weightDict['dimension'] = dimensions
    return weightDict

def calc_spherical_trapz_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for trapezooidal rule
    '''
    N = n_radial_points
    ps=np.arange(1,N)
    ks=np.arange(N)

    ls = orders
    jmpk = spherical_bessel_jnu(np.repeat(np.repeat(ls[:,None,None],N-1,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
    weights = ps[None,:,None]**2*jmpk
    return weights

def calc_polar_trapz_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for trapezoidal rule.
    '''
    N = n_radial_points
    ps=np.arange(1,N)
    ks=np.arange(N)
    ms = orders
    Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N-1,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
    zero_order_mask = (ms == 0)
    #Jmpk[zero_order_mask]*=np.sqrt(2)
    weights = ps[None,:,None]*Jmpk
    return weights

def assemble_weights_trapz(weights,orders,r_max,reciprocity_coefficient,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient = reciprocity_coefficient)
    #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
    
    if dimensions == 2:
        all_orders = np.concatenate((orders,-orders[:0:-1]))
        #log.info(all_orders)
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/n_radial_points)**2#*np.sqrt(2)
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/n_radial_points)**2#*np.sqrt(2)
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)
    elif dimensions == 3:
        forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/n_radial_points)**3*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/n_radial_points)**3*np.sqrt(2/np.pi)
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}


##########################################################
###    polar/spherical HT weights via midpoint rule    ###
def generate_weightDict_mid(max_order, n_radial_points,reciprocity_coefficient = np.pi,dimensions=3,n_cpus=False):
    '''    
    '''
    orders = np.arange(max_order+1)
    #assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)
    
    worker_by_dimensions= {2:calc_polar_mid_weights,3:calc_spherical_mid_weights}
    worker = worker_by_dimensions[dimensions]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'trapz'
    weightDict['dimension'] = dimensions
    return weightDict

def calc_spherical_mid_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for trapezooidal rule
    '''
    N = n_radial_points
    ps=np.arange(N)+1/2
    ks=np.arange(N)+1/2

    ls = orders
    jmpk = spherical_bessel_jnu(np.repeat(np.repeat(ls[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
    weights = ps[None,:,None]**2*jmpk
    return weights

def calc_polar_mid_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for trapezoidal rule.
    '''
    N = n_radial_points
    ps=np.arange(N)+1/2
    ks=np.arange(N)+1/2
    ms = orders
    Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient/N)
    #zero_order_mask = (ms == 0)
    #Jmpk[zero_order_mask]*=np.sqrt(2)
    weights = ps[None,:,None]*Jmpk
    return weights

def assemble_weights_mid(weights,orders,r_max,reciprocity_coefficient,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient = reciprocity_coefficient)
    #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
    
    if dimensions == 2:
        all_orders = np.concatenate((orders,-orders[:0:-1]))
        #log.info(all_orders)
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/n_radial_points)**2#*np.sqrt(2)
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/n_radial_points)**2#*np.sqrt(2)
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)
    elif dimensions == 3:
        forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/n_radial_points)**3*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/n_radial_points)**3*np.sqrt(2/np.pi)
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}


######################################################################
###    polar/spherical HT weights via Gauss Legendre quadrature    ###
def generate_weightDict_gauss(max_order, n_radial_points,reciprocity_coefficient = np.pi,dimensions=3,n_cpus=False):
    '''    
    '''
    orders = np.arange(max_order+1)
    #assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)
    

    worker_by_dimensions= {2:calc_polar_gauss_weights,3:calc_spherical_gauss_weights}
    worker = worker_by_dimensions[dimensions]
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(worker,input_arrays=[orders],const_inputs=[n_radial_points,reciprocity_coefficient],call_with_multiple_arguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'trapz'
    weightDict['dimension'] = dimensions
    return weightDict

def calc_spherical_gauss_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for gauss legendre quadrature.
    '''
    N = n_radial_points
    xi,wgauss = gauss_legendre(N)
    
    ps=xi+1
    ks=xi+1

    ls = orders
    jmpk = spherical_bessel_jnu(np.repeat(np.repeat(ls[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*N/4)
    weights = ps[None,:,None]**2*jmpk*wgauss[None,:,None]
    return weights

def calc_polar_gauss_weights(orders,n_radial_points,reciprocity_coefficient,**kwargs):
    '''
    Generates weights for gauss legendre quadrature.
    '''

    N = n_radial_points
    xi,wgauss = gauss_legendre(N)
    
    ps=xi+1
    ks=xi+1
    ms = orders
    Jmpk = bessel_jnu(np.repeat(np.repeat(ms[:,None,None],N,axis=1),N,axis=2),ks[None,:]*ps[:,None]*reciprocity_coefficient*N/4)
    #zero_order_mask = (ms == 0)
    #Jmpk[zero_order_mask]*=np.sqrt(2)
    weights = ps[None,:,None]*Jmpk*wgauss[None,:,None]
    return weights

def assemble_weights_gauss(weights,orders,r_max,reciprocity_coefficient,dimensions=3):
    '''
    Generates weights for the forward and inverse transform by multiplieing with the propper constants.
    And changes the Array order of the weights from (order,summation_radial_corrdinate,new_radial_coordinate) to 
    (summation_radial_coordinate,new_radial_coordinate,order)
    '''
    n_radial_points = weights.shape[-1]
    q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs(r_max,n_radial_points,reciprocity_coefficient = reciprocity_coefficient)
    #q_max=polar_spherical_dft_reciprocity_relation_radial_cutoffs_new(r_max,n_radial_points,scale = pi_in_q)
    
    if dimensions == 2:
        all_orders = np.concatenate((orders,-orders[:0:-1]))
        #log.info(all_orders)
        forward_prefactor = (-1.j)**(all_orders[None,None,:])*(r_max/2)**2#*np.sqrt(2)
        inverse_prefactor = (1.j)**(all_orders[None,None,:])*(q_max/2)**2#*np.sqrt(2)
        # weights for negative orders are given by $w_{-mpk}=(-1)^m $w_{mpk}$ due to $J_{-m}(x)=(-1)^m J_m(x)$.
        #log.info('wights.shape={},orders = {}'.format(weights.shape,orders))
        weights = np.concatenate((weights,(-1)**orders[:0:-1,None,None]*weights[:0:-1]),axis = 0)
    elif dimensions == 3:
        forward_prefactor = (-1.j)**(orders[None,None,:])*(r_max/2)**3*np.sqrt(2/np.pi)
        inverse_prefactor = (1.j)**(orders[None,None,:])*(q_max/2)**3*np.sqrt(2/np.pi)
        
    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    #log.info('weights shape = {}'.format(forward_weights.shape))
    return {'forward':forward_weights,'inverse':inverse_weights}


############################################################
###     create polar/spherical HT for given weights      ###
def generate_ht(weights,used_orders,r_max,reciprocity_coefficient=np.pi,dimensions=3,use_gpu=False,mode = ht_modes[0]):
    '''
    Routine that selects the construciton routine for 2D/3D Hankel transforms via zernike expansion. 
    '''
    w = assemble_weights(weights,used_orders,r_max,reciprocity_coefficient,dimensions=dimensions,mode=mode)

    spherical_gpu_version = (dimensions == 3) and use_gpu
    spherical_version = (dimensions == 3) and (not use_gpu)
    polar_version = (dimensions == 2)  and (not use_gpu)
    polar_gpu_version = (dimensions == 2) and use_gpu
    
    if spherical_gpu_version:
        zht,izht = generate_spherical_ht_gpu(w,used_orders)
    elif spherical_version:
        zht,izht = generate_spherical_ht(w,used_orders)
    elif polar_version:
        zht,izht = generate_polar_ht(w,used_orders)
    elif polar_gpu_version:
        zht,izht = generate_polar_ht_gpu(w,used_orders)
    return zht,izht

def generate_polar_zernike_ht(w,used_orders):
    '''
    Generates polar hankel transform using Zernike weights. $HT_m(f_m) = \sum_k f_m(k)*w_kpm$ for $m\geq 0$.
    '''
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    #log.info('weight shape = {}'.format(forward_weights.shape))
    n_orders = int((forward_weights.shape[-1]+1)//2)
    n_radial_steps = forward_weights.shape[1]
    all_orders = np.concatenate((np.arange(n_orders),np.arange(n_orders)[:0:-1]))
    
    unused_order_mask = ~np.in1d(all_orders,used_orders)
    
    full_shape = (n_radial_steps,len(all_orders))
    out_forward = np.zeros(full_shape,dtype = complex)
    out_inverse = np.zeros(full_shape,dtype = complex)
    def zht(harmonic_coeff):
        #log.info('harmonic shape = {}'.format(harmonic_coeff.shape))
        #log.info('out forward = {}'.format(out_forward.shape))
        reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[1:,None,:],axis = 0,out=out_forward)
        reciprocal_harmonic_coeff[:,unused_order_mask]=0
        return reciprocal_harmonic_coeff
    def izht(reciprocal_coeff):
        harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[1:,None,:],axis = 0,out=out_inverse)
        harmonic_coeff[:,unused_order_mask]=0
        return harmonic_coeff
    return zht,izht

def generate_spherical_zernike_ht(w,l_orders):
    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht


def generate_polar_ht(w,used_orders):
    '''
    Generates polar hankel transform using Zernike weights. $HT_m(f_m) = \sum_k f_m(k)*w_kpm$ for $m\geq 0$.
    '''
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    #log.info('weight shape = {}'.format(forward_weights.shape))
    n_orders = int((forward_weights.shape[-1]+1)//2)
    n_radial_steps = forward_weights.shape[1]
    all_orders = np.concatenate((np.arange(n_orders),np.arange(n_orders)[:0:-1]))
    
    unused_order_mask = ~np.in1d(all_orders,used_orders)
    
    full_shape = (n_radial_steps,len(all_orders))
    out_forward = np.zeros(full_shape,dtype = complex)
    out_inverse = np.zeros(full_shape,dtype = complex)
    if w['mode'] in ht_modes[:2]:
        def zht(harmonic_coeff):
            #log.info('harmonic shape = {}'.format(harmonic_coeff.shape))
            #log.info('out forward = {}'.format(out_forward.shape))
            reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[1:,None,:],axis = 0,out=out_forward)
            reciprocal_harmonic_coeff[:,unused_order_mask]=0
            return reciprocal_harmonic_coeff
        def izht(reciprocal_coeff):
            harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[1:,None,:],axis = 0,out=out_inverse)
            harmonic_coeff[:,unused_order_mask]=0
            return harmonic_coeff
    else:
        def zht(harmonic_coeff):
            #log.info('harmonic shape = {}'.format(harmonic_coeff.shape))
            #log.info('out forward = {}'.format(out_forward.shape))
            reciprocal_harmonic_coeff = np.sum(forward_weights*harmonic_coeff[:,None,:],axis = 0,out=out_forward)
            reciprocal_harmonic_coeff[:,unused_order_mask]=0
            return reciprocal_harmonic_coeff
        def izht(reciprocal_coeff):
            harmonic_coeff = np.sum(inverse_weights*reciprocal_coeff[:,None,:],axis = 0,out=out_inverse)
            harmonic_coeff[:,unused_order_mask]=0
            return harmonic_coeff
    return zht,izht

def generate_spherical_ht(w,l_orders):
    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    if w['mode'] in ht_modes[:2]:
        def zht(harmonic_coeff):
            return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
        def izht(reciprocal_coeff):
            return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    else:
        def zht(harmonic_coeff):
            return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
        def izht(reciprocal_coeff):
            return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_ht_gpu(w,l_orders):
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    n_radial_points = forward_weights.shape[1]

    l_max=l_orders.max()

    nq = n_radial_points
    nl = len(l_orders)
    nlm = l_max*(l_max+2)+1
    if w['mode'] in ht_modes[:2]:        
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
    else:
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
        for (int q = 0; q < nq; ++q)
        {
        double2 wqql = w[q*nq*nl + i*nl + l];
        double2 rqlm = rho[q*nlm + j];
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
            'functions': ({
                'name': 'apply_weights',
                'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                'shapes' : ((nq,nlm),forward_weights.shape,(nq,nlm),None,None,None),
                'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                'const_inputs' : (None,forward_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)),
                'global_range' : (nq,nlm),
                'local_range' : None
            },)
        }
    
    kernel_dict_inverse={
            'kernel': kernel_str,
            'name': 'inverse_hankel',
            'functions': ({
                'name': 'apply_weights',
                'dtypes' : (complex,complex,complex,np.int64,np.int64,np.int64),
                'shapes' : ((nq,nlm),inverse_weights.shape,(nq,nlm),None,None,None),
                'arg_roles' : ('output','const_input','input','const_input','const_input','const_input'),
                'const_inputs' : (None,inverse_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)),
                'global_range' : (nq,nlm),
                'local_range' : None
            },)
        }
    
    forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
    inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

    zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process)
    inverse_zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process)
    return zernike_hankel_transform,inverse_zernike_hankel_transform

def generate_polar_ht_gpu(w,used_orders):
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    n_radial_points = forward_weights.shape[1]

    m_max=used_orders.max()

    nq = n_radial_points
    nm = m_max*2+1

    if w['mode'] in ht_modes[:2]:
        kernel_str = """
        __kernel void
        apply_weights(__global double2* out, 
        __global double2* w, 
        __global double2* rho, 
        long nq,long nm)
        {
  
        long i = get_global_id(0); 
        long j = get_global_id(1);
 
        // value stores the element that is 
        // computed by the thread
        double2 value = 0;
        // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
        for (int q = 0; q < nq-1; ++q)
        {
        double2 wqqm = w[q*nq*nm + i*nm + j];
        double2 rqm = rho[(q+1)*nm + j];
        value.x += wqqm.x * rqm.x - wqqm.y * rqm.y;
        value.y += wqqm.x * rqm.y + wqqm.y * rqm.x;
        }
    
        // Write the matrix to device memory each 
        // thread writes one element
        out[i * nm + j] = value;//w[nq*nl+i*nl + l];
        }
        """
    else:
        kernel_str = """
        __kernel void
        apply_weights(__global double2* out, 
        __global double2* w, 
        __global double2* rho, 
        long nq,long nm)
        {
  
        long i = get_global_id(0); 
        long j = get_global_id(1);
 
        // value stores the element that is 
        // computed by the thread
        double2 value = 0;
        // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
        for (int q = 0; q < nq; ++q)
        {
        double2 wqqm = w[q*nq*nm + i*nm + j];
        double2 rqm = rho[q*nm + j];
        value.x += wqqm.x * rqm.x - wqqm.y * rqm.y;
        value.y += wqqm.x * rqm.y + wqqm.y * rqm.x;
        }
    
        // Write the matrix to device memory each 
        // thread writes one element
        out[i * nm + j] = value;//w[nq*nl+i*nl + l];
        }
        """

    kernel_dict_forward={
            'kernel': kernel_str,
            'name': 'forward_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [complex,complex,complex,np.int64,np.int64],
                'shapes' : [(nq,nm),forward_weights.shape,(nq,nm),None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input'],
                'const_inputs' : [None,forward_weights,None,np.int64(nq),np.int64(nm)],
                'global_range' : (nq,nm),
                'local_range' : None
            }]
        }
    
    kernel_dict_inverse={
            'kernel': kernel_str,
            'name': 'inverse_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [complex,complex,complex,np.int64,np.int64],
                'shapes' : [(nq,nm),inverse_weights.shape,(nq,nm),None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input'],
                'const_inputs' : [None,inverse_weights,None,np.int64(nq),np.int64(nm)],
                'global_range' : (nq,nm),
                'local_range' : None
            }]
        }

    forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
    inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

    zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process)
    inverse_zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process)
    return zernike_hankel_transform,inverse_zernike_hankel_transform



def generate_spherical_zernike_ht_gpu(w,l_orders):
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
                'dtypes' : [complex,complex,complex,np.int64,np.int64,np.int64],
                'shapes' : [(nq,nlm),forward_weights.shape,(nq,nlm),None,None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input','const_input'],
                'const_inputs' : [None,forward_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)],
                'global_range' : (nq,nlm),
                'local_range' : None
            }]
        }
    
    kernel_dict_inverse={
            'kernel': kernel_str,
            'name': 'inverse_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [complex,complex,complex,np.int64,np.int64,np.int64],
                'shapes' : [(nq,nlm),inverse_weights.shape,(nq,nlm),None,None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input','const_input'],
                'const_inputs' : [None,inverse_weights,None,np.int64(nq),np.int64(nlm),np.int64(nl)],
                'global_range' : (nq,nlm),
                'local_range' : None
            }]
        }

    forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
    inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

    zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process)
    inverse_zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process)
    return zernike_hankel_transform,inverse_zernike_hankel_transform

def generate_polar_zernike_ht_gpu(w,used_orders):
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    n_radial_points = forward_weights.shape[1]

    m_max=used_orders.max()

    nq = n_radial_points
    nm = m_max*2+1

    kernel_str = """
    __kernel void
    apply_weights(__global double2* out, 
    __global double2* w, 
    __global double2* rho, 
    long nq,long nm)
    {
  
    long i = get_global_id(0); 
    long j = get_global_id(1);
 
    // value stores the element that is 
    // computed by the thread
    double2 value = 0;
    // wlm is of shape (sum_q,nq,m) where sum_q = nq-1
    for (int q = 0; q < nq-1; ++q)
    {
    double2 wqqm = w[q*nq*nm + i*nm + j];
    double2 rqm = rho[(q+1)*nm + j];
    value.x += wqqm.x * rqm.x - wqqm.y * rqm.y;
    value.y += wqqm.x * rqm.y + wqqm.y * rqm.x;
    }
    
    // Write the matrix to device memory each 
    // thread writes one element
    out[i * nm + j] = value;//w[nq*nl+i*nl + l];
    }
    """

    kernel_dict_forward={
            'kernel': kernel_str,
            'name': 'forward_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [complex,complex,complex,np.int64,np.int64],
                'shapes' : [(nq,nm),forward_weights.shape,(nq,nm),None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input'],
                'const_inputs' : [None,forward_weights,None,np.int64(nq),np.int64(nm)],
                'global_range' : (nq,nm),
                'local_range' : None
            }]
        }
    
    kernel_dict_inverse={
            'kernel': kernel_str,
            'name': 'inverse_hankel',
            'functions': [{
                'name': 'apply_weights',
                'dtypes' : [complex,complex,complex,np.int64,np.int64],
                'shapes' : [(nq,nm),inverse_weights.shape,(nq,nm),None,None],
                'arg_roles' : ['output','const_input','input','const_input','const_input'],
                'const_inputs' : [None,inverse_weights,None,np.int64(nq),np.int64(nm)],
                'global_range' : (nq,nm),
                'local_range' : None
            }]
        }

    forward_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
    inverse_gpu_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_inverse)

    zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(forward_gpu_process)
    inverse_zernike_hankel_transform = Multiprocessing.comm_module.add_gpu_process(inverse_gpu_process)
    return zernike_hankel_transform,inverse_zernike_hankel_transform




############################################################
###    create spherical weights for sin cos expansion    ###
def generate_sin_cos_weights_spherical(max_order, n_radial_points,expansion_limit = -1,pi_in_q = True,dimensions=3,n_cpus=False):
    orders = np.arange(max_order+1)
    assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)    
    if expansion_limit < 0:
        expansion_limit = 2*(2*n_radial_points - 1)
        log.info('Default sin cos expansion limit = {}'.format(expansion_limit))
    expansion_orders = np.arange(expansion_limit+1)
    radial_ids = np.arange(n_radial_points)
    mp_mode = Multiprocessing.MPMode_Queue(ignore_first_dimension=True)
    integrals = Multiprocessing.comm_module.request_mp_evaluation(sin_cos_integral_worker,mode = mp_mode,input_arrays=[orders,expansion_orders,radial_ids],const_inputs=[n_radial_points],call_with_multiple_arguments = True,n_processes = n_cpus,split_mode = 'modulus')
    integrals = np.array(integrals)
    
    weights = np.zeros((max_order +1,n_radial_points,n_radial_points),dtype = float)
    summands = np.zeros((max_order +1,expansion_limit + 1,n_radial_points,n_radial_points),dtype = float)
    pi = np.pi
    N = n_radial_points
    ks = np.arange(N)[None,None,:,None]
    ns = expansion_orders[None,:,None,None]
    #log.info(integrals)
    summands[::2] = np.cos(pi*ks*ns/N)*integrals[::2,:,None,:]

    #summands[::2] = np.cos(pi*ks*ns/N)*integrals[1::2,:,None,:]

    summands[1::2] = np.sin(pi*ks*ns/N)*integrals[1::2,:,None,:]
    summands[...,0,:]*= 0.5 # c_target radial index ks
    summands[:,0,...]*= 0.5 # c_sin_order ns
    weights = np.sum(summands,axis = 1)*2*n_radial_points**2
    return weights

def sin_cos_integral_worker(sp_orders,sc_orders,q_ids,n_radial_points,**kwargs):
    n_parts = len(sp_orders)
    integrals = np.zeros((n_parts),dtype = float)
    odd_mask = (sp_orders%2).astype(bool)
    #log.info('integrals shape = {}'.format(integrals.shape))
    integrands = get_integrands_sin_cos(n_radial_points,sp_orders,odd_mask,sc_orders,q_ids)
    
    integrals[~odd_mask] = spIntegrate.quad_vec(integrands[0],0,1)[0]
    integrals[odd_mask] = spIntegrate.quad_vec(integrands[1],0,1)[0]
    return integrals
        
def get_integrands_sin_cos(n_radial_points,ls,odd_mask,ns,ps):
    Nr=n_radial_points
    pi=np.pi
    cos=np.cos
    sin=np.sin
    jnu = spherical_bessel_jnu
    even_mask= ~odd_mask
    even_ls = ls[even_mask]
    even_ps = ps[even_mask]
    even_ns = ns[even_mask]
    odd_ls = ls[odd_mask]
    odd_ps = ps[odd_mask]
    odd_ns = ns[odd_mask]
    #log.info('even ls = {}'.format(even_ls))
    #log.info('odd ls = {}'.format(odd_ls))
    def even(x):
        return jnu(even_ls,x*pi*even_ps)*cos(pi*even_ns*x)*x**2
    def odd(x):
        return jnu(odd_ls,x*pi*odd_ps)*sin(pi*odd_ns*x)*x**2
        #return jnu(odd_ls,x*pi*odd_ps)*sin(pi*odd_ns*x)*x**2
    return [even,odd]
    


############################################################
###    create spherical weights for sin cos expansion    ###
def generate_sin_cos_weights_spherical_midpoint(max_order, n_radial_points,expansion_limit = -1,pi_in_q = True,dimensions=3,n_cpus=False):
    orders = np.arange(max_order+1)
    assert 2*n_radial_points-1>max_order,'Not enough radial points (N_r={}) to calculate weights upto order {}. max_order<2*n_radial_points-1 '.format(n_radial_points,max_order)    
    if expansion_limit < 0:
        expansion_limit = 2*(2*n_radial_points - 1)
        log.info('Default sin cos expansion limit = {}'.format(expansion_limit))
    expansion_orders = np.arange(expansion_limit+1)
    radial_ids = np.arange(n_radial_points)+1/2
    mp_mode = Multiprocessing.MPMode_Queue(ignore_first_dimension=True)
    integrals = Multiprocessing.comm_module.request_mp_evaluation(sin_cos_integral_worker,mode = mp_mode,input_arrays=[orders,expansion_orders,radial_ids],const_inputs=[n_radial_points],call_with_multiple_arguments = True,n_processes = n_cpus,split_mode = 'modulus')
    integrals = np.array(integrals)
    
    weights = np.zeros((max_order +1,n_radial_points,n_radial_points),dtype = float)
    summands = np.zeros((max_order +1,expansion_limit + 1,n_radial_points,n_radial_points),dtype = float)
    pi = np.pi
    N = n_radial_points
    ks = np.arange(N)[None,None,:,None]+1/2
    ns = expansion_orders[None,:,None,None]
    #log.info(integrals)
    summands[::2] = np.cos(pi*ks*ns/N)*integrals[::2,:,None,:]
    summands[1::2] = np.sin(pi*ks*ns/N)*integrals[1::2,:,None,:]
    #summands[:,0,...]*= 0.5 # c_sin_order ns original
    summands[::2,0,...]*= 0.5 # in paper
    weights = np.sum(summands,axis = 1)*2*n_radial_points**2
    return weights
    
