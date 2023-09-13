import numpy as np
import numpy.ma as mp
import scipy.integrate as spIntegrate
from scipy.special import jv as bessel_jv
from scipy.special import roots_legendre
import logging
from itertools import repeat

log=logging.getLogger('root')

import xframe.library.pythonLibrary as pyLib
import xframe.library.mathLibrary as mLib
from xframe.library.gridLibrary import NestedArray

from .ft_grid_pairs import extractGridParameters_Donatelli as extractGridParameters
from .harmonic_transforms import generatePolarIndexList
from .harmonic_transforms import calculateHarmonicOrder
from .harmonic_transforms import HarmonicTransform
from xframe import Multiprocessing

doublePrecision=pyLib.doublePrecision

def generate_ht_bessel(ft_grid_pair):
    real_grid=ft_grid_pair.realGrid
    reciprocal_grid=ft_grid_pair.reciprocalGrid
    n_angular_points=real_grid.shape[-1]
    n_radial_points=real_grid.shape[0]
    max_r=real_grid[-1,0,0]
    orders=np.concatenate((np.arange(int(n_angular_points/2)+1),-1*np.arange(int(n_angular_points/2)+n_angular_points%2)[:0:-1]))    
#    jmn=reciprocal_grid[:,:,0]*max_r
    jnm=mLib.besselZeros(orders,np.arange(n_radial_points))
    jnmk=jnm[:,:,None]*jnm[:,None,:]
    jnN=jnm[:,-1,None,None]
    J_jnmk=np.array(tuple(map( bessel_jv,orders,jnmk/jnN )))
    J2_jnm=np.square(np.array(tuple(map(bessel_jv,orders+1,jnm))))        
    kernel=np.swapaxes((J_jnmk/J2_jnm[:,:,None])[:,1:],0,2)
    forward_prefactor=2*max_r**2/np.square(jnm[None,:,-1])
    log.info('forward shape={}'.format(forward_prefactor.shape))
    inverse_prefactor=2/max_r**2
    def dht(harmonic_coeff):        
        reciprocal_harmonic_coeff=forward_prefactor*np.sum(harmonic_coeff[None,1:,:]*kernel,axis=1)
        return reciprocal_harmonic_coeff
    def i_dht(reciprocal_harmonic_coeff):
        harmonic_coeff=inverse_prefactor*np.sum(reciprocal_harmonic_coeff[None,1:,:]*kernel,axis=1)
        return harmonic_coeff
    return dht,i_dht
        
        

####################
###HT via adaptive Quad###
def generateHT_quad(FTGridPair):
    params=extractGridParameters(FTGridPair)
    nRadialSteps=params['nRadialSteps']
    nAngularSteps=params['nAngularSteps']
    reciprocalCutOff=params['reciprocalCutOff']
    realCutOff=params['realCutOff']

    realRadGridSpec=[0,realCutOff/nRadialSteps]
    reciprocalRadGridSpec=[0,1/realCutOff]
    def ht_forward(harmonicCoefficients):
        reciprocalHarmonicCoefficients=hankelTransform_quad_forward(harmonicCoefficients,nAngularSteps,nRadialSteps,realRadGridSpec,realCutOff)
        return reciprocalHarmonicCoefficients
    def ht_inverse(reciprocalHCoefficients):
        harmonicCoefficients=hankelTransform_quad_inverse(reciprocalHarmonicCoefficients,nAngularSteps,nRadialSteps,reciprocalRadGridSpec,reciprocalCutOff)
        return harmonicCoefficients
    return ht_forward,ht_inverse
        
def getHT_Integrand(coeffFunction,harmOrder,radialPoint):
    jnu=mLib.bessel_jnu
    m=harmOrder
    pi=np.pi
    rp=radialPoint
    def integrand(x):
        value=coeffFunction(x)*jnu(m,2*pi*rp*x)*x
        return value
    return integrand

def hankelTransform_quad_forward(harmonicCoefficients,nAngularSteps,nRadialSteps,radialGridSpec,realCutOff):        
    coefficientFunctions=[pyLib.uniformGridToFunction(harmonicCoefficients.array[:,angularIndex],radialGridSpec) for angularIndex in range(nAngularSteps)]
    newArray=np.zeros((harmonicCoefficients.shape),dtype=np.complex)
    for angularIndex in range(nAngularSteps):
        m=calculateHarmonicOrder(angularIndex,nAngularSteps)
        coeffFunction=coefficientFunctions[angularIndex]
        for radialIndex in range(nRadialSteps):
            qn=radialIndex/realCutOff
            integrand=getHT_Integrand(coeffFunction,m,qn)
            newArray[radialIndex,angularIndex]=2*np.pi*(-1.j)**m*spIntegrate.quad(integrand,0,realCutOff)[0]
    harmonicCoefficients.array=newArray
    reciprocalHarmonicCoefficients=harmonicCoefficients
    return reciprocalHarmonicCoefficients

def hankelTransform_quad_inverse(reciprocalHarmonicCoefficients,nAngularSteps,nRadialSteps,radialGridSpec,reciprocalCutOff):        
    coefficientFunctions=[pyLib.uniformGridToFunction(reciprocalHarmonicCoefficients.array[:,angularIndex],radialGridSpec) for angularIndex in range(nAngularSteps)]    
    radialStepSize=realCutOff/nRadialSteps
    newArray=np.zeros((harmonicCoefficients.shape),dtype=np.complex)
    for angularIndex in range(nAngularSteps):
        m=calculateHarmonicOrder(angularIndex,nAngularSteps)
        coeffFunction=coefficientFunctions[angularIndex]
        for radialIndex in range(nRadialSteps):
            rn=radialIndex*radialStepSize
            integrand=getHT_Integrand(coeffFunction,m,rn)
            newArray[radialIndex,angularIndex]=2*np.pi*(1.j)**m*spIntegrate.quad(integrand,0,reciprocalCutOff)
    reciprocalHarmonicCoefficients.array=newArray
    harmonicCoefficients=reciprocalHarmonicCoefficients
    return harmonicCoefficients

#####################
###HT via Trapezoidal rule###
def generateHT_trapezeoidal(FTGridPair):
    params=extractGridParameters(FTGridPair)
    nRadialSteps=params['nRadialSteps']
    nAngularSteps=params['nAngularSteps']
    reciprocalCutOff=params['reciprocalCutOff']
    realCutOff=params['realCutOff']

    def ht_forward(harmonicCoefficients):
        reciprocalHarmonicCoefficients=hankelTransform_trapezoidal(harmonicCoefficients,realCutOff,getSummand_trapezoidal_forward)
        return reciprocalHarmonicCoefficients
    def ht_inverse(reciprocalHarmonicCoefficients):
        harmonicCoefficients=hankelTransform_trapezoidal(reciprocalHarmonicCoefficients,realCutOff,getSummand_trapezoidal_inverse)
        return harmonicCoefficients
    return ht_forward,ht_inverse

def hankelTransform_trapezoidal(harmonicCoefficients,realCutOff,summandFunction):
    array=harmonicCoefficients.array
    nAngularSteps=array.shape[1]
    nRadialSteps=array.shape[0]
        
    trf_Array=np.zeros((nRadialSteps,nAngularSteps),dtype=np.complex)
    for angularIndex in range(nAngularSteps):
        harmonicOrder=calculateHarmonicOrder(angularIndex,nAngularSteps)
        for radialIndex in range(nRadialSteps):
            fixedVariables= [array[:,angularIndex],radialIndex,harmonicOrder,nRadialSteps,realCutOff]
            summands=np.array(list(map(summandFunction,range(nRadialSteps),repeat(fixedVariables)  ) ) )
            trf_Array[radialIndex,angularIndex]=np.sum(summands)
    harmonicCoefficients.array=trf_Array
    reciprocalHarmonicCoefficients=harmonicCoefficients
    return reciprocalHarmonicCoefficients

def getSummand_trapezoidal_forward(sumIndex, fixedVariables):    
    R=fixedVariables[4]
    N=fixedVariables[3]
    n=fixedVariables[1]
    m=fixedVariables[2]
    rho=fixedVariables[0]
#    log.info('sumIndex={}'.format(sumIndex))
    k=sumIndex

    #    log.info('rho={}'.format(rho))
    #    log.info('k={} R={} N={}'.format(k,R,N))
    prefix=2*np.pi*(R/N)**2*((-1.j)**(m))
    if k==N-1:
        prefix*=1.5
        
    summand=rho[k]*mLib.bessel_jnu(m,2*np.pi*n*k/N)*k
#    log.info('prefix={},  summand={}'.format(prefix,summand))
    return prefix*summand

def getSummand_trapezoidal_inverse(sumIndex, fixedVariables):
    R=fixedVariables[4]
    N=fixedVariables[3]
    n=fixedVariables[1]
    m=fixedVariables[2]
    rho=fixedVariables[0]
    k=sumIndex

    prefix=2*np.pi*(1/R**2)*(1.j)**(m)
    if k==N-1:
        prefix*=1.5

    summand=rho[k]*mLib.bessel_jnu(m,2*np.pi*n*k/N)*k
    return prefix*summand



########################################
###     HT via Zernike expansion     ###
def calc_zernike_weights(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    radial_points=opt['n_radial_points']+1
    ks=np.arange(1,radial_points)
    ps=np.arange(1,radial_points)
    zernike_dict=mLib.eval_zernike_polynomials(pos_orders,expansion_limit,points=ps/radial_points)

    def weights_for_fixed_order(m):
        ns=np.arange(m,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-m)/2 )*(2*ns+2)
        Jk=mLib.bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks)
        Zp=zernike_dict[m]
        summands=prefactor[:,None,None]*Zp[:,:,None]*Jk[:,None,:]
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))

    c_pk=ps[None,:,None]/(ks[None,None,:])

    weights=weights*c_pk
    return weights

def calc_zernike_weights_pi(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    radial_points=opt['n_radial_points']+1
    ks=np.arange(1,radial_points)
    ps=np.arange(1,radial_points)
    zernike_dict=mLib.eval_zernike_polynomials(pos_orders,expansion_limit,points=ps/radial_points)

    def weights_for_fixed_order(m):
        ns=np.arange(m,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-m)/2 )*(2*ns+2)
        Jk=mLib.bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),2*np.pi*ks)
        Zp=zernike_dict[m]
        summands=prefactor[:,None,None]*Zp[:,:,None]*Jk[:,None,:]
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))

    c_pk=ps[None,:,None]/(ks[None,None,:])

    weights=weights*c_pk
    return weights

def generate_zernike_ht(weights,pos_orders,r_max):
    n_pos_orders=len(pos_orders)
    if pos_orders[0]==0:
        full_weights=np.zeros((2*n_pos_orders-1,)+weights.shape[1:])
        full_weights[:n_pos_orders]=weights[:n_pos_orders]
        full_weights[:n_pos_orders-1:-1]=(-1)**pos_orders[1:,None,None]*weights[1:n_pos_orders]
        orders=np.concatenate((pos_orders,-1*pos_orders[:0:-1]))
    else:
        full_weights=np.zeros((2*n_pos_orders,)+weights.shape[1:])
        full_weights[:n_pos_orders]=weights[:n_pos_orders]
        full_weights[:n_pos_orders-1:-1]=(-1)**pos_orders*weights[:n_pos_orders]
        orders=np.concatenate((pos_orders,-1*pos_orders[::-1]))

    n_radial_points=weights.shape[-1]+1
    forward_prefactor=(-1.j)**(orders[None,None,:])*((r_max**2)/(n_radial_points**2))
    inverse_prefactor=(1.j)**(orders[None,None,:])/(r_max**2)
    
    weights=np.swapaxes(full_weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor

    def zht(harmonic_coeff):
        temp=forward_weights*harmonic_coeff
        reciprocal_harmonic_coeff=np.sum(temp,axis=1)
        #log.info('reciprocal_harmonic_coeff shape={}'.format(reciprocal_harmonic_coeff.shape))
        return reciprocal_harmonic_coeff
    def izht(reciprocal_coeff):
        temp=inverse_weights*reciprocal_coeff
        harmonic_coeff=np.sum(temp,axis=1)
        return harmonic_coeff
    return zht,izht





#################################################
###    spherical HT via Zernike expansion     ###

def generate_weightDict_zernike_spherical(max_order, n_radial_points,expansion_limit = -1,pi_in_q = True,n_cpus=False):
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
    if not isinstance(n_cpus,bool):
        weights = Multiprocessing.comm_module.request_mp_evaluation(calc_spherical_zernike_weights,argArrays=[orders],const_args=[n_radial_points,expansion_limit,pi_in_q],callWithMultipleArguments=True,n_processes=n_cpus)
    else:
        weights = Multiprocessing.comm_module.request_mp_evaluation(calc_spherical_zernike_weights,argArrays=[orders],const_args=[n_radial_points,expansion_limit,pi_in_q],callWithMultipleArguments=True)
    weightDict={}
    weightDict['weights'] = weights
    weightDict['posHarmOrders'] = orders
    weightDict['mode'] = 'Zerinike_3D'
    return weightDict
    
def calc_spherical_zernike_weights_old(orders,n_radial_points,expansion_limit,pi_in_q,**kwargs):
    '''
    Calculates the weights for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
    '''    
    radial_points=n_radial_points    
    #radial_points=opt['n_radial_points']+1
    ks=np.arange(1,radial_points)
    n_k = radial_points-1 
    ps=np.arange(radial_points)
    if pi_in_q:
        j_factor=np.pi
    else:
        j_factor=1/2
    #    log.info('ps={}'.format(ps/2))
    n_p = radial_points
    #weights=np.zeros((len(orders),n_p,n_k))
    #zernike_dict=mLib.eval_ND_zernike_polynomials(orders,expansion_limit,ks/radial_points,3)

    def weights_for_fixed_order(l):
        zernike_dict=mLib.eval_ND_zernike_polynomials(np.array([l]),expansion_limit,ks/radial_points,3)
        log.info('finished_zernike dict')
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ps[1:]*j_factor)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        #log.info('making summands for order = {}'.format(l))
        summands = np.zeros((len_ns,n_k,n_p))
        summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zk[0,:]
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,orders)))
    c_kp=np.zeros((n_k,n_p))
    c_kp[:,1:]=np.square(ks)[:,None]/(ps[None,1:])
    c_kp[:,0]=np.square(ks)
    weights=weights*c_kp[None,:,:]*np.sqrt(2/np.pi)
    return weights

def calc_spherical_zernike_weights(orders,n_radial_points,expansion_limit,pi_in_q,**kwargs):
    '''
    Calculates the weights for the Zernike version of the approximated hankel transform.
    There is no sense to chose expansion_limit higher than 2*n_radial_points-1, since this is the limit ing degree of a polynomial (i.e. also the Zernike polynomials) that could be resolved on a gauss legendre grid. In this routine we use the trapezoidal rule so it is probably advisible to stay below that limit.
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
    #zernike_dict=mLib.eval_ND_zernike_polynomials(orders,expansion_limit,ks/radial_points,3)

    def weights_for_fixed_order(l):
        zernike_dict=mLib.eval_ND_zernike_polynomials(np.array([l]),expansion_limit,ps/radial_points,3)
        #log.info('finished_zernike dict')
        s=np.arange(l,expansion_limit+1,2)
        len_s=len(s)
        prefactor=(-1)**( (s-l)/2 )*(2*s+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((s+1)[:,None],radial_points-1,axis=1),ks[1:]*j_factor)
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
    weights=weights*c_kp[None,:,:]*np.sqrt(2/np.pi)
    return weights



def calc_spherical_zernike_weights_old(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(1,radial_points)
    ps=np.arange(1,radial_points)
    weights=np.zeros((len(pos_orders),radial_points,radial_points))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ps/radial_points,3)

    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jk=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks)
        Zp=zernike_dict[l]
        summands=prefactor[:,None,None]*Zp[:,:,None]*jk[:,None,:]
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_pk=np.square(ps)[None,:,None]/(ks[None,None,:])
    weights=weights*c_pk*np.sqrt(2/np.pi)
    return weights

def calc_spherical_zernike_weights_new2_old(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=max( max(pos_orders)+1,opt['expansion_limit'])
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(1,radial_points)
    n_k = radial_points-1 
    ps=np.arange(radial_points)
    #    log.info('ps={}'.format(ps/2))
    n_p = radial_points
    #weights=np.zeros((len(pos_orders),n_p,n_k))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ks/radial_points,3)
    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ps[1:]/2)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        summands = np.zeros((len_ns,n_k,n_p))
        summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zk[0,:]*1/3*1/2 #*1/2 is important
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        if l == 1:
            pass
            #log.info('w sum = {}'.format(weights))
            #log.info('w sum = {}'.format(weights.sum()))
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_kp=np.zeros((n_k,n_p))
    c_kp[:,1:]=np.square(ks)[:,None]/(ps[None,1:])
    c_kp[:,0]=np.square(ks)
    weights=weights*c_kp[None,:,:]*np.sqrt(2/np.pi)
    return weights

def calc_spherical_zernike_weights_pi(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=max( max(pos_orders)+1,opt['expansion_limit'])
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(1,radial_points)
    n_k = radial_points-1 
    ps=np.arange(radial_points)
    #    log.info('ps={}'.format(ps/2))
    n_p = radial_points
    #weights=np.zeros((len(pos_orders),n_p,n_k))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ks/radial_points,3)
    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ps[1:]*np.pi)
        #log.info('jp shape =  {}'.format(jp.shape))
        Zk=zernike_dict[l]
        #log.info('Zk shape = {}'.format(Zk.shape))
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        summands = np.zeros((len_ns,n_k,n_p))
        summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zk[0,:] #*1/2 is important
        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        if l == 1:
            pass
            #log.info('w sum = {}'.format(weights))
            #log.info('w sum = {}'.format(weights.sum()))
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_kp=np.zeros((n_k,n_p))
    c_kp[:,1:]=np.square(ks)[:,None]/(ps[None,1:])
    c_kp[:,0]=np.square(ks)
    weights=weights*c_kp[None,:,:]*2/np.sqrt(2*np.pi)
    return weights


def calc_spherical_zernike_weights_gauss(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']

    N=opt['n_radial_points']
    kk,w_gauss = roots_legendre(N)
    ids=np.argsort(kk)
    k=(kk[ids]+1)/2
    w_gauss=w_gauss[ids]
    
    p=N/2*k.copy()
    #log.info('p={}'.format(p))


    ks=np.arange(0,radial_points)
    n_k = radial_points
    ps=np.arange(radial_points)
    n_p = radial_points
    #weights=np.zeros((len(pos_orders),n_p,n_k))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,k,3)
    n = tuple(np.arange(l,expansion_limit+1,2) for l in pos_orders)
    
    def n_sum(l):
        ns=n[l]
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points,axis=1),p)
        #log.info('jp = \n {}'.format(jp))
        Zk=zernike_dict[l]
        # log.info('prefactor = \n {}'.format(prefactor))
        #log.info('Zk = \n {}'.format(Zk))
        summands = np.zeros((len_ns,n_k,n_p))
        if 0 in p:
            summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,1:]
            if 0 in ns:
                summands[0,:,0]= prefactor[0]*Zk[0,:]*1/3*1/2 #*1/2 is important
        else:
            summands[:,:,:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]

        weights=np.sum(summands,axis=0)
        #log.info('w sum = {}'.format(weights.sum()))#
        if l == 1:
            log.info('w = {}'.format(weights))
            log.info('w sum = {}'.format(weights.sum()))
        return weights
    n_sums= np.array(tuple(n_sum(l) for l in pos_orders))
    if 0 in p:
        kp=np.zeros((len(k),len(p)))
        kp[:,1:]=np.square(N*k)[:,None]/(p[None,1:])
        kp[:,0]=np.square(N*k)
    else:
        kp=np.square(k)[:,None]/(p[None,:])

    weights=kp[None,:,:]*w_gauss[None,:,None]*n_sums#*np.sqrt(2/np.pi) #l k p
    return weights

def calc_spherical_zernike_weights_gauss2(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']    
    #radial_points=opt['n_radial_points']+1
    N=opt['n_radial_points']
    kk,w_gauss = roots_legendre(N)
    ids=np.argsort(kk)
    k=(kk[ids]+1)/2
    p=N/2*k.copy()
    log.info(p)
    p=k.copy()

    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,k,3)
    n = tuple(np.arange(l,expansion_limit+1,2) for l in pos_orders)
    def n_sum(l):
        prefactor = ((-1)**((n[l]-l)/2)) * (2*n[l]+3)
        jp=mLib.spherical_bessel_jnu(np.repeat((n[l]+1)[:,None],N,axis=1),p)
        #log.info('jp = \n {}'.format(jp))
        if (0 in n[l]) and (0 in p):
            jp[0,0]=1/3
        #log.info('prefactor = \n {}'.format(prefactor))
        Zk = zernike_dict[l]
        summands = np.zeros((len(n[l]),len(k),len(p)))
        if 0 in p:
            summands[:,:,1:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,1:]
            if 0 in ns:
                summands[0,:,0]= prefactor[0]*Zk[0,:]*1/3*1/2 #*1/2 is important
        else:
            summands[:,:,:] = prefactor[:,None,None]*Zk[:,:,None]*jp[:,None,:]
        weights=np.sum(summands,axis=0)
        #log.info('Zk = \n {}'.format(Zk))
        w =np.sum( prefactor[:,None,None] * Zk[:,:,None]*jp[:,None,:],axis=0)
        if l == 0:
            log.info('w sum = {}'.format(w))
        return w
    n_sums= np.array(tuple(n_sum(l) for l in pos_orders))
    if 0 in p:
        kp=np.zeros((len(k),len(p)))
        kp[:,1:]=np.square(k)[:,None]/(p[None,1:])
        kp[:,0]=np.square(k)
    else:
        kp=np.square(k)[:,None]/(p[None,:])

    weights=kp[None,:,:]*w_gauss[None,:,None]*n_sums#*np.sqrt(2/np.pi) #l k p
    return weights

def calc_spherical_zernike_weights_backup(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(radial_points)
    ps=np.arange(radial_points)
    weights=np.zeros((len(pos_orders),radial_points,radial_points))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ps/radial_points,3)

    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jk=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks[1:]/2)        
        Zp=zernike_dict[l]
        summands = np.zeros((len_ns,radial_points,radial_points))
        summands[:,:,1:] = prefactor[:,None,None]*Zp[:,:,None]*jk[:,None,:]
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zp[0,:]*1/3*1/2 #*1/2 is important
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_pk=np.zeros((radial_points,radial_points))
    c_pk[:,1:]=np.square(ps)[:,None]/(ks[None,1:])
    c_pk[:,0]=np.square(ps)
    weights=weights*c_pk[None,:,:]*np.sqrt(2/np.pi)
    return weights

def calc_spherical_zernike_weights_new(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(radial_points)
    ps=np.arange(radial_points)
    #weights=np.zeros((len(pos_orders),radial_points,radial_points))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ps/radial_points,3)

    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jk=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks[1:])
        Zp=zernike_dict[l]
        summands = np.zeros((len_ns,radial_points,radial_points))
        summands[:,:,1:]=prefactor[:,None,None]*Zp[:,:,None]*jk[:,None,:]#ns,ps,ks
        summands[:,:,0]=0
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zp[0,:]*1/3
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_pk=np.zeros((radial_points,radial_points))
    c_pk[:,1:] = np.square(ps)[:,None]/(ks[None,1:])
    c_pk[:,0] = np.square(ps)
    weights=weights*c_pk[None,:,:]*np.sqrt(2/np.pi)
    return weights
def calc_spherical_zernike_weights_new22(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(radial_points)
    ps=np.arange(radial_points)
    #weights=np.zeros((len(pos_orders),radial_points,radial_points))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ps/radial_points,3)

    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jk=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks[1:]/2)
        Zp=zernike_dict[l]
        summands = np.zeros((len_ns,radial_points,radial_points))
        summands[:,:,1:]=prefactor[:,None,None]*Zp[:,:,None]*jk[:,None,:]#ns,ps,ks
        #summands[:,:,0]=0
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zp[0,:]*1/3
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_pk=np.zeros((radial_points,radial_points))
    c_pk[:,1:] = np.square(ps)[:,None]/(np.square(ks)[None,1:])
    c_pk[:,0] = np.square(ps)
    weights=weights*c_pk[None,:,:]*np.sqrt(2/np.pi)
    return weights
def calc_spherical_zernike_weights_new223(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']
    ks=np.arange(radial_points)
    ps=np.arange(radial_points)
    #weights=np.zeros((len(pos_orders),radial_points,radial_points))
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ps/radial_points,3)

    def weights_for_fixed_order(l):
        ns=np.arange(l,expansion_limit+1,2)
        len_ns=len(ns)
        prefactor=(-1)**( (ns-l)/2 )*(2*ns+3)
        jk=mLib.spherical_bessel_jnu(np.repeat((ns+1)[:,None],radial_points-1,axis=1),ks[1:]/2)
        Zp=zernike_dict[l]
        summands = np.zeros((len_ns,radial_points,radial_points))
        summands[:,:,1:]=prefactor[:,None,None]*Zp[:,:,None]*jk[:,None,:]#ns,ps,ks
        summands[:,:,0]=0
        if l == 0:
            summands[0,:,0]= prefactor[0]*Zp[0,:]*1/3
        weights=np.sum(summands,axis=0)
        return weights
    weights=np.array(tuple(map(weights_for_fixed_order,pos_orders)))
    c_pk=np.zeros((radial_points,radial_points))
    c_pk[:,1:] = np.square(ps)[:,None]/(ks[None,1:])
    c_pk[:,0] = np.square(ps)
    weights=weights*c_pk[None,:,:]*np.sqrt(2/np.pi)
    return weights


def assemble_weights_zernike_old(weights,r_max,l_orders,pi_in_q):
    l_max = l_orders.max()
    n_radial_points = weights.shape[-1]
    if pi_in_q:
        q_max = np.pi*n_radial_points/r_max
    else:
        q_max = n_radial_points/(r_max*2)
        
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    forward_prefactor_0 = r_max**3/(3*n_radial_points**3)
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))
    inverse_prefactor_0 = q_max**3/(3*n_radial_points**3)

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    forward_weights[:,0,0]=weights[:,0,0]*forward_prefactor_0
    inverse_weights=weights*inverse_prefactor
    inverse_weights[:,0,0]=weights[:,0,0]*inverse_prefactor_0
    return {'forward':forward_weights,'inverse':inverse_weights}

def assemble_weights_zernike(weights,r_max,l_orders,pi_in_q):
    l_max = l_orders.max()
    n_radial_points = weights.shape[-1]
    if pi_in_q:
        q_max = np.pi*n_radial_points/r_max
    else:
        q_max = n_radial_points/(r_max*2)
        
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(np.pi**2/q_max**3)
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(np.pi**2/r_max**3)

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    return {'forward':forward_weights,'inverse':inverse_weights}

def generate_spherical_zernike_ht(weights,l_orders,r_max,pi_in_q=False):
    w = assemble_weights_zernike(weights,r_max,l_orders,pi_in_q)

    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht


def generate_spherical_zernike_ht_gpu(weights,l_orders,r_max,pi_in_q = False):
    log.info('assemble weight shape = {}'.format(weights.shape))
    n_radial_points = weights.shape[-1]
    w = assemble_weights_zernike(weights,r_max,l_orders,pi_in_q)

    
    forward_weights= w['forward']
    inverse_weights= w['inverse']
    log.info('assemble weight shape 2 = {}'.format(forward_weights.shape))
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
    for (int q = 0; q < nq; ++q)
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
    
def generate_spherical_zernike_ht_gpu_double2(weights,l_orders,r_max,pi_in_q = False):
    n_radial_points = weights.shape[-1]
    w = assemble_weights_zernike(weights,r_max,l_orders,pi_in_q)

    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()

    nq = n_radial_points
    nl = len(l_orders)
    nlm = l_max*(l_max+2)+1

    cld = Multiprocessing.load_openCL_dict()
    cl = cld['cl']
    ctx = cld['context']
    queue = cld['queue']

    apply_weights = cl.Program(ctx, """
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
    for (int q = 0; q < nq; ++q)
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
    """).build().apply_weights
    
    apply_weights.set_scalar_arg_dtypes([None,None,None,np.int64,np.int64,np.int64])
        
    out_f = np.zeros((nq,nlm),dtype = complex)
    out_i = np.zeros((nq,nlm),dtype = complex)

    mf = cl.mem_flags
    w_f_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(forward_weights))
    w_i_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(inverse_weights))

    #queue = cl.CommandQueue(ctx)

    # w_f_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=forward_weights.nbytes)
    # w_i_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=inverse_weights.nbytes)
    rho_f_dev = cl.Buffer(ctx , mf.READ_WRITE, size=out_f.nbytes)
    rho_i_dev = cl.Buffer(ctx , mf.READ_WRITE, size=out_i.nbytes)
    out_f_dev = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_f.nbytes)
    out_i_dev = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_i.nbytes)
    
    #cl.enqueue_copy(queue,w_f_dev,np.ascontiguousarray(forward_weights))
    #cl.enqueue_copy(queue,w_i_dev,np.ascontiguousarray(inverse_weights))

    local_range = None
    global_range = (nq,nlm)
    #log.info('start test')
    #cl.enqueue_copy(queue,rho_f_dev,np.random.rand(*out_f.shape))
    #apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,nq,nlm,nl)
    #cl.enqueue_copy(queue, out_f,out_f_dev)
    #log.info(queue)
    #log.info(rho_f_dev)
    #log.info('end test')
    def zht(harmonic_coeff):
        #log.info(queue)
        #log.info(rho_f_dev)     
        cl.enqueue_copy(queue,rho_f_dev,harmonic_coeff)
        apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,nq,nlm,nl)
        cl.enqueue_copy(queue, out_f,out_f_dev)
        return out_f
    
    def izht(reciprocal_coeff):
        cl.enqueue_copy(queue,rho_i_dev,reciprocal_coeff)
        apply_weights(queue,global_range,local_range,out_i_dev,w_i_dev,rho_i_dev,nq,nlm,nl)
        cl.enqueue_copy(queue, out_i,out_i_dev)
        return out_i    
    return zht,izht


def generate_spherical_zernike_ht_gpu_double(weights,l_orders,r_max,pi_in_q = False):
    cld = Multiprocessing.load_openCL_dict()
    cl = cld['cl']
    ctx = cld['context']
    queue = cld['queue']
    
    n_radial_points = weights.shape[-1]
    w = assemble_weights_zernike(weights,r_max,l_orders,pi_in_q)

    forward_weights= w['forward']
    inverse_weights= w['inverse']

    l_max=l_orders.max()

    nq = n_radial_points
    nl = len(l_orders)
    nlm = l_max*(l_max+2)+1    

    apply_weights = cl.Program(ctx, """
    __kernel void
    apply_weights(__global double* out_r, 
    __global double* out_i, 
    __global double* w_r, 
    __global double* w_i, 
    __global double* rho_r, 
    __global double* rho_i, 
    long nq,long nlm, long nl)
    {
  
    long i = get_global_id(0); 
    long j = get_global_id(1);
    long l = (long) sqrt((double)j);

 
    // value stores the element that is 
    // computed by the thread
    double value_r = 0;
    double value_i = 0;
    for (int q = 0; q < nq; ++q)
    {
    long weight_index = q*nq*nl + i*nl + l;
    long coeff_index = (q+1)*nlm + j;
    double wqql_r = w_r[weight_index];
    double wqql_i = w_i[weight_index];
    double rqlm_r = rho_r[coeff_index];
    double rqlm_i = rho_i[coeff_index];
    value_r += wqql_r * rqlm_r - wqql_i * rqlm_i;
    value_i += wqql_r * rqlm_i + wqql_i * rqlm_r;
    }
    
    // Write the matrix to device memory each 
    // thread writes one element
    out_r[i * nlm + j] = value_r;//w[nq*nl+i*nl + l];
    out_i[i * nlm + j] = value_i;//w[nq*nl+i*nl + l];
    }
    """).build().apply_weights
    
    apply_weights.set_scalar_arg_dtypes([None,None,None,None,None,None,np.int64,np.int64,np.int64])

    out_f = np.zeros((nq,nlm),dtype = np.complex)
    out_i = np.zeros((nq,nlm),dtype = np.complex)
    out_f_r = np.zeros((nq,nlm),dtype = np.double)
    out_f_i = np.zeros((nq,nlm),dtype = np.double)
    out_i_r = np.zeros((nq,nlm),dtype = np.double)
    out_i_i = np.zeros((nq,nlm),dtype = np.double)

    mf = cl.mem_flags
    w_f_dev_r = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(forward_weights.real))
    w_f_dev_i = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(forward_weights.imag))
    w_i_dev_r = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(inverse_weights.real))
    w_i_dev_i = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(inverse_weights.imag))

    #queue = cl.CommandQueue(ctx)

    rho_f_dev_r = cl.Buffer(ctx , mf.READ_WRITE, size=out_f_r.nbytes)
    rho_f_dev_i = cl.Buffer(ctx , mf.READ_WRITE, size=out_f_i.nbytes)
    rho_i_dev_r = cl.Buffer(ctx , mf.READ_WRITE, size=out_i_r.nbytes)
    rho_i_dev_i = cl.Buffer(ctx , mf.READ_WRITE, size=out_i_i.nbytes)
    out_f_dev_r = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_f_r.nbytes)
    out_f_dev_i = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_f_i.nbytes)
    out_i_dev_r = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_i_r.nbytes)
    out_i_dev_i = cl.Buffer(ctx , mf.WRITE_ONLY, size=out_i_i.nbytes)

    #rho_f_dev_r = cl.Buffer(ctx , mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_f_r))
    #rho_f_dev_i = cl.Buffer(ctx , mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_f_i))
    #rho_i_dev_r = cl.Buffer(ctx , mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_i_r))
    #rho_i_dev_i = cl.Buffer(ctx , mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_i_i))    
    #out_f_dev_r = cl.Buffer(ctx , mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_f_r))
    #out_f_dev_i = cl.Buffer(ctx , mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_f_i))
    #out_i_dev_r = cl.Buffer(ctx , mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_i_r))
    #out_i_dev_i = cl.Buffer(ctx , mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(out_i_i))


    local_range = None
    global_range = (nq,nlm)

    def zht(harmonic_coeff):
        #log.info(queue)
        #log.info(rho_f_dev)
        cl.enqueue_copy(queue,rho_f_dev_r,np.ascontiguousarray(harmonic_coeff.real))
        cl.enqueue_copy(queue,rho_f_dev_i,np.ascontiguousarray(harmonic_coeff.imag))
        #completeEvent = apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,np.int64(nq),np.int64(nlm),np.int64(nl))
        apply_weights(queue,global_range,local_range,
                                       out_f_dev_r,
                                       out_f_dev_i,
                                       w_f_dev_r,
                                       w_f_dev_i,
                                       rho_f_dev_r,
                                       rho_f_dev_i,
                                       nq,nlm,nl)
        #cl.enqueue_copy(queue, out_f,out_f_dev)
        cl.enqueue_copy(queue, out_f_r,out_f_dev_r)
        cl.enqueue_copy(queue, out_f_i,out_f_dev_i)
        out_f.real = out_f_r
        out_f.imag = out_f_i
        return out_f
    
    def izht(reciprocal_coeff):
        cl.enqueue_copy(queue,rho_i_dev_r,np.ascontiguousarray(reciprocal_coeff.real))
        cl.enqueue_copy(queue,rho_i_dev_i,np.ascontiguousarray(reciprocal_coeff.imag))
        #completeEvent = apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,np.int64(nq),np.int64(nlm),np.int64(nl))
        apply_weights(queue,global_range,local_range,
                                       out_i_dev_r,
                                       out_i_dev_i,
                                       w_i_dev_r,
                                       w_i_dev_i,
                                       rho_i_dev_r,
                                       rho_i_dev_i,
                                       nq,nlm,nl)        
        cl.enqueue_copy(queue, out_i_r,out_i_dev_r)
        cl.enqueue_copy(queue, out_i_i,out_i_dev_i)
        out_i.real = out_i_r
        out_i.imag = out_i_i
        return out_i

    #test transforms
    result = izht(zht(np.full(out_f.shape,1+1.j)))
    return zht,izht


def generate_spherical_zernike_ht_old(weights,l_orders,r_max):
    #n_radial_points=weights.shape[-1]+1
    n_radial_points=weights.shape[-1]
    forward_prefactor=(-1.j)**(l_orders[None,None,:])*((r_max**3)/(n_radial_points**3))
    inverse_prefactor=(1.j)**(l_orders[None,None,:])/(r_max**3)

    weights=np.swapaxes(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor

    def zht(harmonic_coeff):
        return np.sum(forward_weights*harmonic_coeff,axis=1) 
    def izht(reciprocal_coeff):
        return np.sum(inverse_weights*reciprocal_coeff,axis=1)
    return zht,izht

def generate_spherical_zernike_ht_gauss(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    #forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    #inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2)
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2)
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    
    weights=np.swapaxes(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor

    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_new(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))

    weights=np.swapaxes(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor

    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_new2_old(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor

    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_new2(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    forward_prefactor_0 = r_max**3/(3*n_radial_points**3)
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))
    inverse_prefactor_0 = q_max**3/(3*n_radial_points**3)

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    forward_weights[:,0,0]=weights[:,0,0]*forward_prefactor_0
    inverse_weights=weights*inverse_prefactor
    inverse_weights[:,0,0]=weights[:,0,0]*inverse_prefactor_0
    
    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_pi(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = np.pi*n_radial_points/r_max
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    forward_prefactor_0 = r_max**3/(3*n_radial_points**3)
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))
    inverse_prefactor_0 = q_max**3/(3*n_radial_points**3)

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    forward_weights[:,0,0]=weights[:,0,0]*forward_prefactor_0
    inverse_weights=weights*inverse_prefactor
    inverse_weights[:,0,0]=weights[:,0,0]*inverse_prefactor_0

    def zht(harmonic_coeff):
        return tuple(np.sum(forward_weights[:,:,np.abs(m):]*harmonic_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders) 
    def izht(reciprocal_coeff):
        return tuple(np.sum(inverse_weights[:,:,np.abs(m):]*reciprocal_coeff[m][1:,None,:l_max - np.abs(m)+1],axis=0) for m in m_orders)
    return zht,izht

def generate_spherical_zernike_ht_big_sum(weights,l_orders,r_max):
    #n_radial_points = weights.shape[-1]+1
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    forward_prefactor = (-1.j)**(l_orders)*(r_max**2/(q_max*n_radial_points**2))
    inverse_prefactor = (1.j)**(l_orders)*(q_max**2/(r_max*n_radial_points**2))


#    repeated_forward_weights=np.zeros((l_max*(2+l_max)+1,)+weights.shape[1:])
    repeated_forward_weights=np.concatenate(tuple(np.repeat(weights[l][None,...]*forward_prefactor[l],2*l+1,axis=0) for l in range(l_max+1)),axis=0)
    repeated_inverse_weights=np.concatenate(tuple(np.repeat(weights[l][None,...]*inverse_prefactor[l],2*l+1,axis=0) for l in range(l_max+1)),axis=0)
    repeated_forward_weights=np.moveaxis(repeated_forward_weights,0,2)
    repeated_inverse_weights=np.moveaxis(repeated_inverse_weights,0,2)
    log.info('big_weights.shape = {} L*(2+L)+1= {}'.format(repeated_forward_weights.shape,l_max*(2+l_max)+1))

    def zht(harmonic_coeff):
        return np.sum(repeated_forward_weights*harmonic_coeff[1:,None,:],axis=0),repeated_forward_weights
    def izht(reciprocal_coeff):
        return np.sum(repeated_inverse_weights*reciprocal_coeff[1:,None,:],axis = 0)
    return zht,izht

def generate_spherical_zernike_ht_gpu_old(weights,l_orders,r_max):
    n_radial_points = weights.shape[-1]
    q_max = n_radial_points/(r_max*2)
    l_max = l_orders.max()
    m_orders=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    forward_prefactor = (-1.j)**(l_orders[None,None,:])*(r_max**2/(q_max*n_radial_points**2))
    inverse_prefactor = (1.j)**(l_orders[None,None,:])*(q_max**2/(r_max*n_radial_points**2))

    weights=np.moveaxis(weights,0,2)
    forward_weights=weights*forward_prefactor
    inverse_weights=weights*inverse_prefactor
    

    nq = n_radial_points
    nl = len(l_orders)
    nlm = l_max*(l_max+2)+1

    import pyopencl as cl 
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(ctx)

    apply_weights = cl.Program(ctx, """
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
    for (int q = 0; q < nq; ++q)
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
    """).build().apply_weights

    apply_weights.set_scalar_arg_dtypes([None,None,None,np.int64,np.int64,np.int64])
        
    out_f = np.zeros((nq,nlm),dtype = complex)
    out_i = np.zeros((nq,nlm),dtype = complex)

    mf = cl.mem_flags
    w_f_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(forward_weights))
    w_i_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(inverse_weights))

    #queue = cl.CommandQueue(ctx)

    # w_f_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=forward_weights.nbytes)
    # w_i_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=inverse_weights.nbytes)
    rho_f_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=out_f.nbytes)
    rho_i_dev = cl.Buffer(ctx , cl.mem_flags.READ_WRITE, size=out_i.nbytes)
    out_f_dev = cl.Buffer(ctx , cl.mem_flags.WRITE_ONLY, size=out_f.nbytes)
    out_i_dev = cl.Buffer(ctx , cl.mem_flags.WRITE_ONLY, size=out_i.nbytes)
    #cl.enqueue_copy(queue,w_f_dev,np.ascontiguousarray(forward_weights))
    #cl.enqueue_copy(queue,w_i_dev,np.ascontiguousarray(inverse_weights))

    local_range = None
    global_range = (nq,nlm)
    log.info('start test')
    cl.enqueue_copy(queue,rho_f_dev,np.random.rand(*out_f.shape))
    apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,nq,nlm,nl)
    cl.enqueue_copy(queue, out_f,out_f_dev)
    log.info('end test')
    def zht(harmonic_coeff):
        cl.enqueue_copy(queue,rho_f_dev,harmonic_coeff)
        apply_weights(queue,global_range,local_range,out_f_dev,w_f_dev,rho_f_dev,nq,nlm,nl)
        cl.enqueue_copy(queue, out_f,out_f_dev)
        return out_f
    
    def izht(reciprocal_coeff):
        cl.enqueue_copy(queue,rho_i_dev,reciprocal_coeff)
        apply_weights(queue,global_range,local_range,out_i_dev,w_i_dev,rho_i_dev,nq,nlm,nl)
        cl.enqueue_copy(queue, out_i,out_i_dev)
        return out_i
    return zht,izht


def generate_spherical_ht_trapz_forward(grid_pair,l_orders):
    forward_prefactor=(-1.j)**(l_orders[None,None,:])
    inverse_prefactor=(1.j)**(l_orders[None,None,:])
    q = grid_pair.reziprocalGrid[:,0,0,0]
    trapz= np.trapz
    
    kernel = mLib.spherical_bessel_jnu(m,2*pi*rp*x)*x
    
    def zht(harmonic_coeff):
        return 
    def izht(reciprocal_coeff):
        return np.sum(inverse_weights*reciprocal_coeff,axis=1)
    
    return reciprocalHarmonicCoefficients


def generate_zernike_moments(pos_orders,opt,**kwargs):
    log.info('n_pos_orders={}'.format(len(pos_orders)))
    expansion_limit=opt['expansion_limit']
    #radial_points=opt['n_radial_points']+1
    radial_points=opt['n_radial_points']

    N=opt['n_radial_points']
    kk,w_gauss = roots_legendre(N)
    ids=np.argsort(kk)
    k=(kk[ids]+1)/2
    w_gauss=w_gauss[ids]
    
    p=N/2*k.copy()
    #log.info('p={}'.format(p))
    
    zernike_dict=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,k,3)
    ns = tuple(np.arange(l,expansion_limit+1,2) for l in pos_orders)
    def n_sum_gauss(l):
        n=ns[l]
        #log.info('jp = \n {}'.format(jp))
        Zk=zernike_dict[l]
        Zp=zernike_dict[l].copy()

        n_sum= np.sum((2*n+3)[:,None,None]/2*Zp[:,None,:]*Zk[:,:,None],axis=0)
        # log.info('prefactor = \n {}'.format(prefactor))
        return n_sum    
    n_sums_g= np.array(tuple(n_sum_gauss(l) for l in pos_orders))
    weights_g=np.swapaxes(np.square(k[None,:,None])*w_gauss[None,:,None]*n_sums_g,0,2)#*np.sqrt(2/np.pi) #l k p

    l_max=pos_orders.max()
    ms=np.concatenate((np.arange(l_max+1,dtype = int),-np.arange(l_max,0,-1,dtype = int)))
    def calc_moments_gauss(density):
        return tuple(np.sum(weights_g[:,:,np.abs(m):]*density[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in ms)

    ks=np.arange(N)
    ps=ks.copy()

    zernike_dict2=mLib.eval_ND_zernike_polynomials(pos_orders,expansion_limit,ks/N,3)
    ns = tuple(np.arange(l,expansion_limit+1,2) for l in pos_orders)
    def n_sum(l):
        n=ns[l]        
        #log.info('jp = \n {}'.format(jp))
        Zk=zernike_dict[l]
        Zp=zernike_dict[l].copy()

        n_sum= np.sum((2*n+3)[:,None,None]/N*Zp[:,None,:]*Zk[:,:,None],axis=0)
        # log.info('prefactor = \n {}'.format(prefactor))
        return n_sum
    n_sums= np.array(tuple(n_sum_gauss(l) for l in pos_orders))
    weights=np.swapaxes(np.square(ks[None,:,None]/N)*n_sums,0,2)#*np.sqrt(2/np.pi) #l k p
    def calc_moments(density):
        return tuple(np.sum(weights[:,:,np.abs(m):]*density[m][:,None,:l_max - np.abs(m)+1],axis=0) for m in ms)
    return calc_moments,calc_moments_gauss



#################################################
###          HT via Zernike expansion         ###
def generate_weightDict_zernike(dimension,max_order, n_radial_points,expansion_limit = -1,pi_in_q = True):
    if dimension == 2:
        pass
    elif dimension ==3:
       weight_dict = generate_weightDict_zernike_spherical(max_order, n_radial_points,expansion_limit = -1,pi_in_q = True) 
    return weight_dict

########################################
###  Discrete HT + Lin or interpol   ###

def generate_point_transforms(n_radial_points,harmonic_orders):    
    Nr=n_radial_points
    print('start calculating zeros')
    zeros=mLib.besselZeros(harmonic_orders,np.arange(1,Nr))
    jN=mLib.besselZeros(harmonic_orders,[Nr]).flatten()
    print('finished calculating zeros')
    Jj2=np.square(mLib.bessel_jnu((harmonic_orders+1)[:,None],zeros))
    denominator=np.swapaxes(Jj2,0,1)[:,:,None]*Jj2[None,:,:]*(jN**2)[None,:,None]
    jr_to_r=0
    jq_to_q=0
    print('start sum loop')
    for k in range(Nr-1):
#        print(k)
        log.info('k={}'.format(k))
        jjj=zeros[:,k,None,None]*zeros[None,:,:]
#        log.info('jjj shape = {}'.format(jjj.shape))
        jj=np.swapaxes(np.diagonal(jjj/jN[:,None,None],axis1=1,axis2=0),0,1)
#        log.info('jj shape = {}'.format(jj.shape))
        Jjj=mLib.bessel_jnu(harmonic_orders[:,None],jj)
        Jjr=mLib.bessel_jnu(harmonic_orders[:,None,None],jjj/jN[None,:,None])
        Jjq=mLib.bessel_jnu(harmonic_orders[:,None,None],jjj/jN[:,None,None])
#        J_core=4*Jjj[:,None,:]/(jN**2*Jj2[k,None,:]*Jj2[:,None,:])
        J_core=4*Jjj[:,:,None,None]/denominator[k][:,:,None,None]
        jr_to_r+=J_core*Jjr[:,None,:,:]
        jq_to_q+=J_core*Jjq[:,None,:,:]
    log.info('jr_to_r shape = {}'.format(jr_to_r.shape))
    return (jr_to_r,jq_to_q)

def generate_point_transforms2_old(n_radial_points,harmonic_orders):    
    Nr=n_radial_points
    print('start calculating zeros')
    zeros=mLib.besselZeros(harmonic_orders,np.arange(1,Nr))
    jN=mLib.besselZeros(harmonic_orders,[Nr]).flatten()
    print('finished calculating zeros')
    radi=zeros[0]/jN[0]
    qradi=zeros[0]
    Jj2=np.square(mLib.bessel_jnu((harmonic_orders+1)[:,None],zeros))
    jr_to_r=0
    jq_to_q=0
    print('start sum loop')
    for k in range(Nr-1):
#        print(k)
        
        jj=(zeros[:,k]/jN)[:,None]*zeros[:,:]
#        log.info('jjj shape = {}'.format(jjj.shape))
#        log.info('jj shape = {}'.format(jj.shape))
        Jjj=mLib.bessel_jnu(harmonic_orders[:,None],jj)
        Jjr=mLib.bessel_jnu(harmonic_orders[:,None],zeros[:,k,None]*radi[None,:])
        Jjq=mLib.bessel_jnu(harmonic_orders[:,None],zeros[:,k,None]/jN[:,None]*qradi[None,:])
#        J_core=4*Jjj[:,None,:]/(jN**2*Jj2[k,None,:]*Jj2[:,None,:])
        J_core=4*Jjj/(Jj2[:,k,None]*Jj2[:,:]*(jN**2)[:,None])
        jr_to_r+=J_core[:,:,None]*Jjr[:,None,:]
        jq_to_q+=J_core[:,:,None]*Jjq[:,None,:]
    log.info('jr_to_r shape = {}'.format(jr_to_r.shape))
    return (jr_to_r,jq_to_q)

def generate_point_transforms2(n_radial_points,harmonic_orders):    
    Nr=n_radial_points
    print('start calculating zeros')
    zeros=mLib.besselZeros(harmonic_orders,np.arange(1,Nr))
    jN=mLib.besselZeros(harmonic_orders,[Nr]).flatten()
    print('finished calculating zeros')
    radi=zeros[0]/jN[0]
    qradi=zeros[0]
    Jj2=np.square(mLib.bessel_jnu((harmonic_orders+1)[:,None],zeros))
    jr_to_r=0
    jq_to_q=0
    print('start sum loop')
    for k in range(Nr-1):
        jj0=(zeros[:,k,None]*zeros[None,0,:])/jN[0]
        j0j=(zeros[:,k,None]*zeros[None,0,:])/jN[:,None]
        jj=(zeros[:,k]/jN)[:,None]*zeros[:,:]
#        log.info('jjj shape = {}'.format(jjj.shape))
#        log.info('jj shape = {}'.format(jj.shape))
        Jjj=mLib.bessel_jnu(harmonic_orders[:,None],jj)
        Jjr=mLib.bessel_jnu(harmonic_orders[:,None],jj0)
        Jjq=mLib.bessel_jnu(harmonic_orders[:,None],j0j)
#        J_core=4*Jjj[:,None,:]/(jN**2*Jj2[k,None,:]*Jj2[:,None,:])
        J_core=4*Jjj/(Jj2[:,k,None]*Jj2[:,:]*(jN**2)[:,None])
        jr_to_r+=J_core[:,:,None]*Jjr[:,None,:]
        jq_to_q+=J_core[:,:,None]*Jjq[:,None,:]
    log.info('jr_to_r shape = {}'.format(jr_to_r.shape))
    return (jr_to_r,jq_to_q)

def generate_dht_lin(maxR,point_transforms,harmonic_orders,n_radial_points):
    jr_to_r,jq_to_q=point_transforms
    n_orders=len(harmonic_orders)
    jr_to_jq,jq_to_jr=mLib.generate_dht_2d(maxR,n_radial_points-1,harmonic_orders)
    def dht(diag_coeff):
        
        diag_reciprocal_coeff=jr_to_jq(diag_coeff)
        print('diag r coeff shape ={}'.format(diag_reciprocal_coeff.shape))        
        reciprocal_harmonic_coeff=np.sum(jq_to_q*diag_reciprocal_coeff[:,:,None,None],axis=1)
        print('r coeff shape ={}'.format(reciprocal_harmonic_coeff.shape))
        return reciprocal_harmonic_coeff
    def idht(diag_reciprocal_coeff):
        diag_coeff=jq_to_jr(diag_reciprocal_coeff)
        return diag_coeff
    return dht,idht

def generate_dht_lin_old(maxR,point_transforms,harmonic_orders,n_radial_points):
    jr_to_r,jq_to_q=point_transforms
    n_orders=len(harmonic_orders)
    jr_to_jq,jq_to_jr=mLib.generate_dht_2d(maxR,n_radial_points-1,harmonic_orders)
    def dht(harmonic_coeff):
   #     jr=np.sum(r_to_jr*harmonic_coeff[:,:,None],axis=0)
        jq=jr_to_jq(harmonic_coeff)
        return jq
    def idht(diag_reciprocal_coeff):
        diag_coeff=jq_to_jr(diag_reciprocal_coeff)
        return diag_coeff
    return dht,idht


###########################
###HT via Sin/Cos + adaptive Quad###
def generateHT_SinCos(FTGridPair,weightsDict):    
#    log.info('FTGridpair shapes= real {} reciprocal{}'.format(FTGridPair.realGrid.shape,FTGridPair.reciprocalGrid.shape))


    PNAS=('PNAS'==weightsDict['mode'])
    positive_harm_orders=weightsDict['posHarmOrders']
    

    params=extractGridParameters(FTGridPair)
    nRadialSteps=params['nRadialPoints']
    nAngularSteps=params['nAngularPoints']
    reciprocalCutOff=params['reciprocalCutOff']
    
    positive_weights=weightsDict['weights']
    positive_weights=np.swapaxes(positive_weights,0,2) # w(m,n,n') -> w(n',n,m)
    positive_weights=np.swapaxes(positive_weights,0,1) # w(n',n,m) -> w(n,n',m)

    n_positive_orders=len(positive_harm_orders)

        
    if positive_harm_orders[0]==0:
        used_harm_orders=np.concatenate((positive_harm_orders,-1*positive_harm_orders[:0:-1]))
        negative_order_weights=positive_weights[:,:,n_positive_orders:0:-1]*((-1.)**used_harm_orders[n_positive_orders:])
    else:
        used_harm_orders=np.concatenate((positive_harm_orders,-1*positive_harm_orders[::-1]))
        negative_order_weights=positive_weights[:,:,n_positive_orders::-1]*((-1.)**used_harm_orders[n_positive_orders:])
    nOrders=len(used_harm_orders)
                    
    weights=np.zeros((nRadialSteps,nRadialSteps,nOrders))
    weights[:,:,:n_positive_orders]=positive_weights
    weights[:,:,n_positive_orders:]=negative_order_weights
    

    try:
        if PNAS:
            assert reciprocalCutOff==weightsDict['maxQ'],'fourier transform grid maxQ({}) does not match hankel trf weight maxQ({}) '.format(reciprocalCutOff,weightsDict['maxQ'])
    except AssertionError as e:
        log.error(e)
        raise
    
    realCutOff=params['realCutOff']


    forwardPrefactor=(-1.j)**used_harm_orders*realCutOff**2
    forwardPrefactorPNAS=(-1.j)**used_harm_orders
    inversePrefactorPNAS=(1.j)**used_harm_orders*reciprocalCutOff**2
    inversePrefactor=(1.j)**used_harm_orders*reciprocalCutOff**2

    '''        
    try:
        nIndependendWeights=weights.shape[0]
        assert nIndependendWeights==maxOrder+1,'Error: weights do not fit with given maximal Order. maxOrder+1={} weight orders ={}'.format(maxOrder+1,nIndependendWeights)
        assert nOrders<=nAngularSteps,'Error: Specified number of harmonic Orders ({}) exeeds the number of angular steps ({}) of input coefficients.'.format(nOrders,nAngularSteps)
    except AssertionError as error:
        log.error(error)
        raise
    '''

    if PNAS:
        def ht_forward(harmonicCoefficients):
            reciprocalHarmonicCoefficients=hankelTransform_SinCos(harmonicCoefficients,weights,used_harm_orders,forwardPrefactorPNAS)
            return reciprocalHarmonicCoefficients
        def ht_inverse(reciprocalHarmonicCoefficients):        
            harmonicCoefficients=hankelTransform_SinCos(reciprocalHarmonicCoefficients,weights,used_harm_orders,inversePrefactorPNAS)
            return harmonicCoefficients
    else:
        def ht_forward(harmonicCoefficients):
            reciprocalHarmonicCoefficients=hankelTransform_SinCos(harmonicCoefficients,weights,used_harm_orders,forwardPrefactor)
            return reciprocalHarmonicCoefficients
        def ht_inverse(reciprocalHarmonicCoefficients):        
            harmonicCoefficients=hankelTransform_SinCos(reciprocalHarmonicCoefficients,weights,used_harm_orders,inversePrefactor)
            return harmonicCoefficients
    return ht_forward,ht_inverse

def generateHT_SinCos_old(FTGridPair,weightsDict):
    weights=weightsDict['weights']
    PNAS=('PNAS'==weightsDict['mode'])
    posHarmOrders=weightsDict['posHarmOrders']
    
    if posHarmOrders[0]==0:
        harmOrders=np.concatenate((posHarmOrders,-1*posHarmOrders[:0:-1]))
    else:
        harmOrders=np.concatenate((posHarmOrders,-1*posHarmOrders[::-1]))
    nOrders=len(harmOrders)
    
    params=extractGridParameters(FTGridPair)
    nRadialSteps=params['nRadialPoints']
    nAngularSteps=params['nAngularPoints']
    reciprocalCutOff=params['reciprocalCutOff']
    try:
        if PNAS:
            assert reciprocalCutOff==weightsDict['maxQ'],'fourier transform grid maxQ({}) does not match hankel trf weight maxQ({}) '.format(reciprocalCutOff,weightsDict['maxQ'])
    except AssertionError as e:
        log.error(e)
        raise
    
    realCutOff=params['realCutOff']


    forwardPrefactor=(-1.j)**harmOrders*realCutOff**2
    forwardPrefactorPNAS=(-1.j)**harmOrders
    inversePrefactorPNAS=(1.j)**harmOrders*reciprocalCutOff**2
    inversePrefactor=(1.j)**harmOrders*reciprocalCutOff**2

    '''        
    try:
        nIndependendWeights=weights.shape[0]
        assert nIndependendWeights==maxOrder+1,'Error: weights do not fit with given maximal Order. maxOrder+1={} weight orders ={}'.format(maxOrder+1,nIndependendWeights)
        assert nOrders<=nAngularSteps,'Error: Specified number of harmonic Orders ({}) exeeds the number of angular steps ({}) of input coefficients.'.format(nOrders,nAngularSteps)
    except AssertionError as error:
        log.error(error)
        raise
    '''

    if PNAS:
        def ht_forward(harmonicCoefficients):
            reciprocalHarmonicCoefficients=hankelTransform_SinCos_old(harmonicCoefficients,weights,harmOrders,forwardPrefactorPNAS)
            return reciprocalHarmonicCoefficients
        def ht_inverse(reciprocalHarmonicCoefficients):        
            harmonicCoefficients=hankelTransform_SinCos_old(reciprocalHarmonicCoefficients,weights,harmOrders,inversePrefactorPNAS)
            return harmonicCoefficients
    else:
        def ht_forward(harmonicCoefficients):
            reciprocalHarmonicCoefficients=hankelTransform_SinCos_old(harmonicCoefficients,weights,harmOrders,forwardPrefactor)
            return reciprocalHarmonicCoefficients
        def ht_inverse(reciprocalHarmonicCoefficients):        
            harmonicCoefficients=hankelTransform_SinCos_old(reciprocalHarmonicCoefficients,weights,harmOrders,inversePrefactor)
            return harmonicCoefficients
    return ht_forward,ht_inverse
    
###quadrature weights###
def generateQuadratureWeights_SinCos(harmOrders,radialIndices,nRadialSteps,useAngularFrequencies):
    jnu=mLib.bessel_jnu
    if useAngularFrequencies:
        ftCoefficient=1
    else:
        ftCoefficient=2*np.pi
        
    def integrandEven(r,harmonicOrder,k,radialIndex1):
        value=jnu(harmonicOrder,ftCoefficient*radialIndex1*r)*r
        return value
    def integrandOdd(r,harmonicOrder,k,radialIndex1):
        value=jnu(harmonicOrder,ftCoefficient*radialIndex1*r)*r
        return value
    integrands=[integrandEven,integrandOdd]
    getWeightFunctions_SinCos=pyLib.plugVariableIntoFunction(getWeightFunctions,[integrands,2])
    
    getWeightFuncRoutine=getWeightFunctions_SinCos
    weights=generateQuadratureWeights(harmOrders,radialIndices,nRadialSteps,getWeightFuncRoutine,useAngularFrequencies=useAngularFrequencies)
    #apply c_n'
    weights[:,0]*=.5
    return weights

def generateQuadratureWeights_PNAS(harmOrders,radialIndices,totalRadialSteps,reciprocalCutOff,calc_offset,**kwargs):
    integrands=get_integrands_PNAS(totalRadialSteps,reciprocalCutOff,calc_offset)
    getWeightFunctions_PNAS=pyLib.plugVariableIntoFunction(getWeightFunctions,[integrands,2])
    
    getWeightFuncRoutine=getWeightFunctions_PNAS
    weights,errors=generateQuadratureWeights(harmOrders,radialIndices,totalRadialSteps,getWeightFuncRoutine)
    #apply c_n'
    weights[:,0]*=.5
    errors[:,0]*=.5

    weights/=calc_offset
    errors/=calc_offset
    results=np.concatenate((np.expand_dims(weights,-1),np.expand_dims(errors,-1)),axis=-1)
    log.info('result shape ={}'.format(results.shape))
    return results
def get_integrands_PNAS(n_radial_points,reciprocal_cut_off,calc_offset):
    Nr=n_radial_points
    Q=reciprocal_cut_off
    pi=np.pi
    cos=np.cos
    sin=np.sin
    # x = integration variable
    # o = harmonic order
    # r1 = first radial index
    def integrand_even(x,o,k,r1):
        value=mLib.bessel_jnu(o,2*pi*r1/Nr*Q*x)*x*cos(pi*k*x)*calc_offset
        return value
    def integrand_odd(x,o,k,r1):
        value=mLib.bessel_jnu(o,2*pi*r1/Nr*Q*x)*x*sin(pi*k*x)*calc_offset
        return value    
    return [integrand_even,integrand_odd]

def generateQuadratureWeightsOld(harmOrders,nRadialSteps,routines):
    radialIndices=np.arange(nRadialSteps)
        
    getWeightFunctions=routines[0]
    getAxisWeights=routines[1]

    prefix=4*np.pi/nRadialSteps

    quadratureWeights=np.zeros((len(harmOrders),nRadialSteps,nRadialSteps))
    for index,order in enumerate(harmOrders):
        #        log.info('harmonic order ={}'.format(harmonicOrder))
        weightIntegral,trig=getWeightFunctions(order,nRadialSteps)
        for radialIndex1 in radialIndices:
            #                log.info('calculating weights for m={} and n={}'.format(harmonicOrder,radialIndex1))
            if order!=0 and radialIndex1==0:
                quadratureWeights[index,radialIndex1,:]=np.zeros(nRadialSteps)
            else:
                htValues=np.fromiter(map(weightIntegral,radialIndices,repeat(radialIndex1)),dtype=np.float)                
                quadratureWeights[index,radialIndex1,:]=getAxisWeights(htValues,trig,radialIndices)
#            quadratureWeights[index,radialIndex1,:]=getAxisWeightsOld(hankelTrf,trig,radialIndex1,nRadialSteps)

def generateQuadratureWeights(harmOrders,radialIndices1,nRadialSteps,getWeightFuncRoutine,useAngularFrequencies=False):
    radialIndices=np.arange(nRadialSteps)

    getWeightFunctions=getWeightFuncRoutine

    if useAngularFrequencies:
        prefix=2/nRadialSteps
    else:
        prefix=4*np.pi/nRadialSteps

    quadratureWeights=np.zeros((len(harmOrders),nRadialSteps))
    weight_errors=quadratureWeights.copy()

    for index,order,radialIndex1 in zip(np.arange(harmOrders.shape[0]),harmOrders,radialIndices1):
        #        log.info('index={} order={} radialIndex1={}'.format(index,order,radialIndex1))
        if order!=0 and radialIndex1==0:
            quadratureWeights[index,:]=np.zeros(nRadialSteps)
            weight_errors[index,:]=quadratureWeights[index,:].copy()
        else:
            if radialIndex1%100==0:
                log.debug('order={},radialIndex1={}'.format(order,radialIndex1))
            weightIntegral,trig=getWeightFunctions(order,nRadialSteps)
            htValues=np.array(tuple(map(weightIntegral,np.arange(nRadialSteps),repeat(radialIndex1))))
            quadratureWeights[index,:]=getAxisWeights(htValues[...,0],trig,radialIndices,order)
            weight_errors[index,:]=getAxisWeights(htValues[...,1],trig,radialIndices,order)
            
    quadratureWeights*=prefix
#    log.info('quadrature weights part shape:{}'.format(quadratureWeights.shape))
    return quadratureWeights,weight_errors

##
def getAxisWeights(htValues,trig,radialIndices,order):
    def getSum(radialIndex2):
        if order!=0 and radialIndex2==0:
            sumValue=0
        else:
            summands=htValues*trig(radialIndices,radialIndex2)
            summands[0]*=.5
            sumValue=np.sum(summands)
        return sumValue

    #axisWeights=np.array(tuple(np.sum(sumGenerator(htValues,trig,radialIndex2)) for radialIndex2 in radialIndices))
#    axisWeights=np.array(tuple(getSum(htValues,trig,radialIndex2,radialIndices) for radialIndex2 in radialIndices))
    axisWeights=np.fromiter(map(getSum,radialIndices),dtype=np.float)
    return axisWeights
        
##
def getWeightFunctions(harmonicOrder,nRadialSteps,integrands):
    #sin/cos(pi*k*r) factor of the integrand is added in weightIntegralEven/weightIntegralOdd for performace reasons
    def weightIntegralEven(k,radialIndex1):
        value=spIntegrate.quad(integrands[0],0,1,args=(harmonicOrder,k,radialIndex1),epsrel=1e-9,limit=1000)
        return value
    
    def weightIntegralOdd(k,radialIndex1):
        value=spIntegrate.quad(integrands[1],0,1,args=(harmonicOrder,k,radialIndex1),epsrel=1e-9,limit=1000)
        return value
    
    def trigEven(k,radialIndex2):
        value=np.cos(np.pi*k*radialIndex2/nRadialSteps)
        return value
    def trigOdd(k,radialIndex2):
        value=np.sin(np.pi*k*radialIndex2/nRadialSteps)
        return value
    
    orderIsEven= (harmonicOrder%2==0)
    if orderIsEven:
        weightIntegral=weightIntegralEven
        trig=trigEven
    else:
        weightIntegral=weightIntegralOdd
        trig=trigOdd
    return weightIntegral,trig

##

###hankel transform###
def hankelTransform_SinCos(harmonicCoefficients,weights,harmOrders,prefactor):
    significantHarmonicCoeff=harmonicCoefficients[:,harmOrders]
   # log.info('harmOrders={}'.format(harmOrders))
#    log.info('prefactor={}'.format(prefactor))
#    log.info('weights shape={}'.format(weights.shape))
#    log.info('harmonic coefficients shape ={}'.format(significantHarmonicCoeff.shape))

#    log.info('sum for m=n=1 : {}'.format(np.sum(summandGenerator(weights,significantHarmonicCoeff,forwardFactor))[1,1]))
    transformedArrayPart=prefactor*getSum(weights,significantHarmonicCoeff,harmOrders)
    transformedArray=np.zeros(harmonicCoefficients.shape,dtype=transformedArrayPart.dtype)

    transformedArray[:,harmOrders]=transformedArrayPart
    #    log.info('transformedArray shape ={}'.format(transformedArray.shape ))
#    transformedGrid=Grid(transformedArray,gridType=harmonicCoefficients.gridType)
    #    log.info('harmonic coeff ={}'.format(harmonicCoefficients))
    return transformedArray
    
def getSum(weights,harmonicCoefficients,harmOrders):
    summands=harmonicCoefficients*weights
#    log.info('coefficients at n prime =0 : {}'.format(harmonicCoefficients[0,:]))
    sumValue=np.sum(summands,axis=1)
    return sumValue
    


def hankelTransform_SinCos_old(harmonicCoefficients,weights,harmOrders,prefactor):
    significantHarmonicCoeff=harmonicCoefficients.array[:,harmOrders]
#    log.info('harmOrders={}'.format(harmOrders))
#    log.info('prefactor={}'.format(prefactor))
#    log.info('weights={}'.format(weights[1,1,:]))

#    log.info('sum for m=n=1 : {}'.format(np.sum(summandGenerator(weights,significantHarmonicCoeff,forwardFactor))[1,1]))
    transformedArrayPart=prefactor*getSum_old(weights,significantHarmonicCoeff,harmOrders)
    transformedArray=np.zeros(harmonicCoefficients.array.shape,dtype=np.complex)

    transformedArray[:,harmOrders]=transformedArrayPart
    #    log.info('transformedArray shape ={}'.format(transformedArray.shape ))
    transformedGrid=Grid(transformedArray,gridType=harmonicCoefficients.gridType)
    #    log.info('harmonic coeff ={}'.format(harmonicCoefficients))
    return transformedGrid
    
def getSum_old(weights,harmonicCoefficients,harmOrders):
    nPosWeights=len(weights)
    nOrders=len(harmOrders)
    weights=np.swapaxes(weights,0,2) # w(m,n,n') -> w(n',n,m)
    weights=np.swapaxes(weights,0,1) # w(n',n,m) -> w(n,n',m)
    nRadialSteps=len(harmonicCoefficients)
    
    summands=np.zeros((nRadialSteps,nRadialSteps,nOrders),dtype=np.complex)
    summands[:,:,:nPosWeights]=harmonicCoefficients[:,:nPosWeights]*weights
#    log.info('coefficients at n prime =0 : {}'.format(harmonicCoefficients[0,:]))

    negHarmWeights=weights[:,:,nPosWeights:0:-1]*((-1.)**harmOrders[nPosWeights:])
    summands[:,:,nPosWeights:]=harmonicCoefficients[:,nPosWeights:]*negHarmWeights
    sumValue=np.sum(summands,axis=1)
    return sumValue



### PNAS Hankel transform###
def generate_ht_spherical_SinCos(weights_dict):    
    weights=weights_dict['weights']
    ls=np.arange(weights.shape[0])    
    N=weights.shape[1]
    Q=weights_dict['maxQ']
    log.info('maxQ = {}'.format(Q))
    
    weights=np.moveaxis(weights,0,2) # w(l,n,n') -> w(n,n',l)
    
    forward_prefactor=(-1.j)**ls*2
    inverse_prefactor=(1.j)**ls*Q**3*2

    forward_weights=forward_prefactor[None,None,:]*weights
    inverse_weights=inverse_prefactor[None,None,:]*weights
    ms = np.concatenate((ls,-ls[:0:-1]))
    def ht_forward(harm_coeff):
        r_harm_coeff=tuple(np.sum(forward_weights[...,np.abs(m):]*harm_coeff[m][None,:,:],axis=1) for m in ms)
        return r_harm_coeff
    def ht_inverse(r_harm_coeff):
        harm_coeff=tuple(np.sum(inverse_weights[...,np.abs(m):]*r_harm_coeff[m][None,:,:],axis=1) for m in ms)
        return harm_coeff
    return ht_forward,ht_inverse

###quadrature weights Spherical###
def generate_weights_spherical_PNAS(harmOrders,radialIndices,totalRadialSteps,reciprocalCutOff,calc_offset,**kwargs):
    integrands=get_spherical_integrands_PNAS(totalRadialSteps,reciprocalCutOff,calc_offset)
    get_functions_routine=pyLib.plugVariableIntoFunction(get_weight_functions_spherical_PNAS,[integrands,2])
    
    getWeightFuncRoutine=get_functions_routine
    weights,errors=generate_quad_weights(harmOrders,radialIndices,totalRadialSteps,getWeightFuncRoutine)
    #apply c_n'
    weights[:,0]*=.5
    errors[:,0]*=.5

    weights/=calc_offset
    errors/=calc_offset
    results=np.concatenate((np.expand_dims(weights,-1),np.expand_dims(errors,-1)),axis=-1)
    log.info('result shape ={}'.format(results.shape))
    return results
def get_spherical_integrands_PNAS(n_radial_points,reciprocal_cut_off,calc_offset):
    Nr=n_radial_points
    Q=reciprocal_cut_off
    pi=np.pi
    cos=np.cos
    sin=np.sin
    # x = integration variable
    # o = harmonic order
    # r1 = first radial index
    def integrand_even(x,o,k,r1):
        value=mLib.spherical_bessel_jnu(o,2*pi*r1/Nr*Q*x)*(x**2)*cos(pi*k*x)*calc_offset
        return value
    def integrand_odd(x,o,k,r1):
        value=mLib.spherical_bessel_jnu(o,2*pi*r1/Nr*Q*x)*(x**2)*sin(pi*k*x)*calc_offset
        return value    
    return [integrand_even,integrand_odd]

def generateQuadratureWeightsOld(harmOrders,nRadialSteps,routines):
    radialIndices=np.arange(nRadialSteps)
        
    getWeightFunctions=routines[0]
    getAxisWeights=routines[1]

    prefix=4*np.pi/nRadialSteps

    quadratureWeights=np.zeros((len(harmOrders),nRadialSteps,nRadialSteps))
    for index,order in enumerate(harmOrders):
        #        log.info('harmonic order ={}'.format(harmonicOrder))
        weightIntegral,trig=getWeightFunctions(order,nRadialSteps)
        for radialIndex1 in radialIndices:
            #                log.info('calculating weights for m={} and n={}'.format(harmonicOrder,radialIndex1))
            if order!=0 and radialIndex1==0:
                quadratureWeights[index,radialIndex1,:]=np.zeros(nRadialSteps)
            else:
                htValues=np.fromiter(map(weightIntegral,radialIndices,repeat(radialIndex1)),dtype=np.float)                
                quadratureWeights[index,radialIndex1,:]=getAxisWeights(htValues,trig,radialIndices)
#            quadratureWeights[index,radialIndex1,:]=getAxisWeightsOld(hankelTrf,trig,radialIndex1,nRadialSteps)

def generate_quad_weights(harmOrders,radialIndices1,nRadialSteps,getWeightFuncRoutine,useAngularFrequencies=False):
    radialIndices=np.arange(nRadialSteps)

    getWeightFunctions=getWeightFuncRoutine

    if useAngularFrequencies:
        prefix=2/nRadialSteps
    else:
        prefix=4*np.pi/nRadialSteps

    quadratureWeights=np.zeros((len(harmOrders),nRadialSteps))
    weight_errors=quadratureWeights.copy()

    for index,order,radialIndex1 in zip(np.arange(harmOrders.shape[0]),harmOrders,radialIndices1):
        #        log.info('index={} order={} radialIndex1={}'.format(index,order,radialIndex1))
        if order!=0 and radialIndex1==0:
            quadratureWeights[index,:]=np.zeros(nRadialSteps)
            weight_errors[index,:]=quadratureWeights[index,:].copy()
        else:
            if radialIndex1%100==0:
                log.info('order={},radialIndex1={}'.format(order,radialIndex1))
            weightIntegral,trig=getWeightFunctions(order,nRadialSteps)
            htValues=np.array(tuple(map(weightIntegral,np.arange(nRadialSteps),repeat(radialIndex1))))
            quadratureWeights[index,:]=getAxisWeights(htValues[...,0],trig,radialIndices,order)
            weight_errors[index,:]=getAxisWeights(htValues[...,1],trig,radialIndices,order)
            
    quadratureWeights*=prefix
#    log.info('quadrature weights part shape:{}'.format(quadratureWeights.shape))
    return quadratureWeights,weight_errors

##
def getAxisWeights(htValues,trig,radialIndices,order):
    def getSum(radialIndex2):
        if order!=0 and radialIndex2==0:
            sumValue=0
        else:
            summands=htValues*trig(radialIndices,radialIndex2)
            summands[0]*=.5
            sumValue=np.sum(summands)
        return sumValue

    #axisWeights=np.array(tuple(np.sum(sumGenerator(htValues,trig,radialIndex2)) for radialIndex2 in radialIndices))
#    axisWeights=np.array(tuple(getSum(htValues,trig,radialIndex2,radialIndices) for radialIndex2 in radialIndices))
    axisWeights=np.fromiter(map(getSum,radialIndices),dtype=np.float)
    return axisWeights
        
##
def get_weight_functions_spherical_PNAS(harmonicOrder,nRadialSteps,integrands):
    #sin/cos(pi*k*r) factor of the integrand is added in weightIntegralEven/weightIntegralOdd for performace reasons
    def weightIntegralEven(k,radialIndex1):
        value=spIntegrate.quad(integrands[0],0,1,args=(harmonicOrder,k,radialIndex1),epsrel=1e-9,limit=1000)
        return value
    
    def weightIntegralOdd(k,radialIndex1):
        value=spIntegrate.quad(integrands[1],0,1,args=(harmonicOrder,k,radialIndex1),epsrel=1e-9,limit=1000)
        return value
    
    def trigEven(k,radialIndex2):
        value=np.cos(np.pi*k*radialIndex2/nRadialSteps)
        return value
    def trigOdd(k,radialIndex2):
        value=np.sin(np.pi*k*radialIndex2/nRadialSteps)
        return value
    
    orderIsEven= (harmonicOrder%2==0)
    if orderIsEven:
        weightIntegral=weightIntegralEven
        trig=trigEven
    else:
        weightIntegral=weightIntegralOdd
        trig=trigOdd
    return weightIntegral,trig

##

###hankel transform###
def hankel_transform_spherical_SinCos(harmonicCoefficients,weights,ls,prefactor):
    significantHarmonicCoeff=harmonicCoefficients[:,harmOrders]
   # log.info('harmOrders={}'.format(harmOrders))
#    log.info('prefactor={}'.format(prefactor))
#    log.info('weights shape={}'.format(weights.shape))
#    log.info('harmonic coefficients shape ={}'.format(significantHarmonicCoeff.shape))

#    log.info('sum for m=n=1 : {}'.format(np.sum(summandGenerator(weights,significantHarmonicCoeff,forwardFactor))[1,1]))
    transformedArrayPart=prefactor*getSum(weights,significantHarmonicCoeff,harmOrders)
    transformedArray=np.zeros(harmonicCoefficients.shape,dtype=transformedArrayPart.dtype)

    transformedArray[:,harmOrders]=transformedArrayPart
    #    log.info('transformedArray shape ={}'.format(transformedArray.shape ))
#    transformedGrid=Grid(transformedArray,gridType=harmonicCoefficients.gridType)
    #    log.info('harmonic coeff ={}'.format(harmonicCoefficients))
    return transformedArray
    
def getSum(weights,harmonicCoefficients,harmOrders):
    summands=harmonicCoefficients*weights
#    log.info('coefficients at n prime =0 : {}'.format(harmonicCoefficients[0,:]))
    sumValue=np.sum(summands,axis=1)
    return sumValue
    


def hankelTransform_SinCos_old(harmonicCoefficients,weights,harmOrders,prefactor):
    significantHarmonicCoeff=harmonicCoefficients.array[:,harmOrders]
#    log.info('harmOrders={}'.format(harmOrders))
#    log.info('prefactor={}'.format(prefactor))
#    log.info('weights={}'.format(weights[1,1,:]))

#    log.info('sum for m=n=1 : {}'.format(np.sum(summandGenerator(weights,significantHarmonicCoeff,forwardFactor))[1,1]))
    transformedArrayPart=prefactor*getSum_old(weights,significantHarmonicCoeff,harmOrders)
    transformedArray=np.zeros(harmonicCoefficients.array.shape,dtype=np.complex)

    transformedArray[:,harmOrders]=transformedArrayPart
    #    log.info('transformedArray shape ={}'.format(transformedArray.shape ))
    transformedGrid=Grid(transformedArray,gridType=harmonicCoefficients.gridType)
    #    log.info('harmonic coeff ={}'.format(harmonicCoefficients))
    return transformedGrid
    
def getSum_old(weights,harmonicCoefficients,harmOrders):
    nPosWeights=len(weights)
    nOrders=len(harmOrders)
    weights=np.swapaxes(weights,0,2) # w(m,n,n') -> w(n',n,m)
    weights=np.swapaxes(weights,0,1) # w(n',n,m) -> w(n,n',m)
    nRadialSteps=len(harmonicCoefficients)
    
    summands=np.zeros((nRadialSteps,nRadialSteps,nOrders),dtype=np.complex)
    summands[:,:,:nPosWeights]=harmonicCoefficients[:,:nPosWeights]*weights
#    log.info('coefficients at n prime =0 : {}'.format(harmonicCoefficients[0,:]))

    negHarmWeights=weights[:,:,nPosWeights:0:-1]*((-1.)**harmOrders[nPosWeights:])
    summands[:,:,nPosWeights:]=harmonicCoefficients[:,nPosWeights:]*negHarmWeights
    sumValue=np.sum(summands,axis=1)
    return sumValue
