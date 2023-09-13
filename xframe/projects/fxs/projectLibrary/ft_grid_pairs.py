import numpy as np
from scipy.special import roots_legendre
import logging

log=logging.getLogger('root')

from xframe.library import mathLibrary as mLib
from xframe.library.mathLibrary import  polar_spherical_dft_reciprocity_relation_radial_cutoffs
from xframe.library import pythonLibrary as pyLib
from xframe.library.pythonLibrary import DictNamespace
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import GridFactory
from xframe.library.gridLibrary import uniformGrid_func
from .classes import FTGridPair
from xframe import settings
from .misk import _get_reciprocity_coefficient



def max_order_from_n_angular_steps(dimensions,n_phi,anti_aliazing_degree=2):
    '''
    max_order for highest power of 2 smaller than n_phi
    '''
    n_phi = 2**int(np.log2(n_phi))
    if dimensions == 2:
        max_order = (n_phi -1)//2
    elif dimensions == 3:
        N = anti_aliazing_degree
        max_order = n_phi//(N+1)
    return max_order

def n_angular_step_from_max_order(dimensions,max_order,anti_aliazing_degree=2):
    dim = dimensions
    max_order = max_order
    size_dict={}
    if dim == 2:
        n_phi = 2**(int(np.log2(max_order*2+1))+1)
        size_dict['n_phi'] = n_phi
    elif dim == 3 :
        N = anti_aliazing_degree
        n_phi = 2**(int(np.log2((N+1)*max_order)) + 1)
        n_theta = n_phi//2
        size_dict['n_phi'] = n_phi
        size_dict['n_phi'] = n_theta


### FT grid Pairs
def polarFFTGridPair_Bessel_uniform(realCutOff,N1,N2):
    def reciprocalGrid():
        besselZeros=np.arange(1,N1,1)
        halfBesselOrders=np.arange(np.floor(N2/2)+1)
        besselOrders=np.concatenate((halfBesselOrders[:-1],-1*halfBesselOrders[1:]))
  #      log.info('besselOrders shape = {}'.format(besselOrders))                
        radialPositions=mLib.besselZeros(besselOrders,besselZeros)
        
        radialPositions=radialPositions.flatten()/realCutOff
#        log.info('radial Positions shape = {}'.format(radialPositions.shape))
        polarNodes=[2*np.pi*besselOrders/N2]
        repeatedPolarNodes=np.repeat(polarNodes,N1-1,axis=0).flatten()
#        log.info('polar nodes shape ={}'.format(repeatedPolarNodes.shape))
        gridArray=np.stack((radialPositions,repeatedPolarNodes),axis=1).reshape(N1-1,N2,2)
        gridArray=pyLib.getArrayOfArray(gridArray)
        gridShape=[np.array([N1-1]),np.full(N1-1,N2)]
        grid=NestedArray(gridArray,1)
        return grid

    def realGrid():
        besselZeros=np.arange(1,N1+1,1)
        halfBesselOrders=np.arange(np.floor(N2/2)+1)
        besselOrders=np.concatenate((halfBesselOrders[:-1],-1*halfBesselOrders[1:]))
#        log.info('besselOrders = {}'.format(besselOrders))
        zeros=mLib.besselZeros(besselOrders,besselZeros)
#        log.info('zeros={}'.format(zeros))
        zeros.shape=(N2,N1)
        zeros=zeros.T
#        log.info('zeros shaped ={}'.format(zeros))
        #       log.info('zeros={}'.format(zeros))
        denominators=zeros[:-1,:].flatten()*realCutOff
#        log.info('denominators={}'.format(denominators))
        divisors=np.repeat([zeros[-1,:].flatten()],N1-1,axis=0).flatten()
#        log.info('divisors={}'.format(divisors))
        radialPositions=denominators/divisors
#        log.info('radial Positions ={}'.format(radialPositions))
        radialPositions=radialPositions.flatten()
        polarNodes=[2*np.pi*besselOrders/N2]
        repeatedPolarNodes=np.repeat(polarNodes,N1-1,axis=0).flatten()

        gridArray=np.stack((radialPositions,repeatedPolarNodes),axis=1).reshape(N1-1,N2,2)
        gridArray=pyLib.getArrayOfArray(gridArray)
        gridShape=[np.array([N1-1]),np.full(N1-1,N2)]        
        grid=NestedArray(gridArray,1)
#        log.info('grid array={}'.format(grid.array))
        return grid

    ftGridPair=FTGridPair(realGrid(),reciprocalGrid())
    return ftGridPair

def polar_bessel(realCutOff,N1,N2):
    bessel_zeros=np.arange(1,N1+1)
    max_order=int((N2-N2%2)/2)
    orders=np.concatenate((np.arange(max_order+1),-1*np.arange(max_order+N2%2)[:0:-1]))
    zeros=mLib.besselZeros(orders,bessel_zeros)    
#    zeros[1:]=zeros[1:]/(np.abs(orders[1:,None])*np.pi/2)-0.86 #min for 50:
#    zeros[1:]=zeros[1:]/(np.abs(orders[1:,None])*np.pi/2)-0.72 #smallest center
#    zeros[1:]=zeros[1:]-np.abs(orders[1:,None])*np.pi/2*0.95 # q min for 50
    def phi_func():
        return orders*np.pi/max_order
    
    def reciprocalGrid():
        def q_func(*args):
            return (zeros)/realCutOff
            
        reciprocal_grid=GridFactory.construct_grid('uniform_dependent',(phi_func,q_func))
        reciprocal_grid.array=np.swapaxes(reciprocal_grid.array,0,1)[...,::-1]
        return reciprocal_grid

    def realGrid():
        def r_func(*args):
            #return (zeros*realCutOff)/zeros[:,-1,None]
            return (zeros)
        real_grid=GridFactory.construct_grid('uniform_dependent',(phi_func,r_func))
        real_grid.array=np.swapaxes(real_grid.array,0,1)[...,::-1]
        return real_grid

    ftGridPair=FTGridPair(realGrid(),reciprocalGrid())
    return ftGridPair

def polar_bessel2(realCutOff,N1,N2):
    bessel_zeros=np.arange(1,N1+1)
    max_order=int((N2-N2%2)/2)
    orders=np.concatenate((np.arange(max_order+1),-1*np.arange(max_order+N2%2)[:0:-1]))
    zeros=mLib.besselZeros(orders,bessel_zeros)    
#    zeros[1:]=zeros[1:]/(np.abs(orders[1:,None])*np.pi/2)-0.86 #min for 50:
#    zeros[1:]=zeros[1:]/(np.abs(orders[1:,None])*np.pi/2)-0.72 #smallest center
#    zeros[1:]=zeros[1:]-np.abs(orders[1:,None])*np.pi/2*0.95 # q min for 50
    def phi_func():
        return orders*np.pi/max_order
    
    def reciprocalGrid():
        def q_func(*args):
            return (zeros)/realCutOff
            
        reciprocal_grid=GridFactory.construct_grid('uniform',(phi_func,q_func))
        reciprocal_grid.array=np.swapaxes(reciprocal_grid.array,0,1)[...,::-1]
        return reciprocal_grid

    def realGrid():
        def r_func(*args):
            #return (zeros*realCutOff)/zeros[:,-1,None]
            return (zeros)
        real_grid=GridFactory.construct_grid('uniform',(phi_func,r_func))
        real_grid.array=np.swapaxes(real_grid.array,0,1)[...,::-1]
        return real_grid

    ftGridPair=FTGridPair(realGrid(),reciprocalGrid())
    return ftGridPair

def polar_uniform(realCutOff,N1,N2):
    max_order=int((N2-N2%2)/2)
    orders=np.concatenate((np.arange(max_order+1),-1*np.arange(max_order+N2%2)[:0:-1]))
    radial_steps=np.swapaxes(np.repeat(np.arange(N1)[:,None],N2,axis=1),0,1)
    radial_steps=radial_steps+np.abs(orders[:,None])*np.pi/2*0.95 # q min for 50
    def phi_func():
        return orders*np.pi/max_order
    
    def reciprocalGrid():
        def q_func(*args):
            log.info(args[0].shape)
            return (radial_steps)/realCutOff
            
        reciprocal_grid=GridFactory.construct_grid('uniform_dependent',(phi_func,q_func))
        reciprocal_grid.array=np.swapaxes(reciprocal_grid.array,0,1)[...,::-1]
        return reciprocal_grid

    def realGrid():
        def r_func(*args):
            #return (zeros*realCutOff)/zeros[:,-1,None]
            log.info(args[0].shape)
            return (radial_steps*realCutOff/N1)
        real_grid=GridFactory.construct_grid('uniform_dependent',(phi_func,r_func))
        real_grid.array=np.swapaxes(real_grid.array,0,1)[...,::-1]
        return real_grid

    ftGridPair=FTGridPair(realGrid(),reciprocalGrid())
    return ftGridPair
    
def polarFFTGridPair_Bessel_dependend(realCutOff,N1,N2=0):
    def reciprocalGrid():
        def gridPart(zeroNumber):
   #         maxBesselOrder= 2**(np.floor(np.log(zeroNumber)/np.log(2))+1)
            maxBesselOrder= np.floor(np.sqrt(zeroNumber))**2
  #          log.info('maxBesselOrder={}'.format(maxBesselOrder))
            besselOrders=np.arange(np.floor((maxBesselOrder-1)/2),-np.floor((maxBesselOrder-1)/2)-2,-1)
#            log.info('maxBesselOrder={}'.format(maxBesselOrder))
 #           log.info('besselOrders={}'.format(besselOrders))

            repeatedZeroNumber=np.repeat(zeroNumber,maxBesselOrder)
            zeros=mLib.besselZeros(besselOrders,zeroNumber)
            polarSteps=2*np.pi*besselOrders/maxBesselOrder
#            log.info('polarSteps= \n {}'.format(polarSteps))
            gridPart=np.stack((zeros,polarSteps),axis=1)
            return gridPart,int(maxBesselOrder)
        numbersOfZeros=np.arange(1,N1,1)
        gridPartAndShapes=np.array(list(map(gridPart,numbersOfZeros)))
        gridArray=np.concatenate((gridPartAndShapes[:,0]))
        gridShape=[np.array([N1]),gridPartAndShapes[:,1]]
        reziprocalGrid=Grid(gridArray,gridShape)
        return reziprocalGrid

    def realGrid():
        def gridPart(zeroNumber):
#            maxBesselOrder= 2**(np.floor(np.log(zeroNumber)/np.log(2))+2)
            steps=np.array([2,3,5,7])
            maxOrderKandidates=np.floor(np.log(zeroNumber/steps)/np.log(2))
            log.info('Order Kandidates={}'.format(maxOrderKandidates))
            log.info(' selection = {}'.format(np.log(zeroNumber/steps)/np.log(2)-np.floor(np.log(zeroNumber/steps)/np.log(2))))
            minimalindex=np.abs(np.log(zeroNumber/steps)/np.log(2)-np.floor(np.log(zeroNumber/steps)/np.log(2))).argmin()
            maxBesselOrder=steps[minimalindex]*2**maxOrderKandidates[minimalindex]
#            maxBesselOrder= np.floor(np.sqrt(zeroNumber))**2
  #          log.info('maxBesselOrder={}'.format(maxBesselOrder))
            besselOrders=np.arange(np.floor((maxBesselOrder-1)/2),-np.floor((maxBesselOrder-1)/2)-2,-1)
#            log.info('maxBesselOrder={}'.format(maxBesselOrder))
 #           log.info('besselOrders={}'.format(besselOrders))

            repeatedZeroNumber=np.repeat(zeroNumber,maxBesselOrder)
            zeros=mLib.besselZeros(besselOrders,zeroNumber)/mLib.besselZeros(besselOrders,N1)
            polarSteps=2*np.pi*besselOrders/maxBesselOrder
#            log.info('polarSteps= \n {}'.format(polarSteps))
            gridPart=np.stack((zeros,polarSteps),axis=1)
            return gridPart,int(maxBesselOrder)
        numbersOfZeros=np.arange(1,N1,1)
        gridPartAndShapes=np.array(list(map(gridPart,numbersOfZeros)))
        gridArray=np.concatenate((gridPartAndShapes[:,0]))
        gridShape=[np.array([N1]),gridPartAndShapes[:,1]]
        reziprocalGrid=Grid(gridArray,gridShape)
        return reziprocalGrid
    #factory=gridFactory()
    #reziprocalBesselGrid=factory.constructGrid((radialReciprocal,polarReziprocal),factory.type_dependendUniform)
    fftGridPair=FFTGridPair(realGrid(),reciprocalGrid())
    return fftGridPair


def polarFTGridPair_PNAS(reciprocalCutOff,nRadialSteps,nAngularSteps):
    Nr=nRadialSteps
    Nphi=nAngularSteps
    angular_function=uniformGrid_func([0,2*np.pi],nPoints=Nphi,noEndpoint=True)
    real_radial_function=uniformGrid_func([0,1],nPoints=Nr,noEndpoint=False) 
    real_grid=GridFactory.construct_grid('uniform',[real_radial_function,angular_function])

    reciprocal_radial_function=uniformGrid_func([0,reciprocalCutOff],nPoints=Nr,noEndpoint=False) 
    reciprocal_grid=GridFactory.construct_grid('uniform',[reciprocal_radial_function,angular_function])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair


def radial_grid_func_zernike_old(reciprocal_cut_off,n_radial_points):
    Nr=n_radial_points
    q_max=reciprocal_cut_off
    r_max= Nr/q_max
    r_step=r_max/Nr
    real_radial_function=uniformGrid_func([1/q_max,r_max],nPoints=Nr,noEndpoint=False)
    reciprocal_radial_function=uniformGrid_func([1/r_max,q_max],nPoints=Nr,noEndpoint=False) 
    return {'real':real_radial_function,'reciprocal':reciprocal_radial_function}

    

def n_radial_points_from_oversampling(max_q,reciprocity_coefficient):
    opt = settings.analysis
    max_r=opt.grid.oversampling*opt.particle_radius
    n_radial_points = int(max_r*max_q/reciprocity_coefficient)+1
    return n_radial_points

def radial_grid_func_zernike(reciprocal_cut_off,n_radial_points,reciprocity_coefficient):
    Nr=n_radial_points
    q_max=reciprocal_cut_off
    r_max = polar_spherical_dft_reciprocity_relation_radial_cutoffs(q_max,Nr,reciprocity_coefficient=reciprocity_coefficient)
    r_step=r_max/Nr
    real_radial_function=uniformGrid_func([0,r_max],nPoints=Nr,noEndpoint=False)
    reciprocal_radial_function=uniformGrid_func([0,q_max],nPoints=Nr,noEndpoint=False) 
    return {'real':real_radial_function,'reciprocal':reciprocal_radial_function}
def radial_grid_func_midpoint(reciprocal_cut_off,n_radial_points,reciprocity_coefficient):
    Nr=n_radial_points
    q_max=reciprocal_cut_off
    r_max = polar_spherical_dft_reciprocity_relation_radial_cutoffs(q_max,Nr,reciprocity_coefficient=reciprocity_coefficient)
    r_step=r_max/Nr
    dr = r_max/Nr
    dq = q_max/Nr
    real_radial_function=np.linspace(0+dr/2,r_max-dr/2,num=Nr,endpoint=True)
    reciprocal_radial_function=np.linspace(0+dq/2,q_max-dq/2,num=Nr,endpoint=True) 
    return {'real':real_radial_function,'reciprocal':reciprocal_radial_function}

def radial_grid_gauss(reciprocal_cut_off,n_radial_points,reciprocity_coefficient):
    Nr=n_radial_points
    q_max=reciprocal_cut_off
    r_max = polar_spherical_dft_reciprocity_relation_radial_cutoffs(q_max,Nr,reciprocity_coefficient=reciprocity_coefficient)
    xs,weights = roots_legendre(Nr)
    rs = r_max/2*xs+r_max/2
    qs = q_max/2*xs+q_max/2
    return {'real':rs,'reciprocal':qs}

def radial_grid_func_PNAS(reciprocal_cut_off,n_radial_points):
    Nr=n_radial_points
    q_max=reciprocal_cut_off
    r_max= 1 # should always sattisfy 1=N/(Q*2)
    r_step=r_max/Nr
    real_radial_function=uniformGrid_func([0,r_max],nPoints=Nr,noEndpoint=False)
    reciprocal_radial_function=uniformGrid_func([0,q_max],nPoints=Nr,noEndpoint=False) 
    return {'real':real_radial_function,'reciprocal':reciprocal_radial_function}
    

def polar_ft_grid_pair_zernike(reciprocal_cut_off,nRadialSteps,nAngularSteps,reciprocity_coefficient):
    radial_func=radial_grid_func_zernike(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    real_radial_function=radial_func['real']
    reciprocal_radial_function=radial_func['reciprocal']

    angular_function=uniformGrid_func([0,2*np.pi],nPoints=nAngularSteps,noEndpoint=True)
#    real_grid=GridFactory.construct_grid('uniform',[np.arange(r_step,r_max,r_step),np.arange(0,2*np.pi,a_step)])
    real_grid=GridFactory.construct_grid('uniform',[real_radial_function,angular_function])
    reciprocal_grid=GridFactory.construct_grid('uniform',[reciprocal_radial_function,angular_function])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair

def polar_ft_grid_pair_midpoint(reciprocal_cut_off,nRadialSteps,nAngularSteps,reciprocity_coefficient):
    radial_func=radial_grid_func_midpoint(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    real_radial_function=radial_func['real']
    reciprocal_radial_function=radial_func['reciprocal']

    angular_points=uniformGrid_func([0,2*np.pi],nPoints=nAngularSteps,noEndpoint=True)()
#    real_grid=GridFactory.construct_grid('uniform',[np.arange(r_step,r_max,r_step),np.arange(0,2*np.pi,a_step)])
    real_grid=GridFactory.construct_grid('uniform',[real_radial_function,angular_points])
    reciprocal_grid=GridFactory.construct_grid('uniform',[reciprocal_radial_function,angular_points])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair

def polar_ft_grid_pair_gauss(reciprocal_cut_off,nRadialSteps,nAngularSteps,reciprocity_coefficient):
    radial_grids=radial_grid_gauss(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    rs=radial_grids['real']
    qs=radial_grids['reciprocal']

    angular_points=uniformGrid_func([0,2*np.pi],nPoints=nAngularSteps,noEndpoint=True)()
#    real_grid=GridFactory.construct_grid('uniform',[np.arange(r_step,r_max,r_step),np.arange(0,2*np.pi,a_step)])
    real_grid=GridFactory.construct_grid('uniform',[rs,angular_points])
    reciprocal_grid=GridFactory.construct_grid('uniform',[qs,angular_points])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair

def spherical_ft_grid_pair_zernike_old(reciprocal_cut_off,nRadialSteps,thetas,phis):
    radial_func=radial_grid_func_zernike_old(reciprocal_cut_off,nRadialSteps)
    rs=radial_func['real']()
    qs=radial_func['reciprocal']()
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair


def spherical_ft_grid_pair_zernike(reciprocal_cut_off,nRadialSteps,thetas,phis,reciprocity_coefficient):
    radial_func=radial_grid_func_zernike(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    rs=radial_func['real']()
    qs=radial_func['reciprocal']()
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair

def spherical_ft_grid_pair_midpoint(reciprocal_cut_off,nRadialSteps,thetas,phis,reciprocity_coefficient):
    radial_grid=radial_grid_func_midpoint(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    rs=radial_grid['real']
    qs=radial_grid['reciprocal']
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair

def spherical_ft_grid_pair_gauss(reciprocal_cut_off,nRadialSteps,thetas,phis,reciprocity_coefficient):
    radial_grid = radial_grid_gauss(reciprocal_cut_off,nRadialSteps,reciprocity_coefficient)
    rs=radial_grid['real']
    qs=radial_grid['reciprocal']
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair



def spherical_ft_grid_pair_zernike_dict(reciprocal_cut_off,nRadialSteps,thetas,phis,pi_in_q):
    radial_func=radial_grid_func_zernike(reciprocal_cut_off,nRadialSteps,pi_in_q)
    rs=radial_func['real']()
    qs=radial_func['reciprocal']()
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    grid_pair = DictNamespace(real=real_grid[:],reciprocal=reciprocal_grid[:])
    return grid_pair

def spherical_ft_grid_pair_SinCos(reciprocal_cut_off,nRadialSteps,thetas,phis):
    radial_func=radial_grid_func_PNAS(reciprocal_cut_off,nRadialSteps)
    rs=radial_func['real']()
    qs=radial_func['reciprocal']()
    if isinstance(thetas,(list,tuple,np.ndarray)) and isinstance(phis,(list,tuple,np.ndarray)): 
        real_grid=GridFactory.construct_grid('uniform',[rs,thetas,phis])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs,thetas,phis])
    else:
        real_grid=GridFactory.construct_grid('uniform',[rs])
        reciprocal_grid=GridFactory.construct_grid('uniform',[qs])

    FTGrid_pair=FTGridPair(real_grid,reciprocal_grid)
    return FTGrid_pair


def polarFTGridPair_SinCos_new(realCutOff,n_radial_points,n_angular_points):
    reciprocalCutOff=(n_radial_points-1)/realCutOff
    Nr=n_radial_points
    Nphi=n_angular_points
    def realGrid():
        angularFunction=uniformGrid_func([0,2*np.pi],nPoints=Nphi,noEndpoint=True)
        radialFunction=uniformGrid_func([0,realCutOff],nPoints=Nr,noEndpoint=True) 
        realGrid=GridFactory.construct_grid('uniform',[radialFunction,angularFunction])
        return realGrid
    def reciprocalGrid():
        angularFunction=uniformGrid_func([0,2*np.pi],nPoints=Nphi,noEndpoint=True)
        radialFunction=uniformGrid_func([0,reciprocalCutOff],nPoints=Nr,noEndpoint=True) 
        reciprocalGrid=GridFactory.construct_grid('uniform',[radialFunction,angularFunction])
        return reciprocalGrid

    FTGrid_pair=FTGridPair(realGrid(),reciprocalGrid())
    return FTGrid_pair
    

def extractGridParameters_Donatelli(ftGridPair):
    realGrid=ftGridPair.realGrid
    reciprocalGrid=ftGridPair.reciprocalGrid
    gridShape=realGrid.array.shape
    nRadialPoints=gridShape[0]
    nAngularPoints=gridShape[1]

    stepSizes_real=pyLib.uniformGridGetStepSizes(realGrid.array)
    realCutOff=(nRadialPoints-1)*stepSizes_real[0]
    stepSizes_reciprocal=pyLib.uniformGridGetStepSizes(reciprocalGrid.array)
    reciprocalCutOff=(nRadialPoints-1)*stepSizes_reciprocal[0]

    return {'nRadialPoints':nRadialPoints,'nAngularPoints':nAngularPoints,'realCutOff':realCutOff,'reciprocalCutOff':reciprocalCutOff}


#new GridFactory hands full grid part to function woud need to rewrite angularFunction to that.
def polarFFTGridPair_Chebyshev(realCutOff,numberOfRadialNodes):
    def reziprocalGrid():
        radialDomain=[0,numberOfRadialNodes/realCutOff]
        radialFunction=uniformGrid_func(radialDomain,nPoints=numberOfRadialNodes)
        def angularFunction(*args):
            radialPoint=args[0]
            radialPointNumber=args[1]
            if radialPointNumber >=2:
                numberOfPolarNodes=2**(np.floor(np.log(radialPointNumber)/np.log(2))+2)
            elif radialPointNumber==1:
                numberOfPolarNodes=2**2
            else:
                numberOfPolarNodes=1
            polarDomain=np.array([0,2*np.pi])
            func=uniformGrid_func(polarDomain,nPoints=numberOfPolarNodes,noEndpoint=True)
            gridPart=func()
            return gridPart
        grid=GridFactory.construct_grid('dependent',(radialFunction,angularFunction))
        return grid
    def realGrid():
        radialDomain=[0,numberOfRadialNodes/realCutOff]
        radialFunction=uniformGrid_func(radialDomain,nPoints=numberOfRadialNodes)
        def angularFunction(*args):
            radialPoint=args[0]
            radialPointNumber=args[1]
            if radialPointNumber >=2:
                numberOfPolarNodes=2**(np.floor(np.log(radialPointNumber)/np.log(2))+2)
            elif radialPointNumber==1:
                numberOfPolarNodes=2**2
            else:
                numberOfPolarNodes=1
            polarDomain=np.array([0,2*np.pi])
            func=uniformGrid_func(polarDomain,nPoints=numberOfPolarNodes,noEndpoint=True)
            gridPart=func()
            return gridPart
        grid=GridFactory.construct_grid('dependend',(radialFunction,angularFunction))
        return grid
    grid=reziprocalGrid()
    return grid


def get_grid(grid_opt):
    dim=grid_opt['dimensions']
    _type=grid_opt['type']
    #pi_in_q=grid_opt['pi_in_q']
    reciprocity_coefficient = _get_reciprocity_coefficient(grid_opt)
    Q=grid_opt['max_q']
    n_radial_points = grid_opt['n_radial_points']
    #log.info('dim = {},grid_type = {},pi_in_q = {},max_q = {}, n_radial_points = {}'.format(dim,_type,pi_in_q,Q,n_radial_points))
    if isinstance(n_radial_points,int) and not isinstance(n_radial_points,bool):
        N_r = n_radial_points
    elif isinstance(n_radial_points,str):
        N_r = grid_opt['n_radial_points_from_data']
    else:
        N_r = n_radial_points_from_oversampling(Q,reciprocity_coefficient)
        
    if dim == 2:
        #log.info("grid_opt keys = {}".format(grid_opt.keys()))
        N_phi=len(grid_opt['phis'])
        if (_type == 'Zernike') or (_type == 'trapz'):
            grid = polar_ft_grid_pair_zernike(Q,N_r,N_phi,reciprocity_coefficient)
        elif _type == 'PNAS':
            grid = polarFTGridPair_PNAS(Q,N_r,N_phi)
        elif _type == 'midpoint':
            grid = polar_ft_grid_pair_midpoint(Q,N_r,N_phi,reciprocity_coefficient)
        elif _type == 'gauss':
            grid = polar_ft_grid_pair_gauss(Q,N_r,N_phi,reciprocity_coefficient)
    elif dim == 3:
        thetas=grid_opt['thetas']
        phis=grid_opt['phis']
        #log.info('grid type = {}'.format(_type))
        if (_type == 'Zernike') or (_type == 'trapz'):
            grid = spherical_ft_grid_pair_zernike(Q,N_r,thetas,phis,reciprocity_coefficient)
        elif _type == 'PNAS':
            grid = spherical_ft_grid_pair_SinCos(Q,N_r,thetas,phis)
        elif _type == 'midpoint':
            grid = spherical_ft_grid_pair_midpoint(Q,N_r,thetas,phis,reciprocity_coefficient)
        elif _type == 'gauss':
            grid = spherical_ft_grid_pair_gauss(Q,N_r,thetas,phis,reciprocity_coefficient)

    return grid
