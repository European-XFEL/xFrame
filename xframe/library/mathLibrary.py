import numpy as np
import warnings
from scipy.stats import special_ortho_group
import math
from math import factorial
import logging
import inspect
import scipy.special as specialFunctions
from scipy.special import eval_jacobi
from scipy.special import hyp2f1
from scipy.special import roots_legendre
from scipy.special import comb as binomial
from scipy.linalg import polar as polar_decomp
from scipy.linalg import eigh
import sys
import abc
log=logging.getLogger('root')

from xframe.library.interfaces import GSLDependency
from xframe.library.interfaces import SphericalHarmonicTransformDependency
from xframe.library.interfaces import SoftDependency
from xframe.library.interfaces import DiscreteLegendreTransformDependency
from xframe.library.interfaces import PeakDetectorDependency
from scipy.stats import ttest_1samp
from scipy.optimize import root_scalar as sp_root

module_self = __import__(__name__)

gsl = GSLDependency
shtns = SphericalHarmonicTransformDependency
Soft = SoftDependency
leg_trf = DiscreteLegendreTransformDependency
nj = False
PeakDetector = PeakDetectorDependency


class plane3D:    
    base=np.empty(3)
    x_direction=np.empty(3)
    y_direction=np.empty(3)
    standardForm={'base':np.empty(3), 'x_direction':np.empty(3) , 'y_direction':np.empty(3)}

    def __init__(self,base=np.zeros(3),x_direction=np.array([1,0,0]),y_direction=np.array([0,1,0])):
        self._base=base
        self._x_direction=x_direction
        self._y_direction=y_direction
        self.standardForm='not set'
        self.updateStandardForm()
        self.normal=self._calc_normal()
    def _calc_normal(self):
        #selects normal pointing away from origin
        norm = np.linalg.norm
        n=np.cross(self._x_direction,self._y_direction)
        n_base_prod=n.dot(self.base)
        if not n_base_prod == 0:
            n*=n_base_prod/norm(n_base_prod)
            is_non_zero= np.nonzero(n)
            n[is_non_zero]/=norm(n[is_non_zero])
        return n
    
    def from_array(self,plane:np.ndarray):
        self.__init__(base=plane[0],x_direction=plane[1],y_direction=plane[2])
    
    def setPlain(self,base,x_direction,y_direction):
        self.base=base
        self.x_direction=x_direction
        self.y_direction=y_direction
        self.standardForm=self._calculateStandardForm()

    def updateStandardForm(self):
        normedY_direction=normalizeVector(self._y_direction)
        projectionOfXtoY=self._x_direction.dot(normedY_direction)
        standardX_direction=self._x_direction-projectionOfXtoY*normedY_direction
        standardX_direction=normalizeVector(standardX_direction)
        standardY_direction=normalizeVector(self._y_direction)
        self.standardForm={'base':self.base, 'x_direction':standardX_direction, 'y_direction':standardY_direction}

    @property
    def base(self):
        return self._base
    @base.setter
    def base(self,base:np.ndarray):
        self._base = base
        self.standardForm['base'] = self._base

    @property
    def x_direction(self):
        return self._x_direction
    @x_direction.setter
    def x_direction(self,x_direction:np.ndarray):
        self._x_direction = x_direction
        self.updateStandardForm()

    @property
    def y_direction(self):
        return self._y_direction
    @y_direction.setter
    def y_direction(self,y_direction:np.ndarray):
        self._y_direction = y_direction
        self.updateStandardForm()
    
        
class SampleShapeFunctions:
    coords_cartesian='cartesian'
    coords_polar='polar'
    coords_spherical='spherical'
    @classmethod
    def select_fct_by_coord_sys(cls,coord_sys,function_list):
        if coord_sys=='cartesian':
            #log.info('cartesian')
            function=function_list[0]
        elif coord_sys=='polar':
            #log.info('polar')
            function=function_list[1]
        elif coord_sys=='spherical':
            #log.info('spherical')
            function=function_list[2]
        else:
            log.error('Unknown coodinate System Type = {}.'.format(coordSys))
            raise AssertionError
        return function

    @classmethod
    def get_shape_function(cls,is_in_support,amplitude_function):
        def shape_function(points):
            #log.info('amplitude = {}'.format(amplitude_function(np.arange(100))))
            values=np.zeros(points[:].shape[:-1])
            support_mask=is_in_support(points)
            #log.info('support mask=\n{}'.format(support_mask))
            # log.info(points[support_mask].shape)
            # log.info(values[support_mask].shape)
            values[support_mask]=amplitude_function(points[support_mask])
            return values
        return shape_function

    @classmethod
    def get_disk_function(cls,radius,amplitude_function=lambda points: np.full(points.shape[:-1],1),coordSys='spherical',center=0,norm='standard',random_orientation=False):
        if norm=='standard':
            norm=np.linalg.norm
        elif norm == 'inf':
            def inf(points,axis=0):
                return np.max(np.abs(points),axis=axis)
            norm = inf
        so = special_ortho_group.rvs    
        def isInDisk_cartesian(points):
            if random_orientation:
                points = points[:].dot(so(points[:].shape[-1]))
#            log.info('point {} < radius {} = {}'.format(point,radius,point[0]<radius))
            return norm(points[:]-center,axis=-1)<radius
        def isInDisk_polar(points):
            cart_points=spherical_to_cartesian(points)[:]
            
            if isinstance(center,(np.ndarray,tuple,list)):
                cart_center=spherical_to_cartesian(np.array(center))
                log.info('cart center = {} input = {}'.format(cart_center,center))
            else:
                cart_center = 0
            cart_points-=cart_center
            if random_orientation:
                cart_points = cart_points.dot(so(cart_points.shape[-1]))
    
            #log.info('point {} < radius {} = {}'.format(point,radius,point[0]<radius))
            return norm(cart_points[:],axis=-1)<radius

        isInDisk=cls.select_fct_by_coord_sys(coordSys,[isInDisk_cartesian,isInDisk_polar,isInDisk_polar])
        disk_function=cls.get_shape_function(isInDisk,amplitude_function)
        return disk_function

    @classmethod
    def get_polygon_function(cls,radius,n_corners=3,amplitude_function=lambda points: np.full(points.shape[:-1],1),coordSys='polar',center = (0,0) , random_orientation=False):
        if random_orientation:
            orientation = np.random.rand()*2*np.pi
        else:
            orientation = 0
        corner_vectors=[]
        for n in range(n_corners):
            phi=(2*np.pi/n_corners*n+orientation)%(2*np.pi)
            corner_vector=radius*np.array([np.cos(phi),np.sin(phi)])+spherical_to_cartesian(np.array(center))
            corner_vectors.append(corner_vector)
            
        def isInPolygon_cartesian(points):
            is_in_polygon=True
            n_corners=len(corner_vectors)
            mask=True
            cart_center = spherical_to_cartesian(np.array(center))
            for n in range(n_corners):
                vec1=corner_vectors[n]
                vec2=corner_vectors[(n+1)%n_corners]
                #vec1-=cart_center
                #log.info('center = {}vec 1 = {}'.format(cart_center,vec1))
                #vec2-=cart_center
                #log.info(points.shape)
                #points-=np.array(cart_center)[None,None,:]
                #rotate90_matrix=np.array([[0,1],[-1,0]])
                rotate90_matrix=np.array([[0,1],[-1,0]])
                diff=vec2-vec1
                scal_prod=np.tensordot((points-vec1),rotate90_matrix.dot(diff),axes=([-1],[0]))
                #scal_prod=np.tensordot((points-vec1),diff,axes=([-1],[0]))                
                mask*=np.where(scal_prod>0,False,True)
            return mask
        def isInPolygon_polar(points):
            cart_points = spherical_to_cartesian(points)
            mask=isInPolygon_cartesian(cart_points)
#            log.info('Point {} is in polygon: {}'.format([r*np.cos(phi),r*np.sin(phi)],is_in_polygon))
            return mask
        
        isInPolygon=cls.select_fct_by_coord_sys(coordSys,[isInPolygon_cartesian,isInPolygon_polar,None])
        polygon_function=cls.get_shape_function(isInPolygon,amplitude_function)
        return polygon_function
        
    @classmethod
    def get_tetrahedral_function(cls,radius,amplitude_function=lambda points: np.full(points.shape[:-1],1),coordSys='spherical',center=np.zeros(3), orientation=np.eye(3),random_orientation=False):        
        if random_orientation:
            orientation = special_ortho_group.rvs(3)   
        planes = cls.get_tetrahedron_planes(radius = radius)
        is_in_thetrahedron_bare = cls.get_is_in_shape_from_planes(planes,orientation=orientation)
        def is_in_thetrahedron_cart(points):
            points -= center
            shape = is_in_thetrahedron_bare(points)
            return shape
        
        def is_in_thetrahedron_spherical(points):
            cart_points = spherical_to_cartesian(points)[:]
            cart_center = spherical_to_cartesian(np.asarray(center))
            cart_points -= cart_center 
            shape = is_in_thetrahedron_bare(cart_points)
            #log.info('point {} < radius {} = {}'.format(point,radius,point[0]<radius))
            return shape

        is_in_tetrahedron=cls.select_fct_by_coord_sys(coordSys,[is_in_thetrahedron_cart,None,is_in_thetrahedron_spherical])
        tetrahedral_function=cls.get_shape_function(is_in_tetrahedron,amplitude_function)
        return tetrahedral_function

    @classmethod
    def get_is_in_shape_from_planes(cls,planes,orientation = np.eye(3)):
        SO = special_ortho_group.rvs
        def is_in_shape(cart_points):
            if not isinstance(orientation,np.ndarray):
                rot = SO(3)
            else:
                rot = orientation
            cart_points = cart_points.dot(rot)            
            shape = np.array([True])
            for plane in planes:
                #orth vector from plane to point in units of normal of plane
                d=plane.base.dot(plane.normal)-cart_points[:].dot(plane.normal)
                in_cut = (d>=0)
                shape = (shape & in_cut)
            return shape
        return is_in_shape
    
    @classmethod
    def get_tetrahedron_planes(cls,radius=1):
        z=1/np.sqrt(2)
        R=radius/np.linalg.norm(np.array([0,-1,-z]))
        b1 = R*np.array([0,-1,-z])
        b2 = R*np.array([-1,0,z])
        x1 = np.array([0,2.,0])
        x2 = np.array([2.,0,0])
        y1 = (b1/R-np.array([1.,0,z]))
        y2 = (b1/R-np.array([-1.,0,z]))
        y3 = (b2/R-np.array([0,-1,-z]))
        y4 = (b2/R-np.array([0,1,-z]))
        
        p1=plane3D(base=b1,x_direction = x1,y_direction = y1)
        p2=plane3D(base=b1,x_direction = x1,y_direction = y2)
        p3=plane3D(base=b2,x_direction = x2,y_direction = y3)
        p4=plane3D(base=b2,x_direction = x2,y_direction = y4)
        return [p1,p2,p3,p4]

    @classmethod
    def get_rectangle_function(cls,lengths,center = np.zeros(3),amplitude_function=lambda points: np.full(points.shape[:-1],1),coordSys='spherical'):        
        def is_in_rectangle_cartesian(points):
            n_dims = len(lengths)
            limits = tuple([center[i]-lengths[i]/2,center[i]+lengths[i]/2] for i in range(n_dims))
            mask = True
            for i in range(n_dims):
                p = points[...,i]
                limit = limits[i]
                mask &= ((p>limit[0]) & (p<limit[1]))
            #log.info('rectangle_mask shape = {}'.format(mask.shape))        
            return mask
        def is_in_rectangle_polar(points):
            cart_points = spherical_to_cartesian(points)
            cart_center = spherical_to_cartesian(np.array(center))
            n_dims = len(lengths)
            limits = tuple([cart_center[i]-lengths[i]/2,cart_center[i]+lengths[i]/2] for i in range(n_dims))
            mask = True
            for i in range(n_dims):
                p = cart_points[...,i]
                limit = limits[i]
                mask &= ((p>limit[0]) & (p<limit[1]))
            return mask
        is_in_rectangle=cls.select_fct_by_coord_sys(coordSys,[is_in_rectangle_cartesian,is_in_rectangle_polar,is_in_rectangle_polar])
        rectangle_function=cls.get_shape_function(is_in_rectangle,amplitude_function)
        return rectangle_function
    @classmethod
    def get_anulus_function(cls,inner_radius,outer_radius,center = np.zeros(2),amplitude_function=lambda points: np.full(points.shape[:-1],1),coordSys='spherical',norm = 'standard'):
        if inner_radius == 0:
            anulus_function = cls.get_disk_function(outer_radius,amplitude_function = amplitude_function,coordSys = coordSys, center = center,norm = norm)
        else:
            if norm=='standard':
                norm=np.linalg.norm
            elif norm == 'inf':
                def norm(points,axis=0):
                    return np.max(np.abs(points),axis=axis)
                
            def is_in_anulus_cartesian(points):
                length=norm(points[...,:]-center,axis=-1)
                return (length>inner_radius) & (length<outer_radius)
            def is_in_anulus_polar(points):
                cart_points=spherical_to_cartesian(points)[...,:]
                cart_center=spherical_to_cartesian(np.array(center))
                length = norm(cart_points[:] - cart_center,axis=-1)
                return (length>inner_radius) & (length<outer_radius)
            is_in_anulus=cls.select_fct_by_coord_sys(coordSys,[is_in_anulus_cartesian,is_in_anulus_polar,is_in_anulus_polar])
            anulus_function=cls.get_shape_function(is_in_anulus,amplitude_function)
        return anulus_function

    
def output_lowValuesToZero_decorator(function):
    def lowValuesToZero(array):
        tolerance=1e-15
        isComplex=array.dtype==np.complex
        if isComplex:
            array.real[np.abs(array.real)<tolerance]=0.
            array.imag[np.abs(array.imag)<tolerance]=0.
        else:
            array[np.abs(array)<tolerance]=0.
        return array    
    def newFunction(*args,**kwargs):
        output=function(*args,**kwargs)
        output=lowValuesToZero(output)
        return output

    #applies the parameter signature of function to newFunction
    undecoratedParameters=inspect.signature(function).parameters.values()
    newFunction.__signature__=inspect.signature(newFunction).replace(parameters=undecoratedParameters)
    return newFunction

@output_lowValuesToZero_decorator
def fft_complex_mixedRadix_forward(complex_data_array, step_length=1):    
    transformed_data=np.fft.fft(complex_data_array)
#    log.info('data mathlib forward out =\n{}'.format(newData))
    return transformed_data*step_length

@output_lowValuesToZero_decorator
def fft_complex_mixedRadix_inverse(complex_data_array, step_length=1):
    transformed_data=np.fft.ifft(complex_data_array)
#    log.info('data mathlib inverse out =\n{}'.format(transformedData))
    return transformedData*step_length

def fft_complex_radix2_inverse(complexDataArray, stepLength):
    numberOfSamples=complexDataArray.size/2
    tempData=np.copy(complexDataArray)/stepLength
    gsl.gsl_fft_complex_radix2_inverse(ctypes.c_void_p(tempData.ctypes.data), np.int(1), numberOfSamples)
    tempData=tempData
    return tempData

def fft_real_radix2(realDataArray, stepLength):
    numberOfSamples=realDataArray.size
    tempData=np.copy(realDataArray)
    gsl.gsl_fft_real_radix2_transform(ctypes.c_void_p(tempData.ctypes.data), np.int(1), numberOfSamples)
    tempData=tempData*stepLength
    return tempData


@output_lowValuesToZero_decorator
def fft_real_mixedRadix_forward(realDataArray,stepLength=1):
    newData=np.copy(realDataArray).astype(np.float)
    transformedData=gsl.fft_real_mixedRadix_forward(newData)
    return transformedData*stepLength

@output_lowValuesToZero_decorator
def fft_halfcomplex_mixedRadix_inverse(dataArray,stepLength=1):
    newData=np.copy(dataArray).astype(np.float)
    transformedData=gsl.fft_halfcomplex_mixedRadix_inverse(newData)
    return transformedData*stepLength


def fft_halfcomplex_unpack(halfcomplexPackedArray):
    unpackedArray=gsl.fft_halfcomplex_unpack(halfcomplexPackedArray)
    return unpackedArray


def fft_halfcomplex_pack(complexArray):
    lenData=len(complexArray)   
    packedArray=np.dstack((complexArray.real,complexArray.imag)).flatten()
    lenDataIsEven= lenData%2==0
    if lenDataIsEven:        
        halfcomplexArray=np.concatenate((np.array([packedArray[0]]),packedArray[2:lenData+1]))
#        log.info('halfcomplexArray={}'.format(halfcomplexArray))
    else:
        halfcomplexArray=np.concatenate((packedArray[0],packedArray[2:lenData+1]))
    return halfcomplexArray



def generate_dht_2d(maxR,n_radial_points,harmonic_orders):
    dht,inverse_dht=gsl.generate_dht_2D(n_radial_points,harmonic_orders,maxR)
    return dht,inverse_dht

@output_lowValuesToZero_decorator
def dht_forward(complexDataArray,nu):
    transformedArray=gsl.dht_forward(complexDataArray,nu)
    return transformedArray

@output_lowValuesToZero_decorator
def dht_inverse(complexDataArray,nu):
    transformedArray=gsl.dht_inverse(complexDataArray,nu)
    return transformedArray

def dht_bare(complexDataArray,nu,maxR=1):
    transformedArray=gsl.dht_bare(complexDataArray,nu,maxR=maxR)
    return transformedArray


def besselZeros(orders,zeros):
    ordersArray,zerosArray=np.meshgrid(orders,zeros,indexing='ij')
    ordersArray=np.abs(ordersArray)
    zeros=np.array(list(map(gsl.besselZero,ordersArray.flatten(),zerosArray.flatten()))).reshape(len(orders),len(zeros))
    return zeros
def besselZeros2(order,numberOfZeros):
    zeros=specialFunctions.jn_zeros(order,numberOfZeros)
    return zeros

def bessel_jnu(*args,**kwargs):
    value=specialFunctions.jv(*args,**kwargs)
    return value

def spherical_bessel_jnu(*args,**kwargs):
    value=specialFunctions.spherical_jn(*args,**kwargs)
    return value

def eval_legendre(degrees,arguments):
#    value=gsl.legendrePoly(degree,x)
    return specialFunctions.eval_legendre(degrees,arguments)

def legendrePlFunction(degree):
    polynomial=specialFunctions.legendre(degree)
    return polynomial

@output_lowValuesToZero_decorator
def fft_realToComplex_mixedRadix_forward(realDataArray):
    data=np.copy(realDataArray)
    gsl.fft_real_mixedRadix_forward(data)
    complexData=gsl.fft_halfcomplex_unpack(data)
    return complexData

@output_lowValuesToZero_decorator
def fft_complexToReal_mixedRadix_inverse(complexDataArray):
    packedData=fft_halfcomplex_pack(complexDataArray)
    gsl.fft_halfcomplex_mixedRadix_inverse(packedData)
    realData=packedData
    return realData


def circularHarmonicTransform_real_forward_gsl(dataArray):
    data=np.copy(dataArray)
    harmonicCoefficients=gsl.fft_real_mixedRadix_forward(data)/len(data)    
    return harmonicCoefficients

def circularHarmonicTransform_halfcomplex_inverse_gsl(dataArray,even=True):
    data=np.copy(dataArray)
    realData=gsl.fft_halfcomplex_mixedRadix_inverse(data)/len(data)
    return realData


def circularHarmonicTransform_complex_forward(data_array):
    data=np.copy(data_array)
    #   log.info('circular harmonic forward data shape={}'.format(data.shape))
    harmonic_coefficients=np.fft.fft(data,axis = 1)/data.shape[-1]
    #    harmonicCoefficients=np.fft.fft(data)/len(data)

    return harmonic_coefficients


def circularHarmonicTransform_complex_inverse(data_array):
    data=np.copy(data_array)*data_array.shape[-1]
    real_data=np.fft.ifft(data,axis = 1)
#    realData=np.fft.ifft(data)
    return real_data

def circularHarmonicTransform_real_forward(data_array):
    #log.info("sum input = {}".format(np.sum(data_array)))
    data=np.copy(data_array.real)
    #    log.info('circular harmonic forward data shape={}'.format(data.shape))
    harmonic_coefficients=np.fft.rfft(data)/data.shape[-1]
    #log.info('data shape = {}'.format(data.shape[-1]))
    return harmonic_coefficients

def circularHarmonicTransform_real_inverse(data_array,size):
    data=np.copy(data_array)*size
    #log.info('data shape = {}'.format(size))
    real_data=np.fft.irfft(data,size)
    return real_data

def get_spherical_harmonic_transform_obj(l_max,mode='complex',anti_aliazing_degree=2,n_phi=False,n_theta = False):
    sh=shtns(l_max,mode_flag=mode,anti_aliazing_degree = anti_aliazing_degree,n_phi=n_phi,n_theta=n_theta)
    return sh

def get_soft_obj(l_max):
    global Soft
    soft=Soft(l_max)
    return soft

def complexArraytoRealArray(array):
    realRepresentation=np.stack((array.real,array.imag)).flatten()
    return realRepresentation

def normalizeVector(vector):
    try:
        normedVector=vector*1/np.linalg.norm(vector)
    except ZeroDivisionError as e:
        print('Zero vector can not be normalized')
        print(e)
    return normedVector

def isAlmostZero(number):
    if number < 10**(-15):
        return True
    else:
        return False


def gauss_legendre(n,start=-1,stop=1):
    '''
    returns gauss legendre nodes and weights for integration range from start to stop.
    '''
    xi,w = roots_legendre(n)
    xi = (stop-start)/2*xi+(start + stop)/2
    w = (stop-start)/2*w
    return xi,w
    

def gaussian_fouriertransformed_polar2D_array(pointsArray,sigma):
    valueArray=np.exp(-2*(np.pi**2)*(sigma**2)*(np.square(pointsArray[...,0])) )
    return valueArray
def gaussian_cart(points,sigma):
    '''
    Evaluates a gaussian centered at 0 in cartesian coordinates.
    '''
    if points.ndim ==1:
        points=points[:,None]
    a=1/(2*sigma**2)
    values = np.exp(-np.linalg.norm(points,axis = -1)**2*a)
    return values
def gaussian_spher(points,sigma):
    '''
    Evaluates a gaussian centered at 0 in spherical coordinates.
    '''
    a=1/(2*sigma**2)
    values = np.exp(-np.square(points[...,0])**2*a)
    return values
def gaussian_fourier_transformed_cart(points,sigma):
    '''
    Evaluates the fourier transform of a gaussian centered at 0 in cartesian coordinates.
    '''
    if points.ndim ==1:
        points=points[:,None]
    dim = points.shape[-1]
    pi = np.pi
    a=1/(2*sigma**2)
    prefactor = np.sqrt(pi/a)*1/(sigma*np.sqrt(2*pi))
    values = prefactor**dim * np.exp(-np.pi**2 * np.linalg.norm(points,axis = -1)**2 / a)
    return values
def gaussian_fft_cart(points,sigma):
    if points.ndim ==1:
        points=points[:,None]
    sizes = get_fft_grid_sizes(points)
    values = gaussian_fourier_transformed_cart(points,sigma)
    log.info('points.shape = {}'.format(points.shape))
    phases = 1
    #phases = np.prod((-1)**np.array(np.meshgrid(*[np.concatenate((np.arange(n//2 +1),np.arange(1,n//2+n%2)[::-1])) for n in points.shape[:-1]],indexing = 'ij')[::-1]).T,axis = -1)
    #phases = 1#np.prod((-1)**np.array(np.meshgrid(*[np.arange(n) for n in points.shape[:-1]],indexing = 'ij')[::-1]).T,axis = -1)
    phases = np.squeeze(phases)
    log.info('phases.shape = {}'.format(phases.shape))
    log.info('sizes = {}'.format(sizes))
    values*= phases * np.prod(sizes)
    return values
def get_fft_grid_steps(points):
    if points.ndim ==1:
        points=points[:,None]
    dim = points.shape[-1] 
    steps = np.abs(np.diag(points[tuple(np.eye(dim,dtype = int))]))
    return steps
def get_fft_grid_sizes(points):
    if points.ndim ==1:
        points=points[:,None]
    steps = get_fft_grid_steps(points)
    sizes = steps * points.shape[:-1]
    return sizes
def get_fft_reciprocal_grid(grid):
    if grid.ndim ==1:
        grid=grid[:,None]
    sizes = get_fft_grid_sizes(grid)
    length = grid.shape[:-1]
    new_sizes = 2/(2*sizes)
    
    reciprocal_grid = np.array(
        np.meshgrid(*[np.concatenate((np.arange(n//2 +1),-np.arange(1,n//2+n%2)[::-1])) for n in length],indexing = 'ij')[::-1]
        ,dtype = float).T
    reciprocal_grid *= new_sizes
    return np.squeeze(reciprocal_grid)

def convolve_with_gaussian_cart(data,grid,sigma=1):
    r_grid = get_fft_reciprocal_grid(grid)
    r_data = np.fft.fftn(data)
    r_gaussian = gaussian_fft_cart(r_grid,sigma)
    convolution = np.fft.ifftn(r_data*r_gaussian).real
    data_sum = np.sum(data)
    convolution_sum = np.sum(convolution)
    convolution *= data_sum/convolution_sum
    return convolution

def gaussian_fourier_transformed_spherical(points,sigma):
    '''
    Evaluates the fourier transform of a gaussian centered at 0 in spherical coordinates.
    '''
    pi = np.pi
    a=1/(2*sigma**2)
    prefactor = np.sqrt(pi/a) 
    values = prefactor * np.exp(-np.pi**2 * np.square(points[...,0])**2 / a)
    return values

def id(_in):
    return _in

def cartesian_to_spherical(grid):
    dimension=grid[:].shape[-1]
    n_grid=grid.copy()
    try:
        assert dimension in (2,3) 
    except AssertionError as e:
        log.error(e)
        raise e

    if dimension==2:
        x=grid[...,0].copy()
        y=grid[...,1].copy()

        n_grid[...,0]=np.sqrt(np.square(x)+np.square(y))
        phi=np.arctan2(y,x)
        n_grid[...,1]=np.where(phi<0,phi+2*np.pi,phi)        
    else:
        x=grid[...,0].copy()
        y=grid[...,1].copy()
        z=grid[...,2].copy()

        r=np.sqrt(np.square(x)+np.square(y)+np.square(z))
        n_grid[...,0]=r
        theta=np.zeros(grid[...,0].shape)
        r_nonzero=(r!=0)
        if r_nonzero.any():
            #log.info('r_nonzero={}'.format(r_nonzero))
            tmp=z[r_nonzero]/r[r_nonzero]
            #log.info('shape tmp = {}'.format(tmp.shape))
            #log.info('phi non_zero shape = {}'.format(phi[r_nonzero].shape))
            theta[r_nonzero]=np.arccos(tmp)
        #log.info('phi={}'.format(theta))
        #log.info('n_grid={}'.format(n_grid))
        n_grid[...,1]=theta
        phi=np.arctan2(y,x)
        n_grid[...,2]=np.where(phi<0,phi+2*np.pi,phi)
    return n_grid

def get_gaussian_weights_1d(data,grid_points):
    sum_data=np.sum(data)
    pos=np.sum(data*grid_points)/sum_data
    sigma=np.sqrt(np.abs(np.sum(np.square(grid_points-pos)*data)/sum_data))
    return {'pos':pos,'sigma':sigma}

def spherical_to_cartesian(grid):
    dimension=grid[:].shape[-1]
    n_grid=grid.copy()
    try:
        assert dimension in (2,3) 
    except AssertionError as e:
        log.error(e)
        raise e

    if dimension==2:
        r=grid[...,0].copy()
        phi=grid[...,1].copy()

        n_grid[...,0]=r*np.cos(phi)
        n_grid[...,1]=r*np.sin(phi)
    else:
        r=grid[...,0].copy()
        theta=grid[...,1].copy()
        phi=grid[...,2].copy()

        xy_projection=r*np.sin(theta)

        n_grid[...,0]=np.cos(phi)*xy_projection
        n_grid[...,1]=np.sin(phi)*xy_projection
        n_grid[...,2]=r*np.cos(theta)
    return n_grid
        
def cylindricalToCartesian(Data):
    x=np.cos(Data[:,1])*Data[:,0]
    y=np.sin(Data[:,1])*Data[:,0]
    z=Data[:,2]
    cartesianData=np.stack((x,y,z),axis=1)
    return cartesianData
 
def cartesianToSpherical(Data):
     r=np.linalg.norm(Data,axis=1)
     
     nonZeroRadi_mask= r>0
     theta=np.zeros(r.shape)
     theta[nonZeroRadi_mask]=np.arccos(Data[:,2][nonZeroRadi_mask]/r[nonZeroRadi_mask])

     xPositions=Data[:,0]
     yPositions=Data[:,1]
     positiveY_mask= yPositions>=0
     negativeY_mask= np.logical_not(positiveY_mask)
     thetaNot0orPi_mask=np.logical_and(theta!=0,theta!=np.pi)

     positiveY_NonZeroDiv_mask=np.logical_and(np.logical_and(positiveY_mask,nonZeroRadi_mask),thetaNot0orPi_mask)

     negativeY_NonZeroDiv_mask=np.logical_and(np.logical_and(negativeY_mask,nonZeroRadi_mask),thetaNot0orPi_mask)
     phi=np.zeros(r.shape)
     phi[positiveY_NonZeroDiv_mask]=np.arccos(xPositions[positiveY_NonZeroDiv_mask]/r[positiveY_NonZeroDiv_mask])
     phi[negativeY_NonZeroDiv_mask]=2*np.pi-np.arccos(xPositions[negativeY_NonZeroDiv_mask]/r[negativeY_NonZeroDiv_mask])

     sphericalData=np.stack((r,theta,phi),axis=1)
     return sphericalData

def phiFrom_R_Theta_X_Y(r,theta,x,y):
     nonZeroRadi_mask= r>0
     positiveY_mask= y>=0
     negativeY_mask= np.logical_not(positiveY_mask)
     thetaNot0orPi_mask=np.logical_and(theta!=0,theta!=np.pi)
     positiveY_NonZeroDiv_mask=np.logical_and(np.logical_and(positiveY_mask,nonZeroRadi_mask),thetaNot0orPi_mask)
     negativeY_NonZeroDiv_mask=np.logical_and(np.logical_and(negativeY_mask,nonZeroRadi_mask),thetaNot0orPi_mask)
     
     phi=np.zeros(r.shape)
     phi[positiveY_NonZeroDiv_mask]=np.arccos(x[positiveY_NonZeroDiv_mask]/r[positiveY_NonZeroDiv_mask])
     phi[negativeY_NonZeroDiv_mask]=2*np.pi-np.arccos(x[negativeY_NonZeroDiv_mask]/r[negativeY_NonZeroDiv_mask])
     return phi

def testAdd(a,b):
    return a+b


def second_derivative(data,step_size,accuracy_order):
    '''
    Approximates the second derivative for a dataset sampled on a uniform grid for accuracy_orders in (2,4,6,8):
    See https://en.wikipedia.org/wiki/Finite_difference_coefficient
    '''
    coefficients_by_order = {
        '2':np.array([1,-2,1]),
        '4':np.array([-1/12,4/3,-5/2,4/3,-1/12]),
        '6':np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90]),
        '8':np.array([-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560])
    }
    
    assert str(accuracy_order) in coefficients_by_order, 'Invalid accuracy_order {} valid inputs are {}'.format(accuracy_order,coefficients_by_order.keys())
    return np.convolve(data,coefficients_by_order[str(accuracy_order)],mode='valid')/step_size**2

def first_derivative(data,step_size,accuracy_order):
    '''
    Approximates the second derivative for a dataset sampled on a uniform grid for accuracy_orders in (2,4,6,8):
    See https://en.wikipedia.org/wiki/Finite_difference_coefficient
    '''
    coefficients_by_order = {
        '2':np.array([-1/2,0,1/2]),
        '4':np.array([1/12,-2/3,0,2/3,-1/12]),
        '6':np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60]),
        '8':np.array([1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280])
    }
    
    assert str(accuracy_order) in coefficients_by_order, 'Invalid accuracy_order {} valid inputs are {}'.format(accuracy_order,coefficients_by_order.keys())
    return -np.convolve(data,coefficients_by_order[str(accuracy_order)],mode='valid')/step_size
    
    

def eval_zernike_polynomials(m_array,n_max,points):
    def evaluate(m,n):
        return ((-1)**((n-m)/2))*(points**m)*eval_jacobi((n-m)/2,m,0,1-2*(points**2))
#        return (points**m)#*eval_jacobi((n-m)/2,m+1,m+1,points**2)/eval_jacobi((n-m)/2,m+1,m+1,1)
    values={}
    for m in m_array:
        print('generating zernike polynomials for m={}'.format(m))
        ns=np.arange(m,n_max+1,2)
        len_ns=len(ns)
        values[m]=np.array(tuple(map(evaluate,(m,)*len_ns,ns)))
    return values


def eval_ND_zernike_polynomials_old(m_array,n_max,points,dimension):
    D=dimension
    def evaluate(m,n):
        return ((-1)**((n-m)/2))*(points**m)*eval_jacobi((n-m)/2,m+D/2-1,0,1-2*(points**2))
#        return (points**m)#*eval_jacobi((n-m)/2,m+1,m+1,points**2)/eval_jacobi((n-m)/2,m+1,m+1,1)
    values={}
    for m in m_array:
        #print('generating zernike polynomials for m={}'.format(m))
        ns=np.arange(m,n_max+1,2)
        len_ns=len(ns)
        values[m]=np.array(tuple(map(evaluate,(m,)*len_ns,ns)))
    return values

def eval_ND_zernike_polynomials(l_array,s_max,points,dimension):
    '''
    Computes the radial part $R^l_s(p)$ of the D-dimensional Zernike polynomials. For each $l$ in l\_ array all coefficients $R^l_s(p)$ with $s_{\max} \geq s\geq l$ and even $s-l$ are computed.    
    '''
    D=dimension
    def evaluate(l,s):
        return ((-1)**((s-l)/2))*(points**l)*eval_jacobi((s-l)/2,l+D/2-1,0,1-2*(points**2))
#        return (points**m)#*eval_jacobi((n-m)/2,m+1,m+1,points**2)/eval_jacobi((n-m)/2,m+1,m+1,1)
    values={}
    for l in l_array:
        #print('generating zernike polynomials for m={}'.format(m))
        s=np.arange(l,s_max+1,2)
        len_s=len(s)
        values[l]=np.array(tuple(map(evaluate,(l,)*len_s,s)))
    return values

def eval_ND_zernike_polynomials_new(m_array,n_max,points,dimension):
    D=dimension
    def evaluate(m,n):
        nmD2=(n+m+D)/2
        nm2 = (n-m)/2
        binom_args=[nmD2-1,nm2]
        f1_args = [-nm2,nmD2,m+D/2,points**2]
        return (-1)**nm2*(points**m)*hyp2f1(*f1_args)*binomial(*binom_args)
        #return ((-1)**((n-m)/2))*(points**m)*eval_jacobi((n-m)/2,m-1+D/2,0,1-2*(points**2))
#        return (points**m)#*eval_jacobi((n-m)/2,m+1,m+1,points**2)/eval_jacobi((n-m)/2,m+1,m+1,1)
    values={}
    for m in m_array:
        print('generating zernike polynomials for m={}'.format(m))
        ns=np.arange(m,n_max+1,2)
        len_ns=len(ns)
        values[m]=np.array(tuple(map(evaluate,(m,)*len_ns,ns)))
    return values

def eval_ND_zernike_polynomials_gsl(m_array,n_max,points,dimension):
    D=dimension
    def evaluate(m,n):
        nmD2=(n+m+D)/2
        nm2 = (n-m)/2
        binom_args=[nmD2-1,nm2]
        f1_args = [-nm2,nmD2,m+D/2,points**2]
        h2f1 = gsl.hyperg_2F1(*f1_args)
        return (-1)**nm2*(points**m)*h2f1*binomial(*binom_args)
        #return ((-1)**((n-m)/2))*(points**m)*eval_jacobi((n-m)/2,m-1+D/2,0,1-2*(points**2))
#        return (points**m)#*eval_jacobi((n-m)/2,m+1,m+1,points**2)/eval_jacobi((n-m)/2,m+1,m+1,1)
    values={}
    for m in m_array:
        print('generating zernike polynomials for m={}'.format(m))
        ns=np.arange(m,n_max+1,2)
        len_ns=len(ns)
        values[m]=np.array(tuple(map(evaluate,(m,)*len_ns,ns)))
    return values



def generate_diagonal_circular_ht(n_angular_points,n_orders=False):
    if isinstance(n_orders,bool):
        n_orders=n_angular_points
    pt=np.arange(n_angular_points)
    o=np.concatenate((np.arange(n_orders//2+1),-1*np.arange(n_orders//2+n_orders%2)[:0:-1]))
    exp=1/n_angular_points*np.exp(-1.j*2*np.pi*o[:,None]*pt[None,:]/n_angular_points)

    def cht(f):
        f=f.reshape(n_orders,-1,n_angular_points)
        return np.sum(f*exp[:,None,:],axis=2)
    return cht

def nearest_positive_semidefinite_matrix(A,low_positive_eigenvalues_to_zero = False):
    '''
    Generates the closest matrix to A in the Frobenius norm that is positive semidefinite.
    See: N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6 
    '''
    B = (A + np.swapaxes(A,-1,-2).conj()) / 2
    l,v = np.linalg.eigh(B)#,driver = 'ev')
    #l,v = np.linalg.eigh(B)
    limit = 0
    if low_positive_eigenvalues_to_zero:
        # Estimates eigenvalue noise floor by considering the absolute value of the lowest eigenvalue of A
        # project all eigenvalues with lower values to zero.
        eigvals,_ = np.linalg.eig(A)
        limit = np.abs(np.min(eigvals))
        #limit = 0
    #log.info(f'eig limit = {limit}')
    l[l<limit]=0
    A2 =v*l[...,None,:] @ np.swapaxes(v,-1,-2).conj()
    #A2 = v @ np.diag(l) @ v.T
    return A2

def approximate_tikhonov_parameters(A,b,std_noise=False):
    '''
    Chooses optimal tikhonov parameters as described in
    [1] O'leary, Dianne. (2001). Near-Optimal Parameters for Tikhonov and Other Regularization Methods. SIAM Journal on Scientific Computing. 23. 10.1137/S1064827599354147. 
    '''
    if A.ndim < 3:
        A = A[None,...]
        b = b[None,...]
    m,n = A[0].shape
    if isinstance(std_noise,bool):
        size_for_std = max(m-n,10)
        std_estimate = np.std(b[:,-size_for_std:],axis = -1)
    else:
        std_estimate = np.atleast_1d(std_noise)
    u,s,vh = np.linalg.svd(A,full_matrices=False)
    betas = np.sum(u*b[:,:,None],axis = -2)
    ks = []
    for i,beta in enumerate(betas):
        if abs(b[i,n-1]) > 3.5*std_estimate[i]:
            ks.append(n)
        else:
            k=n
            for j in np.arange(n-1)[::-1]:
                p_value = ttest_1samp(beta[j:n].real,0)[1]
                if p_value>0.05:
                    k=j
            ks.append(k)
    ks = np.asarray(ks)
    def parameter_func(lambd,i):
        D = (s[i]**2+lambd)
        #log.info('A shape = {} betas shape = {} std_estimate_shape ={} ks shape = {}'.format(s.shape,betas.shape,std_estimate.shape,ks.shape))
        val = np.sum(np.abs(betas[i])**2*lambd/(D**3)) - np.sum((np.abs(betas[i])**2/(D**2))[ks[i]:]) - np.sum((std_estimate[i]**2/(D**2))[:ks[i]])
        #print(val)
        return val
    
    def find_root_bracket(i):
        std = std_estimate[i]
        lower_bound = 0
        upper_bound = False       
        for h in np.arange(30)[::-1]:
            lower_lambd = std/(10**h)
            upper_lambd = std*(10**h)
            x = parameter_func(lower_lambd,i)
            y = parameter_func(upper_lambd,i)
            if x<0:
                lower_bound =lower_lambd
            if y>=0:
                upper_bound = upper_lambd
            #print("b = {} x = {},b = {} y = {}".format(lower_lambd,x,upper_lambd,y))
        #log.info('lower {}, upper {}, last x {},last y {}'.format(lower_bound,upper_bound,x,y))
        try:
            assert isinstance(upper_bound,float),'could not find upper bound on the tikhonov parameter for the {}th linear system,'.format(i)
        except Exception as e:
            log.warning('Could not find optimal tikhonov parameter for the {}th linear system. Default to lambda = 0.0.last tested value = {}'.format(i,y))
            return False
        return (lower_bound,upper_bound)
    def find_root(system_id):
        bracket = find_root_bracket(system_id)
        if isinstance(bracket,tuple):
            lambd = sp_root(parameter_func,args = (system_id,),bracket = bracket).root            
        else:
            lambd = 0.0
        return lambd
    lambd = tuple(find_root(i) for i in range(len(A)))
    #log.info(lambd)
    return np.asarray(lambd)
    
def tikhonov_solver_svd(A,b,lambd):
    '''
    Finds x that minimizes $|| A x -B ||^2 - ||\lambda Id x ||^2 $
    '''
    n_orders = A.shape[-1]
    lambd = np.atleast_1d(lambd)
    if A.ndim < 3:
        A = A[None,...]
        b = b[None,...]

    u, s, vh = np.linalg.svd(A, full_matrices=False)
    small_singular_values = s <= 1e-15  # same default value as scipy.linalg.pinv
    uTb = np.sum(u*b[:,:,None],axis = -2)
    d = s / (s**2 + lambd[:,None])
    d[small_singular_values] = 0
    d_uTb = d * uTb
    x = np.sum(vh*d_uTb[:,:,None],axis = -2)        
    return x

def tikhonov_prepare_data(A,b,weights = None):
    A = np.copy(A)
    b = np.copy(b)

    A_offsets = np.average(A, axis=1,weights=weights)
    A -= A_offsets[:,None,:]    
    b_offsets = np.average(b, axis=1,weights=weights)
    b -= b_offsets[:,None]
    return A,b,A_offsets,b_offsets

def tikhonov_calc_offset(xs,A_offsets,b_offsets):
    return b_offsets - np.sum(A_offsets*xs,axis = -1)    

def tikhonov_regularization(As,bs,lambd,allow_offset = False):
    '''  
    For each pair of elements (A,b) in As,bs this method finds x that minimizes: 
    $||Ax-b||+||\lambda*x||$    
    if allow_offset is true it minimizes:
    $||A'x-b'||+||\lambda*x||$ 
    where A'=A-np.mean(A,axes = 0) and b' = b-np.mean(b).
    For that x one has $Ax+offset \approx b$, where the offset is fixed by x,A-A'and b-b'.    
    '''
    if allow_offset:
        As,bs,A_offsets,b_offsets = tikhonov_prepare_data(As,bs)
        xs = tikhonov_solver_svd(As,bs,lambd)        
        offsets = tikhonov_calc_offset(xs,A_offsets,b_offsets)
    else:
        xs = tikhonov_solver_svd(As,bs,lambd)
        offsets = np.zeros(len(As),dtype = As.dtype)
    return xs,offsets    

def optimal_tikhonov_regularization(As,bs,allow_offset = False,std_noise = False):    
    lambdas = approximate_tikhonov_parameters(As,bs,std_noise=std_noise)
    xs,offsets = tikhonov_regularization(As,bs,lambdas,allow_offset=allow_offset)
    return xs,offsets
def optimal_tikhonov_regularization_worker(q1s,q2s,As,bs,allow_offset = False,std_noise = False,**kwargs):
    #log.info(q1s)
    #log.info(q2s)
    As = As[q1s,q2s]
    bs = bs[q1s,q2s]
    #log.info('As shape = {} bs shape = {}'.format(As.shape,bs.shape))
    xs,offsets = optimal_tikhonov_regularization(As,bs,allow_offset = allow_offset,std_noise = std_noise)
    return xs

####################
##   exp ramp     ##
class Ramp(abc.ABC):
    @abc.abstractmethod
    def eval(self,x):
        pass
    def __call__(self,x):
        return self.eval(x)
    
class ExponentialRamp(Ramp):
    def __init__(self,start,stop,exponent,stop_argument=1):
        self.start = start
        self.stop = stop
        self.stop_argument = stop_argument
        if (stop < start):
            exponent *= exponent/abs(exponent)*-1
        else:
            exponent *= exponent/abs(exponent)
        self.exponent = exponent
        self.A=0
        self.B=0
        self.set_model_parameters()
    def set_model_parameters(self):
        self.A = (self.start-self.stop)/(1-np.exp(self.exponent*self.stop_argument))
        self.B = self.start-self.A
    def eval(self,x):
        if self.start>self.stop:
            val = np.maximum(self.A*np.exp(x*self.exponent)+self.B,self.stop)
        else:
            val = np.minimum(self.A*np.exp(x*self.exponent)+self.B,self.stop)
        return val

class LinearRamp(Ramp):
    def __init__(self,start,stop=False,slope=False,default_start=False,default_stop =False):
        self.default_stop = default_stop
        self.default_start = default_start
        #log.info(f'input = {[start,stop,slope]}')
        if not isinstance(start,(list,tuple)):
            self.start = (start,0)
        else:
            self.start = start
            
        self.undefined = False
        if not np.issubdtype(np.array(self.start[0]).dtype,np.number):
            if default_start == False:
                self.undefined = True
            else:
                self.start= (default_start,0)
            
        self.stop,self.stop_is_defined = self.parse_stop(stop)
        self.slope_is_defined = not isinstance(slope,bool)
        self.slope = slope
        #log.info(f'lin ramp options = {[self.start,self.stop,self.slope]}')
        #assert self.stop_is_defined or self.slope_is_defined,'Linear ramp requires either a stop value or a slope value given on instanciation.'
        if not self.undefined:
            self.set_model_parameters()

    def parse_stop(self,stop):
        stop_is_valid = False
        if isinstance(stop,(list,tuple)):
            stop_value_is_number = np.issubdtype(np.array(stop[0]).dtype,np.number)
            if not stop_value_is_number:
                default_is_number = np.issubdtype(np.array(self.default_stop).dtype,np.number)
                if default_is_number:
                    stop[0] = self.default_stop
                    stop_value_is_number = True
            stop_argument_is_number =  np.issubdtype(np.array(stop[1]).dtype,np.number)
            if stop_argument_is_number and stop_value_is_number:
                stop_after_start = stop[1]>=self.start[1]
                if stop_after_start:
                    stop_is_valid = True
        if not stop_is_valid:
            stop = False
        return stop,stop_is_valid
    def set_model_parameters(self):
        start,stop,slope = self.start,self.stop,self.slope
        if (not self.stop_is_defined) and (not self.slope_is_defined):
            self.A=0
            self.B=start[0]
        elif self.stop_is_defined:
            self.C=stop[0]
            if (stop[1]-start[1])==0:
                self.A=0
            else:
                self.A=(stop[0]-start[0])/(stop[1]-start[1])
            if self.slope_is_defined:
                self.A = slope
        elif slope == 0:
            self.C = np.nan
            self.A = slope
        else:
            self.C=np.sign(slope)*np.inf
            self.A = slope
        self.B = start[0]-self.A*start[1]
                
    def eval(self,x):
        if not self.undefined:
            val = self.A*x+self.B
            if self.A<0:
                val = max(val,self.C)
            elif self.A>0:
                val = min(val,self.C)
        else:
            val = np.nan
        return val
#############
## linalg  ##
def distance_from_line_2d(line_points,grid):
    p1,p2 = line_points
    dist = p2-p1
    dist_rot = np.array([[0,1],[-1,0]])@dist
    grid = grid-p1
    dist=np.sum(grid*dist_rot[None,None,:],axis=-1)
    return dist
    

############
###statistics###
def relMeanSquareError(expected,approximation):
    expected=np.asarray(expected)
    approximation=np.asarray(approximation)

    absExpected=np.abs(expected)
    divisors=np.where(absExpected<1e-15,1,np.square(np.abs(expected)))
    differences=np.where(absExpected<1e-15,0,np.square(np.abs(expected-approximation)))

    errors=(differences/divisors).flatten()
    meanError=np.sqrt(1/len(differences)*np.sum(differences/divisors))
    return meanError
    

###############################################
## Polar/Spherical FFT reciprocity relations ##
def polar_spherical_dft_reciprocity_relation_radial_cutoffs_old(cutoff:float,n_points:int,pi_in_q = True):
    '''
    Reciprocity relation between the real and reciprocal cutoff.
    If pi_in_q is true it is assumed that the reciprocal unit is 2Pi/length otherwise it is just 1/length.    
    $2*R*2*Q = 2*N$
    '''
    if pi_in_q:
        other_cutoff = np.pi*n_points/cutoff
    else:
        other_cutoff = n_points/(2*cutoff)
    return other_cutoff

def polar_spherical_dft_reciprocity_relation_radial_cutoffs(cutoff:float,n_points:int,reciprocity_coefficient=np.pi):
    '''
    Reciprocity relation between the real and reciprocal cutoff.
    If pi_in_q is true it is assumed that the reciprocal unit is 2Pi/length otherwise it is just 1/length.    
    $2*R*2*Q = 2*N$
    '''
    other_cutoff = reciprocity_coefficient*n_points/cutoff    
    return other_cutoff

def _pi_in_q__to__reciprocity_coefficient(pi_in_q):
    if pi_in_q:
        return np.pi
    else:
        return 1/2

    
def polar_spherical_dft_reciprocity_relation_radial_steps(step:float,n_points:int,pi_in_q = True):
    '''
    Reciprocity relation between the real and reciprocal step sizes.
    If pi_in_q is true it is assumed that the reciprocal unit is 2Pi/length otherwise it is just 1/length.
    $\delta R*\delta Q = 1/(2*N)$    
    '''
    if pi_in_q:
        other_step = np.pi*1/(n_points*step)
    else:
        other_step = 1/(2*n_points*step)
    return other_step
    
def polar_spherical_dft_reciprocity_relation_radial_step_cutoff(step:float,n_points:int,pi_in_q = True):
    '''
    Reciprocity relation between the real and reciprocal step sizes.
    If pi_in_q is true it is assumed that the reciprocal unit is 2Pi/length otherwise it is just 1/length.
    $\delta R*\delta Q = 1/(2*N)$    
    '''
    if pi_in_q:
        other_cutoff = np.pi/step
    else:
        other_cutoff = 1/(2*step)
    return other_cutoff
    

#assumes sperical coordinate grid where r,phi are uniformely sampled and cos(theta) are gauss legendre nodes.
# grid coords are (r,theta,phi)
class SphericalIntegrator():
    def __init__(self,grid):
        self.n_r,self.n_theta,self.n_phi = grid.shape[:-1]
        self.grid = grid
        self.max_r = np.max(grid[:,0,0,0])
        self.norm = 4/3*np.pi*self.max_r**3
        self.gauss_weights = roots_legendre(self.n_theta)[1]
        self.integrate,self.integrate_normed = self.generate_integration_routines()
        
        

    def generate_integration_routines(self):
        pi = np.pi
        w = self.gauss_weights
        rs = self.grid[:,0,0,0]
        n = self.n_theta
        norm = self.norm
        def integrate(values):
            w_shape = (1,) + w.shape + (1,)*(values.ndim - 3)
            rs_shape = rs.shape + (1,)*(values.ndim - 3)
            s2_int = pi/n*np.sum(w.reshape(w_shape)*np.sum(values,axis=2),axis = 1)
            r_int = np.trapz(s2_int * (rs**2).reshape(rs_shape) , x = rs,axis = 0)
            return r_int
        def integrate_normed(values):
            return integrate(values)/norm
        return integrate,integrate_normed
    def L2_norm(self,values):
        return self.integrate(values*values.conj())
                

class PolarIntegrator():
    '''
    Assumes polar coordinate grid where r,phi are uniformely sampled.
    Grid coords are (r,phi)
    '''
    def __init__(self,grid):
        self.n_r,self.n_phi = grid.shape[:-1]
        self.grid = grid
        self.max_r = np.max(grid[:,0,0])
        self.norm = np.pi*self.max_r**2
        self.integrate,self.integrate_normed = self.generate_integration_routines()

    def generate_integration_routines(self):
        rs = self.grid[:,0,0]
        norm = self.norm
        phis = self.grid[0,:,1]
        def integrate(values):
            rs_shape = rs.shape + (1,)*(values.ndim - 2)
            s_int = np.trapz(values , x = phis,axis = 1)
            r_int = np.trapz(s_int * rs.reshape(rs_shape) , x = rs,axis = 0)
            return r_int
        def integrate_normed(values):
            return integrate(values)/norm
        return integrate,integrate_normed
    def L2_norm(self,values):
        return self.integrate(values*values.conj())


class RadialIntegrator():
    def __init__(self,radial_points,dimension):        
        self.n_r = len(radial_points)
        self.radial_points = radial_points
        self.dimension = dimension
        self.max_r,self.min_r = radial_points.max(),radial_points.min()
        self.norm = np.pi*self.max_r**dimension-np.pi*self.min_r**dimension
        self.rweights = self.radial_points**(self.dimension-1)
        self.integrate,self.integrate_normed = self.generate_integration_routines()
    def generate_integration_routines(self):
        rs = self.radial_points
        w = self.rweights
        norm = self.norm
        def integrate(values,axis = -1):
            #rs_shape = rs.shape + (1,)*(values.ndim - 2)
            len_shape = len(values.shape)
            weight_slice=(None,)*(axis%len_shape) + (slice(None),) + (None,)*(len_shape-axis%len_shape-1)
            r_int = np.trapz(values * w[weight_slice] , x = rs,axis = -1)
            return r_int
        def integrate_normed(values):
            return integrate(values)/norm
        return integrate,integrate_normed
    
    def L2_norm(self,values,axis=-1):
        return self.integrate(values*values.conj(),axis = axis)        
    
class Cacheaware_numpy:
    @staticmethod
    def conjugate(x,*args,block_size=None,out=None,**kwargs):
        if out == None:
            out = np.empty(x)
            

################
###statistics###

def relMeanSquareError(expected,approximation):
    expected=np.asarray(expected)
    approximation=np.asarray(approximation)

    absExpected=np.abs(expected)
    divisors=np.where(absExpected<1e-15,1,np.square(np.abs(expected)))
    differences=np.where(absExpected<1e-15,0,np.square(np.abs(expected-approximation)))

    errors=(differences/divisors).flatten()
    meanError=np.sqrt(1/len(differences)*np.sum(differences/divisors))
    return meanError


def nan_mask_decorator(func):
    def masked_func(*args,fill_value = 0.0, return_mask = False, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = np.atleast_1d(func(*args,**kwargs))
        nan_mask = np.isnan(result)
        result[nan_mask] = fill_value
        if return_mask:
            return result, ~nan_mask
        else:
            return result
    return masked_func

#wrapper for numpy mean and std
@nan_mask_decorator
def masked_mean_old(*args,**kwargs):
    return np.mean(*args,**kwargs)

@nan_mask_decorator
def masked_std(*args,**kwargs):
    return np.std(*args,**kwargs)
    
def combine_means(means,n_data_points):
    n_total_data_points = np.sum(n_data_points)    
    combined_mean = np.sum(tuple(mean*n_data_points for mean,n_data_points in zip(means,n_data_points)))/n_total_data_points
    return combined_mean,n_total_data_points

def masked_mean(data,mask,axis = 0):
    counts = np.sum(mask,axis = axis)
    counts_non_zero = (counts!=0)
    data_sum = np.sum(data,where=mask,axis = axis)
    data_sum[counts_non_zero]=data_sum[counts_non_zero]/counts[counts_non_zero]
    return data_sum,counts
def combine_means_2D(means,counts,axis = 0):
    means = np.asarray(means)
    counts = np.asarray(counts)
    #log.info('means shape in combine ={}'.format(means.shape))
    total_counts = np.sum(counts,axis = axis )
    non_zero_counts = total_counts!=0
    combined_mean = np.sum(means*counts,axis = axis)
    combined_mean[non_zero_counts] /= total_counts[non_zero_counts]
    return combined_mean,total_counts

def masked_variance(data,mask,axis = 0):
    var = np.var(data,where=mask,axis=axis)
    counts = np.sum(mask,axis = axis)
    nan_mask = np.isnan(var)
    var[nan_mask]=0
    return var,counts

def combine_variances_ND(_vars,means,counts,axis=0):
    _vars = np.moveaxis(_vars,axis,0)
    means = np.moveaxis(means,axis,0)
    counts = np.moveaxis(counts,axis,0)
    combined_variance = _vars[0].copy()
    combined_mean = means[0].copy()
    combined_counts = counts[0].copy()
    for var,mean,c in zip(_vars[1:],means[1:],counts[1:]):
        next_combined_mean,denom = combine_means_2D([combined_mean,mean],[combined_counts,c])
        non_zero = denom !=0 
        N1 = combined_counts[non_zero]/denom[non_zero]
        N2 = c[non_zero]/denom[non_zero]
        N3 = (combined_counts*c)[non_zero]/(denom**2)[non_zero]

        a = N1*combined_variance[non_zero]
        combined_variance[non_zero]=a+N2*var[non_zero]+N3*np.square(combined_mean-mean)[non_zero]
        combined_mean = next_combined_mean
        combined_counts = denom
    return combined_variance,combined_mean,combined_counts


### connected area ###
#Simple algorithm to find te area of equal values in a 2d array/image
def _get_connected_area_periodic(image,point,step,value,shape,connected_points,visited_mask):
    #print(point)
    #print(im[point])
    if visited_mask[point]:
        return connected_points
    if image[point]==value:
        connected_points+=[point]
        visited_mask[point]=True
        
        orth_step = ((step[0]+1)%2,(step[1]+1)%2)
        next_steps = (step,orth_step,(-orth_step[0],-orth_step[1]))
        next_points = (((point[0]+s[0])%shape[0],(point[1]+s[1])%shape[1]) for s in next_steps)
        for p,s in zip(next_points,next_steps):
            _get_connected_area_periodic(image,p,s,value,shape,connected_points,visited_mask)
    return connected_points

def _get_connected_area(image,point,step,value,shape,connected_points,visited_mask):
    #print(point)
    #print(im[point])   
    if visited_mask[point]:
        return connected_points

    if image[point]==value:
        connected_points+=[point]
        visited_mask[point]=True
        
        orth_step = ((step[0]+1)%2,(step[1]+1)%2)
        next_steps = (step,orth_step,(-orth_step[0],-orth_step[1]))
        next_points = ((point[0]+s[0],point[1]+s[1]) for s in next_steps)
        for p,s in zip(next_points,next_steps):
            if (p[0]>=0 and p[0]<shape[0] and p[1]>=0 and p[1]<shape[1]):
                _get_connected_area(image,p,s,value,shape,connected_points,visited_mask)
    return connected_points

def find_connected_component(image,start,periodic = False,return_mask = False):
    start=tuple(start)
    value = image[start]
    shape = image.shape
    next_steps= np.array(((1,0),(-1,0),(0,1),(0,-1)))
    
    if periodic:
        find_connected_points = _get_connected_area_periodic
        next_points = (next_steps+np.array(start)[None,:])%np.array(shape)[None,:]
    else:
        find_connected_points = _get_connected_area
        next_points = next_steps+np.array(start)[None,:]
        negative_mask = np.sum(next_points<0,axis = -1).astype(bool)
        to_high_mask = np.sum(next_points>=np.array(shape)[None,:],axis = -1).astype(bool)
        point_mask = ~(negative_mask | to_high_mask)
        next_points = next_points[point_mask]
        next_steps = next_steps[point_mask]
    
    
    visited_mask = np.zeros(shape,dtype = bool)
    connected_points=[tuple(start)]
    for point, step in zip(next_points,next_steps):
        find_connected_points(image,tuple(point),step,value,shape,connected_points,visited_mask)
    if return_mask:
        mask = np.zeros_like(visited_mask)
        mask[tuple(np.stack(connected_points,axis = 1))]=True
        return connected_points,mask
    else:
        return connected_points

def get_test_function(support=[-1,1],slope=1):
    center = np.mean(support)
    size = support[1]-center
    #print('size = {}'.format(size))
    #print('center = {}'.format(center))
    def test_function(data):
        non_zero = ((data>support[0]) & (data<support[1]))
        values = np.zeros_like(data)
        values[non_zero]=np.exp(-slope*size**2/(size**2 - (data[non_zero]-center)**2))
        return values
    return test_function

def get_test_function2(support=[-1,-0.5,0.5,1],slope=1):
    assert support[0]<upport[1]<upport[2]<upport[3]
    #print('size = {}'.format(size))
    #print('center = {}'.format(center))
    def f(x):
        return np.exp(-slope*(1/(1-(x-1)**2)-1))
    def test_function(data):
        non_zero = ((data>support[0]) & (data<support[-1]))
        values = np.zeros_like(data)
        x=data[non_zero]
        values[non_zero]=f((x-support[0])/(support[1]-support[0]))*f((x-support[3])/(support[3]-support[2]))
        return values
    return test_function

#### solve procrustes problem ###

def solve_procrustes_problem(V1,V2):
    '''
    Finds the unitary matrix U that minimizes ||V1-V2U||.
    by computing the svd of V2^\dagger V1
    '''
    #log.info(f'trasposed shapes = {V2.T.shape} @ {V1.shape}')
    return np.matmul(*np.linalg.svd((V2.conj().T) @ V1,full_matrices = False)[::2])

def midpoint_rule(samples,uniform_sampling_points,**kwargs):
    step = uniform_sampling_points[1]-uniform_sampling_points[0]
    #N = len(uniform_sampling_points)
    integral = step*np.sum(samples,**kwargs)
    return integral


def psd_back_substitution(cn,ppm):
    bl = np.zeros(cn.shape,dtype = complex)
    cn = cn.copy()
    pm = ppm.copy()
    for i in np.arange(0,cn.shape[-1])[::-1]:
        bl[...,i] = nearest_positive_semidefinite_matrix(cn[...,-1]/pm[...,-1,-1])
        cn = (cn[...,:-1]-bl[...,i,None]*pm[...,:-1,-1])
        pm = pm[...,:-1,:-1]
    return bl

def back_substitution(cn,ppm):
    bl = np.zeros(cn.shape,dtype = complex)
    cn = cn.copy()
    pm = ppm.copy()
    for i in np.arange(0,cn.shape[-1])[::-1]:
        bl[...,i] = cn[...,-1]/pm[...,-1,-1]
        cn = (cn[...,:-1]-bl[...,i,None]*pm[...,:-1,-1])
        pm = pm[...,:-1,:-1]
    return bl
