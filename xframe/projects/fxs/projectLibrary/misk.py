import numpy as np
import numpy.ma as mp
import logging
import sys

from xframe.library import pythonLibrary as pyLib
from xframe.library.gridLibrary import NestedArray
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library.mathLibrary import _pi_in_q__to__reciprocity_coefficient
from xframe.library.pythonLibrary import get_L2_cache_split_parameters
from xframe.library import mathLibrary as mLib
from xframe.interfaces import DatabaseInterface,PresenterInterface
from xframe.presenters.matplotlibPresenter import heatPolar2D
from xframe import settings
pres=heatPolar2D()
log=logging.getLogger('root')
array=np.array

def getAnalysisRecipeFacotry(database,presenters):
    def getAnalysisOperatorDict(database,presenters):
        def getDataBaseOperatorDict(database):
            interfaceFunctionNames=pyLib.getFunctionNames(DatabaseInterface)
            allMethodsDict=pyLib.getMethods(database, _type='dict')
            dataDict={functionName+'@Database':allMethodsDict[functionName] for functionName in interfaceFunctionNames}
            return dataDict
        
        def getPresenterOperatorDict(presenters):
            interfaceFunctionNames=pyLib.getFunctionNames(PresenterInterface)
            presenterDict={}
            for presenter in presenters:
                Name=type(presenter).__name__
                methodsDict=pyLib.getMethods(presenter, _type='dict')
                tempDict={functionName+'@'+Name:methodsDict[functionName] for functionName in interfaceFunctionNames}
                presenterDict.update(tempDict)
            return presenterDict
    
        operatorDict={}
        analysisLibModule=__import__('analysisLibrary')  #fromlist is neccessary otherwise it only imports toplevel module analysis instead of analysis.analysusLibrary
        libraryDict=pyLib.getFunctions(analysisLibModule, _type='dict',recoursive=True)
#        print('this is the libraryDict: {} \n'.format(libraryDict))
#        print('current Module name = {}'.format(currentModule))
        operatorDict.update(libraryDict)
        mathDict=pyLib.getFunctions(mLib, _type='dict')
        operatorDict.update(mathDict)
        dataDict=getDataBaseOperatorDict(database)
        operatorDict.update(dataDict)        
        presenterDict=getPresenterOperatorDict(presenters)
        operatorDict.update(presenterDict)        
        return operatorDict

    operatorDict= getAnalysisOperatorDict(database,presenters)
    recipeFactory=pyLib.RecipeFactory(operatorDict)
    return recipeFactory


def get_analysis_process_factory(database,presenters):
    def getAnalysisOperatorDict(database,presenters):
        def getDataBaseOperatorDict(database):
            interfaceFunctionNames=pyLib.getFunctionNames(DatabaseInterface)
            allMethodsDict=pyLib.getMethods(database, _type='dict')
            dataDict={functionName+'@Database':allMethodsDict[functionName] for functionName in interfaceFunctionNames}
            return dataDict
        
        def getPresenterOperatorDict(presenters):
            interfaceFunctionNames=pyLib.getFunctionNames(PresenterInterface)
            presenterDict={}
            for presenter in presenters:
                Name=type(presenter).__name__
                methodsDict=pyLib.getMethods(presenter, _type='dict')
                tempDict={functionName+'@'+Name:methodsDict[functionName] for functionName in interfaceFunctionNames}
                presenterDict.update(tempDict)
            return presenterDict
    
        operatorDict={}
        #from .. import analysisLibrary as analysisLibModule
        
        libraryDict=pyLib.getFunctions(sys.modules[__name__], _type='dict',recoursive=True)
#        print('this is the libraryDict: {} \n'.format(libraryDict))
#        print('current Module name = {}'.format(currentModule))
        operatorDict.update(libraryDict)
        mathDict=pyLib.getFunctions(mLib, _type='dict')
        operatorDict.update(mathDict)
        #dataDict=getDataBaseOperatorDict(database)
        #operatorDict.update(dataDict)        
        #presenterDict=getPresenterOperatorDict(presenters)
        #operatorDict.update(presenterDict)        
        return operatorDict

    operatorDict= getAnalysisOperatorDict(database,presenters)
    recipeFactory=pyLib.RecipeFactory(operatorDict)
    return recipeFactory



def returnGrid_deleteFirstDimension(oldGrid):
    def deleteFirstCoordinate(point):
        return np.delete(point,0)
    reducedGridArray=np.array(tuple(map(deleteFirstCoordinate,oldGrid[0]))).reshape(oldGrid.array.shape[1:]+(oldGrid.dimension-1,))
    reducedGridArray=pyLib.getArrayOfArray(reducedGridArray)
    reducedGrid=Grid(reducedGridArray,oldGrid.gridType)
    return reducedGrid

def save_to_dict(dictionary,keys,mode,data):
    if not isinstance(keys,list):
        keys = [keys]
    folder=dictionary
    try:
        for key in keys[:-1]:
            folder = folder[key]
        key = keys[-1]
        if mode == 'append':
            dataset = folder.get(key,[])
            dataset.append(data)
            folder[key]=dataset
        elif mode=='overwrite':            
            folder[key] = data
        elif mode == 'iterative_append':
            folder = folder[key]
            #log.info('saving {}'.format(data))
            #log.info('to dict {}'.format(folder))
            for dkey in data:                
                dataset = folder.get(dkey,[])
                dataset.append(data[dkey])
                folder[dkey]=dataset
        elif mode == 'iterative_overwrite':
            folder = folder[key]
            for dkey in data:                
                folder[dkey]=data[dkey]
        else:
            folder[key] = data
    except AttributeError as e:
        log.error('Saving to dictionary {} failed with keys {}.'.format(dictionary,keys))            
    return data

def load_from_dict(dictionary,keys):
    if not isinstance(keys,list):
        keys = [keys]
    value=dictionary
    try:
        for key in keys:
            value = value[key]
    except AttributeError as e:
        log.error('loading from dictionary {} failed with keys {}.'.format(dictionary,keys))            
    return value

def generate_load_from_dict(dictionary):
    def load_from_dict(key):
        value=dictionary[key]
        return value
    return load_from_dict

        
def copy(data):
    #np.array creates copy by default and is slightly faster than data.copy()
    return array(data)


def _generate_square_default(out_array):
    mult = np.multiply
    np_square = np.square
    if out_array.dtype == np.dtype(complex):
        def square(data):
            return mult(data,data.conj(),out = out_array)
    else:
        def square(data):
            return np_square(data,out = out_array)
    return square
def _generate_square_cache_aware(out_array,L2_cache):
    data_shape = out_array.shape
    data_type = out_array.dtype
    splitting_dimension,step = get_L2_cache_split_parameters(data_shape,data_type,L2_cache)

    mult = np.multiply
    
    def square_1_loop(data):
        for i in range(0,data_shape[0],step):
            i2 = i+step
            d = data[i:i2]
            o = out_array[i:i2]
            mult(d,d.conj(),out = o)
        return out_array
    def square_2_loop(data):
        for i in range(data_shape[0]):
            for j in range(0,data_shape[1],step):
                j2 = j+step
                d = data[i,j:j2]
                o = out_array[i,j:j2]
                mult(d,d.conj(),out = o)
        return out_array
    def square_3_loop(data):
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for k in range(0,data_shape[2],step):
                    k2 = k+step
                    d = data[i,j,k:k2]
                    o = out_array[i,j,k:k2]
                    mult(d,d.conj(),out = o)
        return out_array
    if (splitting_dimension <= -1) or (out_array.dtype != np.dtype(complex)):
        square = _generate_square_default(out_array)
    elif splitting_dimension == 0:
        square = square_1_loop
    elif splitting_dimension == 1:
        square = square_2_loop
    elif splitting_dimension == 2:
        square = square_3_loop
    return square


def generate_square(data_shape,data_type,cache_aware=False,L2_cache=256):
    out = np.zeros(data_shape,dtype = data_type)
    if not cache_aware:
        square = _generate_square_default(out)
    else:
        square = _generate_square_cache_aware(out,L2_cache)
    return square



def _generate_abs_value_default(out_array):
    sqrt = np.sqrt
    def abs_value(data):
        return sqrt((data*data.conj()).real,out=out_array)
    return abs_value
def _generate_abs_value_cache_aware(out_array,L2_cache):
    data_shape = out_array.shape
    data_type = out_array.dtype
    splitting_dimension,step = get_L2_cache_split_parameters(data_shape,data_type,L2_cache)

    mult = np.multiply
    sqrt = np.sqrt
    
    def abs_value_1_loop(data):
        for i in range(0,data_shape[0],step):
            i2 = i+step
            d = data[i:i2]
            o = out_array[i:i2]
            sqrt(mult(d,d.conj(),out = o).real,out=o)
        return out_array
    def abs_value_2_loop(data):
        for i in range(data_shape[0]):
            for j in range(0,data_shape[1],step):
                j2 = j+step
                d = data[i,j:j2]
                o = out_array[i,j:j2]
                sqrt(mult(d,d.conj(),out = o).real,out=o)
        return out_array
    def abs_value_3_loop(data):
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for k in range(0,data_shape[2],step):
                    k2 = k+step
                    d = data[i,j,k:k2]
                    o = out_array[k:k2]
                    sqrt(mult(d,d.conj(),out = o).real,out=o)
        return out_array
    
    if splitting_dimension <= -1:
        abs_value = _generate_abs_value_default(out_array)
    elif splitting_dimension == 0:
        abs_value = abs_value_1_loop
    elif splitting_dimension == 1:
        abs_value = abs_value_2_loop
    elif splitting_dimension == 2:
        abs_value = abs_value_3_loop
    return abs_value

def generate_absolute_value(data_shape,data_type,cache_aware=False,L2_cache=256):
    out = np.zeros(data_shape,dtype = data_type)
    if not cache_aware:
        abs_value = _generate_abs_value_default(out)
    else:
        abs_value = _generate_abs_value_cache_aware(out,L2_cache)
    return abs_value


def generate_calc_center_old(real_grid):
    dim = real_grid.n_shape[0]
    cart_grid = spherical_to_cartesian(real_grid)    
    def calc_center(density):
        density_sum=np.sum((density.real*real_grid[...,0]).flatten())
        if density_sum==0:
            density_sum=1
        center=np.sum((cart_grid*(real_grid[...,0]*density.real)[...,None]).reshape(-1,dim),axis=0)/density_sum
        log.info('center={}'.format(center))
        center=cartesian_to_spherical(center)
        #polygon=mLib.SampleShapeFunctions.get_disk_function(0.05,amplitude_function=lambda points: np.full(points.shape[:-1],0.01),coordSys=1,center=center)
        #temp_density=np.abs(density)+polygon(real_grid)
        #pres.present(temp_density)
        log.info('center={}'.format(center))
        return center        
    return calc_center

def generate_calc_center(real_grid):
    cart_grid = spherical_to_cartesian(real_grid)
    dim = real_grid[:].shape[-1]

    if dim ==2:
        si = mLib.PolarIntegrator(real_grid[:])
    elif dim == 3 :
        si = mLib.SphericalIntegrator(real_grid[:])
        
    def calc_center(density):            
        density_integral = si.integrate(density.real)            
        if density_integral==0:
            density_integral=1
        center = si.integrate(cart_grid[:]*density[...,None].real)/density_integral
        #log.info('ccenter={}'.format(center))
        center=cartesian_to_spherical(center)
        return center        
    return calc_center


def check_negative_intensity(intensity):
    n_negative_values = np.sum(intensity.real<0)/np.prod(intensity.shape)
    #log.info('negative fraction = {}'.format(n_negative_values))

def diff(a,b):
    return a-b
def add(a,b):
    #mask = (b*b.conj()).real<1e-3*(a*a.conj()).real
    #log.info('masked fraction = {}'.format(np.sum(mask)/np.prod(mask.shape)))
    #log.info('a type = {}, b_type = {}'.format(type(a),type(b)))
    return a+b
def add_above_zero_index(a,b):
    result = a+b
    result[0]=a[0]
    return result
def add_low(a,b):
    mask = (b*b.conj()).real<1e-3*(a*a.conj()).real
    log.info('masked fraction = {}'.format(np.sum(mask)/np.prod(mask.shape)))
    return a #np.where(mask,a+b,a)

def debug(a,b):
    log.info(' {}'.format(a[0][:10]))
    #for aa in a :
    #    aa[:]=0
    a[0][:]=10
    #log.info(' {}'.format(a[0][:10]))
    return a
    #args = [a,b]
    #log.info('Got {} arguments with types {}'.format(len(args),[type(a) for a in args]))
    #for i,a in enumerate(args):
    #    if isinstance(a,np.ndarray):
    #        log.info('Argument {} is array of shape {} and dtype {}'.format(i,a.shape,a.dtype))
    
def project(a):
    log.info('old 0 coeff = {}'.format(a[0][:10]))
    a[0][:]=1.9e1
    log.info('new 0 coeff = {}'.format(a[0][:10]))
    return a

def project2(new_I,I,d):
    #log.info('first new I = {}'.format(new_I[1,1,:10]))
    #log.info('first old I = {}'.format(I[1,1,:10]))
    n = d/np.sqrt(I)
    n *= np.sqrt(new_I)
    return n

def sqrt2(a):
    #log.info(a)
    aa = np.sqrt(a)
    b = aa*aa.conj()
    return b

def check(a):
    if np.isinf(a).any():
        log.info('inf values found')
    if np.isnan(a).any():
        log.info('nan values found')

def check3(a):
    if np.isinf(a).any():
        log.info('inf values found 3')
    if np.isnan(a).any():
        log.info('nan values found 3')

def check2(Ilm):
    for I in Ilm:
        if np.isinf(I).any():
            log.info('inf values found 2')
        if np.isnan(I).any():
            log.info('nan values found 2')
            

def _get_reciprocity_coefficient(ft_opt):
    pi_in_q = ft_opt.get('pi_in_q',None)
    
    if isinstance(pi_in_q,bool):
        reciprocity_coefficient = _pi_in_q__to__reciprocity_coefficient(pi_in_q)
    else:
        reciprocity_coefficient = ft_opt.get('reciprocity_coefficient',np.pi)
    return reciprocity_coefficient

