import numpy as np
import numpy.ma as mp
import logging

from xframe.library import pythonLibrary as pyLib
from xframe.library.gridLibrary import NestedArray
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library import mathLibrary as mLib
from xframe.analysis.interfaces import DatabaseInterface
from xframe.analysis.interfaces import PresenterInterface
from xframe.presenters.matplolibPresenter import heatPolar2D
pres=heatPolar2D()
log=logging.getLogger('root')

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



def returnGrid_deleteFirstDimension(oldGrid):
    def deleteFirstCoordinate(point):
        return np.delete(point,0)
    reducedGridArray=np.array(tuple(map(deleteFirstCoordinate,oldGrid[0]))).reshape(oldGrid.array.shape[1:]+(oldGrid.dimension-1,))
    reducedGridArray=pyLib.getArrayOfArray(reducedGridArray)
    reducedGrid=Grid(reducedGridArray,oldGrid.gridType)
    return reducedGrid


def generate_save_to_dict(dictionary):
    def save_to_dict(key,mode,data):
        if mode == 'append':
            if key in dictionary:
                dictionary[key].append(data)
            else:
                dictionary[key]=[data]
        elif mode=='overwrite':            
            dictionary[key]=data
        else:
            dictionary[key]=data
        return data
    return save_to_dict
def generate_load_from_dict(dictionary):
    def load_from_dict(key):
        value=dictionary[key]
        return value
    return load_from_dict

        
def copy(data):
    return data.copy()

def square_grid(data):
    return (data*data.conjugate()).real

def abs_value(data):
    return np.abs(data)


def generate_calc_center(real_grid):
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
    
    

class Selection:
    def __init__(self,name,selection_dict):
        self.name = name
        self.data_range= selection_dict.get('selection',slice(None))
        self.range_type=self.data_range.__class__
        if self.range_type != slice:
            self.data_range = np.asarray(self.data_range)
            self.range_type=self.data_range.__class__
        self.mode = selection_dict.get('mode','relative')
    def mask(self,input_data):
        if self.range_type == slice:
            mask = self.slice_to_mask(input_data)
        else:
            mask = self.numpy_to_mask(input_data)
        return mask
    def slice_to_mask(self,input_data):
        u_input = np.unique(input_data)
        _slice = self.data_range
        if self.mode == 'relative':
            selection_array = u_input[_slice]
        else:
            _min,_max = u_input.min(),u_input.max()
            start = _slice.start
            stop = _slice.stop
            step = _slice.step
            if start == None:
                start = _min
            if stop == None:
                stop == _max
            if step == None:
                step = int((stop -start)/abs(stop-start))
            selection_array = np.array(range(start,stop,step))
        mask = self._array_to_mask(input_data,selection_array)
        return mask
    
    def numpy_to_mask(self,input_data):
        u_input = np.unique(input_data)
        _array = self.data_range
        if self.mode == 'relative':
            try:
                selection_array = u_input[_array]
            except IndexError as e:
                log.error('Selection of {}: Specified relative indices {} are out of range for the {} unique {}. Continue with empty selection.'.format(self.name,_array,len(u_input),self.name))
                selection_array = np.array([])
        else:
            selection_array = _array
        mask = self._array_to_mask(input_data,selection_array)
        return mask
    
    def _array_to_mask(self,input_data,array):
        return np.isin(input_data,array)

        
