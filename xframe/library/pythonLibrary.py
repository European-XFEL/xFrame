import inspect
import os
import ctypes
from ctypes.util import find_library
import time
import numpy as np
import numpy.ma as mp
from scipy.interpolate import griddata
from functools import partial
from itertools import repeat
from types import SimpleNamespace
import logging
from io import StringIO
import sys
from contextlib import contextmanager

sa = 'Shared_Array module'

log=logging.getLogger('root')

doublePrecision=1e-16

def xprint(txt):
    if log.level > logging.INFO:
        print(txt)
    log.info(txt)
    
    
def hash_numpy_in_tuple(_tuple):       
    tmp=[]
    for item in _tuple:
        if isinstance(item,np.ndarray):
            tmp.append(hash(item.data.tobytes()))
        else:
            tmp.append(item)
    return tuple(tmp)
    
    
def load_clib(libraryName, mode = None):
    libraryPath = find_library(libraryName)
    if libraryPath is None:
        log.info('Did not find library with name "{}". Try interpreting it as the librarys path.'.format(libraryName))
        libraryPath=libraryName
    try:
        if mode is None:
            library = ctypes.CDLL(libraryPath)
        else:
            library = ctypes.CDLL(libraryPath, mode)
    except OSError as e:
        log.error(e)
        log.error('current basepath is: {}'.format(os.getcwd()))
        raise e
    return library

def plugVariableIntoFunction(function,variableSpecifier):
    variable=variableSpecifier[0]
    varPosition=variableSpecifier[1]
    def newFunction(*args,**kwargs):
        arguments=args[0:varPosition]+(variable,)+args[varPosition:]
        output=function(*arguments,**kwargs)
        return output
    
    remainingParameters=tuple(inspect.signature(function).parameters.values())[:varPosition]+tuple(inspect.signature(function).parameters.values())[varPosition+1:]
    newFunction.__signature__=inspect.signature(newFunction).replace(parameters=remainingParameters)
    return newFunction

def appendArgumentsToFunction(function,*args,**kwargs):
    def newFunction(firstArgument):
        output=function(firstArgument,*args,**kwargs)
        return output
    #applies the parameter signature of function to newFunction
    firstParameter=tuple(inspect.signature(function).parameters.values())[0]
    newFunction.__signature__=inspect.signature(newFunction).replace(parameters=[firstParameter])
    return newFunction

def partialSums(array1D):
    def sum(elementNumber):
        return np.sum(array1D[:elementNumber+1])
    elementNumbers=np.arange(len(array1D))
    listOfSums=np.array(list(map(sum,elementNumbers)))
    return listOfSums

def listToDict(list):
    dict={list[i]:list[i+1] for i in range(0,len(list),2)}
    return dict

def listOfTuplesToDict(list):
    dict={tuple[0]:tuple[1] for tuple in list}
    return dict

def invertNumpyIndex(index):
    invertedIndex=np.zeros(len(index),dtype=int)
    for pos,i in enumerate(index):
        invertedIndex[i]=pos
    return invertedIndex

def getFunctions(_object, _type='list',recoursive=False):
    functions=[]
    if recoursive:
        submodules=inspect.getmembers(_object,predicate=inspect.ismodule)
        for module in submodules:
            functions+=inspect.getmembers(module[1],predicate=inspect.isfunction)
    else:
        functions+=inspect.getmembers(_object,predicate=inspect.isfunction)
    if _type=='dict':
        functions=listOfTuplesToDict(functions)
    return functions

def getFunctionNames(_object):
    functions=getFunctions(_object)
    functionNames=[tuple[0] for tuple in functions]
    return functionNames

def getMethods(_object, _type='list'):
    methods=inspect.getmembers(_object,predicate=inspect.ismethod)
    if _type=='dict':
        methods=listOfTuplesToDict(methods)
    return methods

def getMethodNames(_object):
    methods=getMethods(_object)
    methodNames=[tuple[0] for tuple in methods]
    return methodNames

def measureTime(func):
    def wrapper(*args,**kwargs):
#        log.info('kwargs={}'.format(kwargs))
        startTime=time.time()
        output=func(*args,**kwargs)
        duration=time.time()-startTime
        xprint('function call took {} seconds'.format(duration))
        return output
    return wrapper

def isIterable(object):
    try:
        iter(object)
    except Exception:
        return False
    else:
        return True
def generateMesh(*args,joiningDimension=2):
    args=list(map(np.asarray,args))
    mesh=np.stack(np.meshgrid(*args,indexing='ij'),joiningDimension)
    return mesh

def meshstack(array1,array2):
    results=[]
    for a,b in zip(array1,array2):
        temp=np.full((len(b),len(a)),a)
        result=np.concatenate((temp,np.expand_dims(b,axis=1)),axis=1)
        results.append(result)
    return results

def flattenList(_list):
    elements=[]
    for sub_list in _list:
        if isinstance(sub_list,list): 
            newElements=flattenList(sub_list)
            elements +=  newElements
        else:
            elements += [sub_list]
    return elements



def uniformGridGetStepSizes(gridArray):
    gridShape=gridArray.shape
#    log.info('grid shape={}'.format(gridShape))
    if gridArray.dtype==np.object:
        gridDimension=len(gridShape)
    else:
        gridDimension=gridShape[-1]

#    log.info('grid Dimension={}'.format(gridDimension))

    secondPointIndices=np.eye(gridDimension,dtype=np.int)
    thirdPointIndices=secondPointIndices*2
    stepSizes=gridArray[tuple(thirdPointIndices)]-gridArray[tuple(secondPointIndices)]
    indices=np.repeat(np.expand_dims(np.arange(gridDimension),0),2,axis=0)
#    log.info('indices={}'.format(indices))
#    log.info('steps={}'.format(np.array(tuple(stepSizes))))
    stepSizes=np.array(tuple(stepSizes))[tuple(indices)]
#    log.info('stepsizes dtype={}'.format(stepSizes.dtype))
    return stepSizes

def uniformGridToFunction(dataPoints,gridSpec):
    #axisTypes is an np array of length grid Dimension.
    origin=gridSpec[0]
    stepSizes=gridSpec[1]
    log.info('origin={} stepSizes={}'.format(origin,stepSizes))
    gridShape=dataPoints.shape
    gridShapeArray=np.array(gridShape,dtype=np.int)
    gridDimension=len(gridShape)

#    if isinstance(axisTypes,bool):
#        axisTypes=np.zeros(gridDimension)
#    else:
#        axisTypes=np.asarray(axisTypes)

#    axisIsPeriodic= (axisTypes==1)
#    axisHasNoEndpoint= (axisTypes==2)

    log.info('gridDimension={} step Sizes={}'.format(gridDimension,stepSizes))
    def dataFunction(*args):        
        point=np.array([args[index] for index in range(gridDimension)])
        point=point-origin
#        log.info('point={}'.format(point))
        dataIndex=np.floor(point/stepSizes).astype(np.int)

#        dataIndex[axisIsPeriodic]=dataIndex[axisIsPeriodic]%gridShapeArray[axisIsPeriodic]
 #       noEndpointIndex=dataIndex[axisHasNoEndpoint]
 #       dataIndex[axisHasNoEndpoint]=np.where(noEndpointIndex==gridShape,noEndpointIndex-1,noEndpointIndex)
        
        isOutsideOfGrid= (gridShape-dataIndex<=0).any()
        if isOutsideOfGrid:
            value=0
            log.warning('grid Function is called at point {}*{} outside of grid. Returning 0!'.format(dataIndex,stepSizes))
        else:
            value=dataPoints[tuple(dataIndex)]        
        return value
    return dataFunction
    
def getGridByXYValues(xValues,yValues):
    def xfunc(*args):
        return xValues
    def yfunc(*args):
        return yValues
    gfact=gridFactory()
    newGrid=gfact.constructGrid([xfunc,yfunc],'uniform')
    return newGrid

def restoreOldParameters(oldFunction,newFunction):
    oldParameters=inspect.signature(oldFunction).parameters.values()
    newFunction.__signature__=inspect.signature(newFunction).replace(parameters=oldParameters)
    return newFunction

def selectElementOfFunctionOutput_decorator(function,elementIndex):
    def newFunction(*args,**kwargs):
        output=function(*args,**kwargs)
        log.info('len output ={}'.format(len(output)))
        log.info('shapes of outputs ={}'.format([x.shape for x in output]))
        return output[elementIndex]

    #resore original parameters
    newFunction=restoreOldParameters(function,newFunction)
    return newFunction

def getFirstElement(subscriptable):
    return subscriptable[0]

def getLastElement(subscriptable):
    return subscriptable[-1]

def swapElements(array,index1,index2):
    array=np.asarray(array)
    value1=array[index1]
    array[index1]=array[index2]
    array[index2]=value1
    return array

def concatenateDecorators(decoratorList):
    def newDecorator(function):
        newFunction=function
        for decorator in decoratorList:
            newFunction=decorator(newFunction)
        return newFunction
    return newDecorator
    
def getArrayOfArray(array):
#    log.info('input array = {}'.format(array))
    shape=array.shape
    if len(shape)!=1:
        array=array.reshape(-1,shape[-1])
        
    arrayOfArrays=np.frompyfunc(array.__getitem__,1,1)(range(array.shape[0]))
    arrayOfArrays=arrayOfArrays.reshape(shape[:-1])
    return arrayOfArrays

def ArrayOfArrays2Array(array):
    shape=array.shape
    new_array=np.array(tuple(array.flatten()))
    return newArray

def optional_arg_decorator(fn):
    def wrapped_decorator(*args,**kwargs):
        if len(kwargs) == 0 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, **kwargs)
            return real_decorator
    return wrapped_decorator


#########
##MP utils##


def splitMPArguments(argArrays,nProcesses,mode='sequential',split_together=False):
    #log.info('argArrays={}'.format(argArrays))
    if isinstance(argArrays,np.ndarray):
    #    log.info('yep')
        argArrays=[argArrays]
    indiceArrays=[np.arange(len(array),dtype=int) for array in argArrays]
    arg_legths_equal= len(set(tuple(len(array) for array in argArrays))) == 1
    dtype_list = [arg.dtype for arg in argArrays]
    n_args = len(argArrays)
    #log.info('len argArrays ={}'.format(len(argArrays)))
    if split_together:
        try:
            assert arg_legths_equal,'split_together requires all argArrays to have the same length but the length are {}'.format(tuple(len(array) for array in argArrays))           
        except AssertionError as e:
            log.error(e)
            raise
        arguments=tuple(argArrays)
        argIndices=indiceArrays
       
    else:
        #    log.info('indiceArrays={}'.format(indiceArrays))
        argIndices=tuple(reversed(tuple(dataset.flatten().astype(np.int) for dataset in np.meshgrid(*reversed(indiceArrays)) )))
        arguments=tuple(argument[indices] for argument,indices in zip(argArrays,argIndices))
    nArguments=len(arguments[0])
    #print('arguments = {}'.format(argArrays))    
    log.info('nArgs = {} nProcess {}'.format(nArguments,nProcesses))
    if nArguments<nProcesses:
        nProcesses=nArguments
    argPerProcess=nArguments//nProcesses    

    #    log.info('nArguments={}, nProcesses={}'.format(nArguments,nProcesses))
    #print(mode)

    if mode=='sequential':    
        splitPoints=[int(argPerProcess*process) for process in range(1,nProcesses)]
        splittedArguments=tuple(np.split(argument,splitPoints) for argument in arguments)
        splittedIndices=tuple(np.split(argIndice,splitPoints) for argIndice in argIndices)
    if mode=='modulus':
        def splitArgument(argument):
            dtype=argument.dtype
            def returnSlizedArgument(processID):
                arg_slice = np.array(argument[processID::nProcesses],dtype=dtype)
 #               log.info('slice shape = {}'.format(arg_slice.shape))
                return arg_slice
            splittedArgument=np.array(tuple(map(returnSlizedArgument,range(nProcesses))),dtype=np.object)
            #log.info('splittedArgument={}'.format(splittedArgument))
            return splittedArgument
        
        splittedArguments=tuple(splitArgument(argument) for argument in arguments)
        splittedIndices=tuple(splitArgument(argIndice) for argIndice in argIndices)
    #      log.info('dtype={} splitted Indices=\n {}'.format(splittedIndices.dtype,splittedIndices))
    
    arguments_per_process = list(zip(*splittedArguments))
    indices_per_process = list(zip(*splittedIndices))    
    return {'arguments':arguments_per_process,'indices':indices_per_process}


def splitMPArguments_old(argArrays,nProcesses,mode='sequential'):    
    if isinstance(argArrays,np.ndarray):
    #    log.info('yep')
        argArrays=[argArrays]
        
   
    indiceArrays=[np.arange(len(array),dtype=int) for array in argArrays]
#    log.info('indiceArrays={}'.format(indiceArrays))

    if len(argArrays) == 1:

        arguments=(argArrays[0],)
        argIndices=(np.arange(len(arguments)),)
    else:
        arguments=tuple(reversed(tuple(dataset.flatten() for dataset in np.meshgrid(*reversed(argArrays)) )))
        argIndices=tuple(reversed(tuple(dataset.flatten().astype(np.int) for dataset in np.meshgrid(*reversed(indiceArrays)) )))
    nArguments=arguments[0].shape[0]
    if nArguments<nProcesses:
        nProcesses=nArguments
    argPerProcess=np.floor(nArguments/nProcesses)

#    log.info('nArguments={}, nProcesses={}'.format(nArguments,nProcesses))

    if mode=='sequential':    
        splitPoints=[int(argPerProcess*process) for process in range(1,nProcesses) ]
        splittedArguments=np.array([np.split(argument,splitPoints) for argument in arguments],dtype=np.object)
        splittedIndices=np.array([np.split(argIndice,splitPoints) for argIndice in argIndices],dtype=np.object)
    if mode=='modulus':
        def splitArgument(argument):
            dtype=argument.dtype
            def returnSlizedArgument(processID):
                arg_slice = np.array(argument[processID::nProcesses],dtype=dtype)
 #               log.info('slice shape = {}'.format(arg_slice.shape))
                return arg_slice
            splittedArgument=np.array(tuple(map(returnSlizedArgument,range(nProcesses))),dtype=np.object)
            return splittedArgument
        
        splittedArguments=np.array([splitArgument(argument) for argument in arguments])
        splittedIndices=np.array([splitArgument(argIndice) for argIndice in argIndices])
  #      log.info('dtype={} splitted Indices=\n {}'.format(splittedIndices.dtype,splittedIndices))
    argumentsPerProcessList=np.swapaxes(splittedArguments,0,1)
    indicesPerProcessList=np.swapaxes(splittedIndices,0,1)
    #    indicesPerProcessList2=[np.array(indices,dtype=np.int) for indices in indicesPerProcessList]
    #log.info('arguments per process=\n{}'.format(argumentsPerProcessList))
    return {'arguments':argumentsPerProcessList,'indices':indicesPerProcessList}

def assembleResults(resultDict,mode='sequential'):
#    log.info('result 0 shape={}'.format(results[0].shape))
    nProcesses=len(resultDict)
    nCallsPerProcess=np.array(tuple(len(resultDict[processID]) for processID in range(nProcesses)))
    nTotalCalls=np.sum(nCallsPerProcess)
#    log.info('nProcesses={}'.format(nProcesses))
#    log.info('nCallsPerProcess={}'.format(nCallsPerProcess))
#    log.info('nResults={}'.format(nResults))
    
    if mode=='sequential':        
        result=()
        for processID in range(nProcesses):
  #            log.info('result part={}'.format(resultDict[processID]))
            result+=tuple(resultDict[processID])
        
    elif mode=='modulus':
        result=list(range(nTotalCalls))
        for processID in range(nProcesses):
            result[processID::nProcesses]=list(resultDict[processID])
        
    result=np.array(result)
    
    return result


def get_mp_function():
    pass


def getMPFunction(function,arguments,args=[],callWithMultipleArguments=True,output_file = False,**kwargs):
    repeatedArguments=[repeat(arg) for arg in args]
    #log.info('arguments={}'.format(arguments))
    if len(arguments)>0:
        nParts=(len(arguments[0]),)
    else:
        nParts=(1,)
        arguments=[[None]]
    #log.info('nParts={}'.format(nParts))
#    log.info('newShape={}'.format(newShape))
    #log.info('repeatedArguments={}'.format(repeatedArguments))
    #log.info('call with multiple args={}'.format(callWithMultipleArguments))
    if callWithMultipleArguments:        
        def mpFunction(queue,processID):
            kwargs["processID"] = processID
            result=function(*arguments,*args,**kwargs)
            #            log.info('result type={}'.format(type(result)))
            if isinstance(result,(list,tuple)):
                result=np.asarray(result)

            result_dict={processID:result}
            #           log.debug('processID type={}'.format(type(processID)))
            queue.put(result_dict,False)
    else:
        def mpFunction(queue,processID):
            kwargs["processID"] = processID
            #log.info(next(zip(*arguments)))
            result=list(function(*argument,*args,**kwargs) for argument in zip(*arguments))
            if len(result)>0:
                all_results_are_numpy_arrays = np.array([isinstance(r,np.ndarray) for r in result]).all()
                if all_results_are_numpy_arrays:
                    all_results_have_same_shape = np.array([r.shape==result[0].shape for r in result]).all()
                    if all_results_have_same_shape:
                        result = np.concatenate(result)
            #results={processID:result_array.reshape(nParts+result_array.shape[1:])}
            results={processID:result}
            #        log.info('results=\n{}'.format(results))
            #        log.info('putting result in queue')
            queue.put(results,False)
            #        log.info('{} elements are in queue'.format(queue.qsize())
    return mpFunction

def get_mp_function_SharedArray(function,arguments,output_shape,args=[],callWithMultipleArguments=True,output_file = False,output_indices=slice(None),**kwargs):
    repeatedArguments=[repeat(arg) for arg in args]
    #log.info('arguments={}'.format(arguments))
    if len(arguments)>0:
        nParts=(len(arguments[0]),)
    else:
        nParts=(1,)
        arguments=['mock_argument']
    #log.info('nParts={}'.format(nParts))
#    log.info('newShape={}'.format(newShape))
    #log.info('repeatedArguments={}'.format(repeatedArguments))
    #log.info('call with multiple args={}'.format(callWithMultipleArguments))
    if callWithMultipleArguments:        
        def mpFunction(processID):
            output_file = sa.attach('shm://shared_output').reshape(output_shape)            
            output_file[output_indices]=function(*arguments,*args,**kwargs)
            del(output_file)
    else:
        def mpFunction(processID):
            output_file = sa.attach('shm://shared_output').reshape(output_shape)
            output_file[output_indices]=np.array(list(function(argument,*args,**kwargs) for argument in arguments))
            del(output_file)

                #        log.info('{} elements are in queue'.format(queue.qsize())
    return mpFunction

def get_mp_function_SharedArray_multi(function,arguments,output_names,output_shape,args=[],callWithMultipleArguments=True,output_file = False,output_indices=slice(None),**kwargs):
    repeatedArguments=[repeat(arg) for arg in args]
    #log.info('arguments={}'.format(arguments))
    if len(arguments)>0:
        nParts=(len(arguments[0]),)
    else:
        nParts=(1,)
        arguments=['mock_argument']
    #log.info('nParts={}'.format(nParts))
    #    log.info('newShape={}'.format(newShape))
    #log.info('repeatedArguments={}'.format(repeatedArguments))
    #log.info('call with multiple args={}'.format(callWithMultipleArguments))
    
    if callWithMultipleArguments:        
        def mpFunction(processID):            
            output_files = [sa.attach('shm://'+ name).reshape(output_shape) for name in output_names]
            outputs = function(*arguments,*args,**kwargs)
            #log.info('len outputs = {}'.format(len(outputs)))
            #log.info('len outputs = {}'.format(len(output_files)))
            #log.info('len outputs = {}'.format(len(output_indices)))
            out_ids = output_indices
            for o_file,out in zip(output_files,outputs):
                #log.info(' shared array file dtype = {}'.format(o_file.dtype))
                #log.info(' out dtype = {}'.format(out.dtype))
                #log.info(' out shape = {}'.format(out.shape))
                ##if out.dtype == np.dtype(bool):
                #    log.info('shared array file dtype = {}'.format(o_file.dtype))
                #    log.info('{} % = {}'.format(np.sum(out)/np.prod(out.shape)))
                o_file[out_ids]=out
                #log.info('dtypes = {}'.format([o_file.dtype,out.dtype]))
                del(o_file)
    else:
        def mpFunction(processID):
            output_files = [sa.attach('shm://'+ name).reshape(output_shape) for name in output_names]
            outputs = list(zip(*list(function(argument,*args,**kwargs) for argument in arguments)))
            for o_file,out,out_ids in zip(output_files,outputs,output_indices):
                o_file[out_ids]=np.asarray(out)
                del(o_file)
                #        log.info('{} elements are in queue'.format(queue.qsize())
    return mpFunction


def get_mp_function_SharedArray_multi_new(function,arguments,output_names,output_shapes,args=[],callWithMultipleArguments=True,output_file = False,output_indices=slice(None),**kwargs):
    repeatedArguments=[repeat(arg) for arg in args]
    #log.info('arguments={}'.format(arguments))
    if len(arguments)>0:
        nParts=(len(arguments[0]),)
    else:
        nParts=(1,)
        arguments=['mock_argument']
    #log.info('nParts={}'.format(nParts))
    #    log.info('newShape={}'.format(newShape))
    #log.info('repeatedArguments={}'.format(repeatedArguments))
    #log.info('call with multiple args={}'.format(callWithMultipleArguments))
    kwargs['output_ids']=output_indices
    if callWithMultipleArguments:        
        def mpFunction():            
            output_files = [sa.attach('shm://'+ name).reshape(shape) for name,shape in zip(output_names,output_shapes)]
            kwargs['outputs']=output_files
            # in this function output have to be placed in outputs[i][output_ids]
            function(*arguments,*args,**kwargs)            
    else:
        def mpFunction():
            output_files = [sa.attach('shm://'+ name).reshape(shape) for name,shape in zip(output_names,output_shapes)]
            kwargs['outputs']=output_files
            for argument,out_ids in zip(arguments,output_indices):
                kwargs['output_ids']=out_ids
                # in this function output have to be placed in outputs[i][output_ids]
                function(argument,*args,**kwargs)
    return mpFunction

###############
#####recipies#####
def id_operator(x):
    return x
class RecipeFactory:
    def __init__(self,operatorDict):
        # operatorDict is a dictionary containing Function names as keys and the function themselfs as values
        self.operatorDict={}
        self.addOperators({'id':id_operator})
        self.addOperators(operatorDict)
        self.number_of_arguments_per_operator={}
    def copy(self):
        return RecipeFactory(self.operatorDict.copy())
    
    def addOperators(self,operatorDict):
        for key,value in operatorDict.items():
            if isinstance(value,list):
                assert isinstance(value[1],int) and (len(value)==2),'opperator list has be of type [function,int:n_umber_of_arguments] but is {}'.format([type(x)for x in value])
                self.operatorDict[key]=value[0]
                self.number_of_arguments_per_operator[key]=value[1]
            else:
                self.operatorDict[key]=value

    def get_operator(self,name):
        return self.operatorDict[name]
    def buildRecipe(self,recipeSketch):
        keys={'preProcessing','processData','processResults','postProcessing'}
        recipePlan=self.parseRecipeSketch(recipeSketch,keys)
#        log.info('Parsed the Sketch to: \n {0}\n'.format(recipePlan['preProcessing']))
        recipeData={}
        for key in keys:
            processPlan=recipePlan[key]
            recipeData[key]=self.buildProcess(processPlan)

        return recipeData
    def buildProcessFromSketch(self,processSketch):
        processPlan=self.parseProcessSketch(processSketch)
        process=self.buildProcess(processPlan)
        return process
    def buildProcess(self,processPlan):
        stepList=[]
        outputInputMappingsList=[]
        previousStepPlan=False
        for outputInputMapping,stepPlan in processPlan:
            try:
                step=self.buildStep(stepPlan)
            except Exception as e:
                log.error('error in step\n {}'.format(stepPlan))
                raise e
            if outputInputMapping.shape==(0,):
                isFirstStep=isinstance(previousStepPlan,bool)
                if isFirstStep:
                    previousStepPlan=np.arange(step.numberOfInputs)
                outputInputMapping=self.getStandardMapping(previousStepPlan,step.numberOfFreeInputs)
            stepList+=[step]
            outputInputMappingsList+=[outputInputMapping]
            previousStepPlan=stepPlan
        outputInputMappingsList=tuple(map(tuple,outputInputMappingsList))
        process=Process(stepList,outputInputMappingsList)
        return process

    def buildStep(self,stepPlan):
        #requires Python 3.7 and higher ! otherwise the order of keys of a dict may not be preserved.
        stepOperatorDict={}
        operatorList=[]
        numberOfInputsPerOperatorList=[]
        
        fixedInputsStep=np.array([],dtype=object)
        fixedInputsIndexStep=np.array([],dtype=int)
        NrInputsStep=0
        for operatorSpecifier in stepPlan:
            operatorName=operatorSpecifier[0]
    #        log.info('operatorName={}'.format(operatorName))
            try:
                operator=self.operatorDict[operatorName]
            except KeyError as error:
                log.error('Operator "{}" not found in operator dictionary keys.\n Operator dictionary conthaines the following keys \n {}'.format(operatorName,self.operatorDict.keys()))
                raise
            #            log.info('operatorSpecifier[1] = {0}'.format(operatorSpecifier[1]))
            try:
                fixedInputs=operatorSpecifier[1][1]
            except IndexError as e:
                log.error(operatorSpecifier)
                raise e
            fixedInputsIndex=operatorSpecifier[1][0]+NrInputsStep
            
            fixedInputsIndexStep=np.concatenate((fixedInputsIndexStep,fixedInputsIndex))
            fixedInputsStep=np.concatenate((fixedInputsStep,fixedInputs))            

            if isinstance(operator,Process):
                NrInputs = len(operator.outputInputMappings[0])
                #log.info('Nr Inputs in initalization = {}'.format(NrInputs))
                operatorFunction = operator.run
            else:
                NrInputs=self.number_of_arguments_per_operator.get(operatorName,'no number of arguments given.')
                operatorFunction = operator                
                if not isinstance(NrInputs,int):
                    NrInputs=len(inspect.signature(operatorFunction).parameters)
            NrFreeInputs=NrInputs-operatorSpecifier[1].shape[-1]
            NrInputsStep+=NrInputs
            operatorPositionsInList=stepOperatorDict.get(operatorName,[])            
            stepOperatorDict[operatorName]=operatorPositionsInList+[len(operatorList)]            
            operatorList+= [operatorFunction]
            numberOfInputsPerOperatorList += [NrInputs]


        freeInputIndexStep=np.delete(np.arange(NrInputsStep),fixedInputsIndexStep.astype(int))
        inversInputIndexStep=np.concatenate((fixedInputsIndexStep,freeInputIndexStep)).astype(int)
        inputIndexStep=invertNumpyIndex(inversInputIndexStep) # calls class external function !

        numberOfInputsPerOperatorList=tuple(numberOfInputsPerOperatorList)
        fixedInputsStep=tuple(fixedInputsStep)
        inputIndexStep=tuple(inputIndexStep)                             
        inputData=[numberOfInputsPerOperatorList,fixedInputsStep,inputIndexStep]
#        log.info('inputData={}'.format(inputData))
        step=Step(stepOperatorDict,operatorList,inputData)
        return step

    
    def getStandardMapping(self,previousStepPlan,NrFreeInputsStep):
        numberOfOutputs=len(previousStepPlan)
        
        if numberOfOutputs==1:
            inOut=self.oneToAll(NrFreeInputsStep)
        else:
            if numberOfOutputs==NrFreeInputsStep:
                inOut=self.oneToOne(NrFreeInputsStep)
            elif numberOfOutputs<NrFreeInputsStep:
#                log.info('OutputNumber(nO) < InputNumber(nI) after step: \n{}\n No mapping data was specified! I will map the first output to the first nI-nO+1 inputs.All other outputs are mapped 1 to 1.'.format(previousStepPlan))
                inOut=self.manyToMany_standard(numberOfOutputs,NrFreeInputsStep)
            else:
                try:
                    raise AssertionError('Invalid process scetch!\n There are more outputs than inputs for the next step after step: \n{}'.format(previousStepPlan))
                except AssertionError as error:
                    log.error(error)
                    raise

        return inOut
    
    def oneToAll(self,NrInputs):
        inOut=np.zeros(NrInputs,dtype=int)
        return inOut
    
    def oneToOne(self,NrInputs):
        inOut=np.arange(NrInputs,dtype=int)
        return inOut

    def manyToMany_standard(self,nrOutputs,nrInputs):
        difference=nrInputs-nrOutputs
        firstToDifference=np.zeros(difference+1,dtype=int)
        restToRest=np.arange(1,nrOutputs,1,dtype=int)
        inOut=np.concatenate((firstToDifference,restToRest))
        return inOut

    def parseRecipeSketch(self,recipeSketch,keys):
        recipePlan={}
        for key in keys:
            processSketch=recipeSketch.get(key,[])
            recipePlan[key]=self.parseProcessSketch(processSketch)
        return recipePlan        

    
    def parseProcessSketch(self,processSketch):
        try:
            processPlan=[]
            placeholderIOMapping=np.array([],dtype=int)
            if isinstance(processSketch,str):
                stepSketch=processSketch
                stepPlan=self.parseStepSketch(stepSketch)
                processPlan=[[placeholderIOMapping,stepPlan]]
            elif isinstance(processSketch,list):
                for element in processSketch:
                    if isinstance(element,str):                    
                        stepPlan=self.parseStepSketch(element)
                        processPlan+=[[placeholderIOMapping,stepPlan]]
                    elif isinstance(element,list):
                        if len(element)==0:
                            log.info('process Sketch: Ignore empty step.')
                        else:
                            first_part = element[0]
                            if isinstance(first_part,(np.ndarray,tuple)):
                                try:
                                    assert len(element)==2
                                except AssertionError:
                                    log.error('Invalid process sketch. Element = {}'.format(element))
                                    raise
                                stepPlan=self.parseStepSketch(element[1])
                                processPlan+=[[np.asarray(first_part).astype(int),stepPlan]]
                            elif isinstance(first_part,str):
                                stepPlan=self.parseStepSketch(element)
                                processPlan+=[[placeholderIOMapping,stepPlan]]
                            else:
                                raise AssertionError('Invalid process sketch. Element = {}'.format(element))
                    else:
                        raise AssertionError('Invalid process sketch: sketch element must be either a string or a list')
            else:
                raise AssertionError('Invalid process sketch: sketch must be either a string or a list')
        except AssertionError as error:
            log.error(error)
            raise
  #      log.info('process Plan={}'.format(processPlan))
        return processPlan


    def parseStepSketch(self,stepSketch):
        try:
            stepPlan=[]
            placeholderFixedInputs=np.array([[],[]],dtype=object)
            if isinstance(stepSketch,str):            
                stepPlan=[[stepSketch,placeholderFixedInputs]]
            elif isinstance(stepSketch,tuple):
                if (len(stepSketch)==2):
                    if isinstance(stepSketch[1],(np.ndarray,tuple)):
                        stepPlan=[stepSketch[0],np.asarray(stepSketch[1],dtype = object)]
            elif isinstance(stepSketch,list):
                for element in stepSketch:
                    if isinstance(element,str):
                        stepPlan+=[[element,placeholderFixedInputs]]
                    elif isinstance(element,(list,tuple)):
                        errorMessage='Invalid step sketch format: given Format = {} expected Format = [str,ndarray] '.format(element)
                        assert (len(element)==2),errorMessage
                        assert isinstance(element[0],str),errorMessage
                        assert isinstance(element[1],(tuple,np.ndarray)),errorMessage
                        fixedInputsSketch=np.asarray(element[1],dtype = object)
                        sketchShape=fixedInputsSketch.shape
                        if (len(sketchShape)==1):
                            mapping=np.arange(sketchShape[0])
                            fixedInputsPlan=np.array([mapping,fixedInputsSketch],dtype=object)
                            element=[element[0],fixedInputsPlan]
                        else:
                            element = [element[0],fixedInputsSketch]                            
                        stepPlan += [element]
                    else:
                        raise AssertionError('Invalid step sketch format: Each entry of a sketch must be either a String or a List')
            else:
                raise AssertionError('Invalid step sketch format: sketch must be either a String or a List')
        except AssertionError as error:
            log.error(error)
            raise
        return stepPlan
    
class Process:
    def __init__(self,steps,outputInputMapping):
        def readOperatorsFromSteps(steps):
            operatorDict={}
            for stepIndex,step in enumerate(steps):
                for operatorName in step.operatorDict.keys():
                    stepList=operatorDict.get(operatorName,[])
                    operatorDict[operatorName]=stepList+[stepIndex]
            return operatorDict
            
        self.steps=steps
        self.outputInputMappings=outputInputMapping
        self.processList=tuple(zip(steps,outputInputMapping))
        self.nextInput=[]
        self.operatorInStepDict=readOperatorsFromSteps(steps)

    def changeOperator(self,operatorName,operator):
        stepIndexList=self.operatorInStepDict.get(operatorName,[])
        for index in stepIndexList:
            self.steps[index].changeOperator(operatorName,operator)
            
    def run(self,*args):
#        if not isinstance(initial_arg,tuple):
        nextInput=tuple(args)
            
#        log.info('nextInput={}'.format(nextInput))        
        processList=self.processList
        for step,mapping in processList:
            try:
                #log.info('input length={}'.format(len(nextInput)))
                #log.info('input output mapping={}'.format(mapping))
                #log.info('step ={}'.format(tuple(step.operatorDict.keys())))
                _input=tuple(map(nextInput.__getitem__,mapping))
                #log.info('alive')
                #log.info('_input length={}'.format(len(_input)))
                nextInput=step.run(_input)
                #log.info('alive2')
                #            log.info('next Input={}'.format(nextInput))
            except IndexError as e:
                log.error('Error in Process containing {}'.format(self.operatorInStepDict))
                log.error(e)
                raise
        if len(nextInput)==1:
            return nextInput[0]
        else:
            return nextInput
                    
class Step:
    def __init__(self,operatorDict,operatorList,inputData):
        self.operatorDict=operatorDict
        self.operatorList=operatorList
        self.numberOfInputsPerOperatorList=inputData[0]
        self.predefinedInputs=inputData[1]
        self.inputMapping=inputData[2]
  #      log.info('operatorDict keys={}'.format(operatorDict.keys()))
        self.numberOfInputs=sum(self.numberOfInputsPerOperatorList)
        self.numberOfFreeInputs=self.numberOfInputs-len(self.predefinedInputs)
        self.stepList=tuple(zip(self.operatorList,self.numberOfInputsPerOperatorList))
#        log.info('step list=\n{}'.format(self.stepList))

    def changeOperator(self,operatorName,operator):
        stepListPositions=self.operatorDict.get(operatorName,[])
        for position in stepListPositions:
            self.operatorList[position]=operator
        self.stepList=tuple(zip(self.operatorList,self.numberOfInputsPerOperatorList))
        
    def run(self,_input):
        try:
            output=()
            _input=self.predefinedInputs+_input
            #log.info('input={}'.format(_input))        
            #log.info('length of _input={}'.format(len(_input)))
            #log.info('input Mapping={}'.format(self.inputMapping))
            getitem=_input.__getitem__
            #        log.info('second Item={}'.format(getitem(self.inputMapping[-1])))
            _input=tuple(map(getitem,self.inputMapping))
        except Exception as e:
            log.error('Input output mismatch in step containing the operators: {}.\n Input = {} InputMapping = {}'.format(self.operatorDict.keys(),_input,self.inputMapping))
            raise e
        log.debug('running step with Operators ={}'.format(self.operatorDict.keys()))
        for operator,NrOfInputs in self.stepList:
            try:
                #log.info('operator={}'.format(operator))
                #log.info('nrOfInputs = {}'.format(NrOfInputs))
                #log.info('nrOf  available Inputs = {}'.format(len(_input)))                
                out=operator(*_input[:NrOfInputs])
                #            log.info('output={}'.format(out))
                if type(out)!=type(None):
                    output+=(out,)
                _input=_input[NrOfInputs:]
            except Exception as e:
                log.error('Error in operator {} in the step containing {}'.format(operator,self.operatorDict.keys()))
                raise e
        return output
    

########################
###### generall classes

class DictNamespace(SimpleNamespace):
    @staticmethod
    def dict_to_dictnamespace(d):
        n=DictNamespace()
        for key,value in d.items():
            if isinstance(value,dict):
                value = DictNamespace.dict_to_dictnamespace(value)
            setattr(n,str(key),value)
        return n
    @staticmethod
    def dictnamespace_to_dict(d):
        n={}
        for key,value in d.items():
            if isinstance(value,DictNamespace):
                value = DictNamespace.dictnamespace_to_dict(value)
            n[str(key)]=value
        return n
    def __init__(self,**kwargs):
        super().__init__(**kwargs)        
    def decorator_convert_dict_to_dictnamespace(func):
        def new_func(self,*args):
            if isinstance(args[-1],dict):
                args = args[:-1] + (self.dict_to_dictnamespace(args[-1]),)
            return func(self,*args)
        return new_func
    def dict(self):
        return self.dictnamespace_to_dict(self)
    def items(self):
        for key,value in self.__dict__.items():
            yield (key,value)
    def keys(self):
        for key in self.__dict__.keys():
            yield key
    def pop(self,key):
        return self.__dict__.pop(key)
    def values(self):
        for values in self.__dict__.values():
            yield values
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return self.keys()
    def __getitem__(self, item):
        return self.__dict__[item]
    def copy(self):
        return self.dict_to_dictnamespace(self.dict())
    def get(self,*args,**kwargs):
        return self.__dict__.get(*args,**kwargs)
    
    @decorator_convert_dict_to_dictnamespace
    def __setitem__(self, key,value):
        self.__setattr__(key, value)
        
    @decorator_convert_dict_to_dictnamespace
    def update(self,data):
        self.__dict__.update(data)
            
    
    def  __getattribute__(self,key):
        try:
            value = super().__getattribute__(key)
        except AttributeError as e:
            log.error(e)
            log.info('Known attributes are {}'.format(list(self.keys())))
            raise
        return value
    
class DictNamespace2(SimpleNamespace):
    @staticmethod
    def dict_to_dictnamespace(d):
        n=DictNamespace()
        for key,value in d.items():
            if isinstance(value,dict):
                value = DictNamespace.dict_to_dictnamespace(value)
            setattr(n,str(key),value)
        return n
    @staticmethod
    def dictnamespace_to_dict(d):
        n={}
        for key,value in d.items():
            if isinstance(value,DictNamespace):
                value = DictNamespace.dictnamespace_to_dict(value)
            n[str(key)]=value
        return n
    def __init__(self,**kwargs):
        super().__init__(**kwargs)        
    def decorator_convert_dict_to_dictnamespace(func):
        def new_func(self,*args):
            if isinstance(args[-1],dict):
                args = args[:-1] + (self.dict_to_dictnamespace(args[-1]),)
            return func(self,*args)
        return new_func
    def dict(self):
        return self.dictnamespace_to_dict(self)
    def items(self):
        for key,value in self.__dict__.items():
            yield (key,value)
    def keys(self):
        for key in self.__dict__.keys():
            yield key
    def pop(self,key):
        return self.__dict__.pop(key)
    def values(self):
        for values in self.__dict__.values():
            yield values
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return self.keys()
    def __getitem__(self, item):
        return self.__dict__[item]
    def copy(self):
        return self.dict_to_dictnamespace(self.dict())
    def get(self,*args,**kwargs):
        return self.__dict__.get(*args,**kwargs)
    
    @decorator_convert_dict_to_dictnamespace
    def __setitem__(self, key,value):
        self.__setattr__(key, value)
        
    @decorator_convert_dict_to_dictnamespace
    def update(self,data):
        self.__dict__.update(data)
            
    
    def  __getattribute__(self,key):
        try:
            value = super().__getattribute__(key)
        except AttributeError as e:
            log.error(e)
            log.info('Known attributes are {}'.format(list(self.keys())))
            raise
        return value
    
class FTGridPair:
    def __init__(self,realGrid,reciprocalGrid):
        self.realGrid=realGrid
        self.reciprocalGrid=reciprocalGrid




def option_switch(opt,default_value=True):
    '''returns True when opt is True or non-bool, if non_bool retun opt else return default value as second parameter'''
    if isinstance(opt,bool):
        if opt:
            return (True,default_value)
        else:
            return (False,default_value)
    else:
        return (True,opt)



def grow_mask(mask,n_pixels):
    mansk = np.array(mask)
    for n in range(n_pixels):        
        grads = np.sum(np.abs(np.gradient(mask.astype(int),edge_order = 1)[1:]),axis = 0)
        mask = mask | (grads!=0)
    return mask


def make_string_tex_conform(string):
    string = string.replace('_','\_')
    string = string.replace('&','\&')
    string = string.replace('{','\{')
    string = string.replace('}','\}')
    string = string.replace('%','\%')
    #log.info('corrected string = {}'.format(string))
    return string


def split_ids_by_unique_values(ids):
    '''
    Assumes input is sorted list of integers.
    For [0,0,0,1,1,2,3,3,3] this funciton returns [3,5,6]
    '''
    split_ids = np.nonzero(np.diff(ids))[0]+1
    return split_ids

def convert_to_slice_if_possible(frame_ids):
    #log.info('to slice = {}'.format(frame_ids))
    frame_ids = np.asarray(frame_ids)
    if len(frame_ids) ==0:
        #empty slice
        return slice(1,0)
    elif len(frame_ids) == 1:
        start = frame_ids[0]
        return slice(start,start+1)
    else:
        start = frame_ids[0]
        step = frame_ids[1] - start
        stop = frame_ids[-1] + step
        #log.info('start = {}, stop = {} step = {}'.format(start,stop,step))
        #log.info('frame_ids = {}'.format(frame_ids))
        sliced_ids = np.arange(start,stop,step,dtype = frame_ids.dtype)
        if len(frame_ids) == len(sliced_ids):
            if (frame_ids == sliced_ids).all():            
                #log.info('frame selection is slice like!')
                frame_ids = slice(start,stop,step)                    
    return frame_ids

def split_into_simple_slices(sequence,return_length=False,return_sliced_args=False,mod=False):
    '''
    splits a 1D non decreasing list of integers into slices which correspond to
    1. if mod = False
    its connected components.  Where two integers i,j are connected if |j-i|=1, eg if they are consequtive.
    2. if mod = int:  The same but in equivalence classes of Z/modZ
    '''
    slices = []
    arg_slices = []
    lengths = []
    if isinstance(mod,int) and (not isinstance(mod,bool)):
        eq_class_ids = sequence%mod
        for c_id in range(mod):
            eq_class = sequence[eq_class_ids==c_id]
            if len(eq_class)<=0:
                continue
            eq_min = eq_class[0]
            temp = (eq_class-eq_min)//mod
            jumps = np.nonzero(np.diff(temp)!=1)[0]+1
            connected_components = np.split(eq_class,jumps)
            #connecteda_args = np.split(np.arange(len(sequence)),jumps)
            for c in connected_components:
                if len(c) >0:
                    slices.append(slice(c[0],c[-1]+mod/2,mod))
                    arg_slices.append(slice(c[0]-eq_min,c[-1]+mod/2-eq_min,mod))
                    lengths.append(len(c))
    else:
        jumps = np.nonzero(np.diff(sequence)!=1)[0]+1
        connected_components = np.split(sequence,jumps)
        connected_args = np.split(np.arange(len(sequence)),jumps)
        _min=sequence.min()                                
        for c,ca in zip(connected_components,connected_args):
            if len(c) >0:
                slices.append(slice(c[0],c[-1]+1))
                arg_slices.append(slice(ca[0],ca[-1]+1))
                lengths.append(len(c))
            
    if return_length and return_sliced_args:
        return slices,arg_slices,lengths
    elif return_length:
        return slices,lengths
    elif return_sliced_args:
        return slices,arg_slices
    else:
        return slices    
    
###### cache aware routines ######
def get_L2_cache_split_parameters(data_shape,data_type,L2_cache):
    unit_size = data_type.itemsize
    units_in_L2 =  L2_cache*1024/unit_size
    submatrix_sizes = np.array([np.prod(np.array(data_shape[i:])) for i in range(len(data_shape))])
    #log.info('submatrix_sizes = {}'.format(submatrix_sizes))
    #log.info("units in  L2 + {}".format(units_in_L2))
    if submatrix_sizes[0]>units_in_L2:
        splitting_dimension = np.nonzero(submatrix_sizes//units_in_L2)[0][-1]
    else:
        splitting_dimension = -1

    #log.info('unit size {} units in L2 {} data shape {} submatrix_sizes {} splitting_dimension = {}'.format(unit_size,units_in_L2,data_shape,submatrix_sizes,splitting_dimension))
    
    if splitting_dimension >= len(data_shape)-1:
        fixed_sub_size = 1*unit_size
    else:
        fixed_sub_size = submatrix_sizes[splitting_dimension+1]
        
    n_sub_sizes_in_L2 = int(units_in_L2//fixed_sub_size)    
    step = n_sub_sizes_in_L2
    #log.info('step = {}'.format(step))
    return (splitting_dimension,step)

def _generate_conjugate_default(out_array):
    conj = np.conj
    def abs_value(data):
        return conj(data,out=out_array)
    return out_array

def _generate_conjugate_cache_aware(out_array,L2_cache):
    data_shape = out_array.shape
    data_type = out_array.dtype
    splitting_dimension,step = get_L2_cache_split_parameters(data_shape,data_type,L2_cache)

    mult = np.multiply
    sqrt = np.sqrt
    conj = np.conjugate
    def conj_1_loop(data):
        for i in range(0,data_shape[0],step):
            i2 = i+step
            d = data[i:i2]
            o = out_array[i:i2]
            conj(d,out = o)
        return out_array
    def conj_2_loop(data):
        for i in range(data_shape[0]):
            for j in range(0,data_shape[1],step):
                j2 = j+step
                d = data[i,j:j2]
                o = out_array[i,j:j2]
                conj(d,out = o)
        return out_array
    def conj_3_loop(data):
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for k in range(0,data_shape[2],step):
                    k2 = k+step
                    d = data[i,j,k:k2]
                    o = out_array[k:k2]
                    conj(d,out = o)
        return out_array
    
    if splitting_dimension == -1:
        abs_value = _generate_conjugate_default(out_array)
    elif splitting_dimension == 0:
        abs_value = conj_1_loop
    elif splitting_dimension == 1:
        abs_value = conj_2_loop
    elif splitting_dimension == 2:
        abs_value = conj_3_loop
    return abs_value

def generate_conjugate(data_shape,data_type,cache_aware=False,L2_cache=256):
    out = np.zeros(data_shape,dtype = data_type)
    if not cache_aware:
        abs_value = _generate_conjugate_default(out)
    else:
        abs_value = _generate_conjugate_cache_aware(out,L2_cache)
    return abs_value


#### Testing routines ####
def default_test(result,expected,precision=False, name='default'):
    #if isinstance(result,np.ndarray) and isinstance(expected,np.ndarray):
    #    result = [result]
    #    expected = [expected]
    def listsAreEqual(result,expected):
        areEqual=True
        if len(result)==len(expected):
            for resPart,expPart in zip(result,expected):
                if isinstance(resPart,(list,tuple)) and isinstance(expPart,(list,tuple)):
                    areEqual = areEqual and listsAreEqual(resPart,expPart)
                elif isinstance(resPart,np.ndarray) and isinstance(expPart,np.ndarray):
                    if resPart.dtype==np.object and expPart.dtype==object and len(resPart)>1:
                        areEqual = areEqual and listsAreEqual(resPart,expPart)
                    else:
                        if precision==False:
                            areEqual = areEqual and (resPart==expPart).all()
                        else:
                            areEquaL=areEqual and np.allclose(resPart,expPart,atol=precision)
                else:
                    try:
                        areEqual = areEqual and (resPart == expPart)
                    except Exception as e:
                        areEqual = False
                if not areEqual:
                    #log.info('{},{}'.format(result,expected))
                    break
        else:
            areEqual=False
        return areEqual

    test_successful = False
    if not (isIterable(result) or isIterable(expected)):
        try:
            test_successful = (result == expected)
        except Exception as e:
            pass
    else:
        test_successful = listsAreEqual(result,expected)
    try:
        assert test_successful
        print('Test {}: Passed!'.format(name))
    except AssertionError:
        print('Test {} Failed: \n Expected:{} \n Got:{}'.format(name,expected,result))
        raise


#### process threshold ####
def create_threshold_projection(threshold):
    no_lower_threshold  = (not isinstance(threshold[0],(float,int))) or isinstance(threshold[0],bool)
    no_upper_threshold = (not isinstance(threshold[1],(float,int))) or isinstance(threshold[1],bool)
    #log.info('no upper thresh = {} now lower thresh = {}'.format(no_upper_threshold,no_lower_threshold))
    if no_upper_threshold and (not no_lower_threshold):
        def threshold_projection(density):
            real_part = density.real            
            is_invalid_mask = real_part<threshold[0]
            #log.info('n = {} pixels lower than {}'.format(is_invalid_mask.sum(),threshold[0]))
            real_part[is_invalid_mask] = threshold[0]           
            return [density,is_invalid_mask]
    elif (not no_upper_threshold) and no_lower_threshold:
        def threshold_projection(density):
            real_part = density.real
            is_invalid_mask = real_part>threshold[1]
            real_part[is_invalid_mask] = threshold[1]            
            return [density,is_invalid_mask]
    elif no_upper_threshold and no_lower_threshold:
        def threshold_projection(density):
            return [density,False]
    else:
        def threshold_projection(density):
            real_part = density.real
            is_smaller_mask = (real_part<threshold[0])
            is_bigger_mask = (real_part>threshold[1])
            real_part[is_smaller_mask] = threshold[0]
            real_part[is_bigger_mask] = threshold[1]            
            return [density,is_bigger_mask | is_smaller_mask] 
    return threshold_projection


#### hide print ####
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
def stdout_redirect(to):
    fd = sys.stdout.fileno()
    sys.stdout.close() # + implicit flush()
    os.dup2(to.fileno(), fd) # fd writes to 'to' file
    sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        
