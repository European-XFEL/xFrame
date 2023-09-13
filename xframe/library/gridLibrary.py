import numpy as np
import numpy.ma as mp
from scipy.interpolate import griddata
import logging

from xframe.library.mathLibrary import cartesian_to_spherical, spherical_to_cartesian, get_gaussian_weights_1d

log=logging.getLogger('root')

class Grid:
    types={'uniform':0,'uniform_dependend':1,'nonUniform_dependend':2,'custom':3}

    ###Class function###
    def copy(grid):
        if grid.gridType!=Grid.types['nonUniform_dependend']:
            copiedArray=np.copy(grid.array)
        else:
            copiedArray=mp.copy(grid.array)
        copiedGrid=Grid(copiedArray,gridType=grid.gridType)
        return copiedGrid

    
    def __init__(self,gridArray,gridType='custom'):    
        '''
        Assumes gridArray is a normal numpy array if dimension =1 or an array of dtype object which contains numpy arrays of type other than object which correspond to grid points.
        '''
         
        try:
            if type(gridType)==str:
                self.gridType=self.types[gridType]
            else:
                assert gridType in self.types.values()
                self.gridType=gridType
        except (KeyError,AssertionError):
            log.error('Given gridType ={} is not known!'.format(gridType))
            raise
        
        firstElement=gridArray.flatten()[0]
        if isIterable(firstElement):
            self.dimension=len(firstElement)
        else:
            self.dimension=0

        if self.gridType==self.types['nonUniform_dependend']:
            assert isinstance(gridArray,np.ma.core.MaskedArray),'Grid Type nonUniform expects MaskedArray gridArrays but type given was {}'.format(type(gridArray))
        self.array=gridArray
        self.shape=gridArray.shape
        
        self.getitem=self.generateGetItemFunction(self.gridType)
        self.setitem=self.generateSetItemFunction(self.gridType)
        self.applyFunctionToElements=self.generateApplyFunctionToElements(self.gridType)
        self.applyFunctionAlongAxis=self.generateApplyFunctionToAxis(self.gridType)

    def setDimension(self,newDimension):
        self.dimension=newDimension

        self.getitem=self.generateGetItemFunction(self.gridType)
        self.setitem=self.generateSetItemFunction(self.gridType)
        
    def setDimensionAndGridType(self,newGridType,newDimension):
        try:
            if type(gridType)==str:
                self.gridType=self.types[newGridType]
            else:
                assert newGridType in types.values()
                self.gridType=newGridType
        except (KeyError,AssertionError):
            log.error('Given gridType ={} is not known!'.format(newGridType))
            raise
        self.dimension=newDimension

        self.getitem=self.generateGetItemFunction(self.gridType)
        self.setitem=self.generateSetItemFunction(self.gridType)
        self.applyFunctionToElements=self.generateApplyFunctionToElements(self.gridType)
        self.applyFunctionAlongAxis=self.generateApplyFunctionToAxis(self.gridType)

    def setArray(self,newArray):
        self.array=newArray
#        log.info('new array.shape={}'.format(newArray))
        self.shape=newArray.shape
        
    def __getitem__(self,key):
        item=self.getitem(self,key)
        return item
    
    def __setitem__(self,key,value):
        self.setitem(self,key,value)

    def __mul__(self,multiplicant):
        if self.gridType==self.types['nonUniform_dependend']:
            self.array[~self.array.mask]*=multiplicant
        else:
            self.array*=multiplicant
        return self
            
    def apply_along_axis(self,function,axis,*args,**kwargs):
#        log.info('apply along axis ={}'.format(self.applyFunctionAlongAxis))
        sampledGrid=self.applyFunctionAlongAxis(self,function,axis,*args,**kwargs)
        return sampledGrid
    
    def apply(self,function,*args,**kwargs):
#        log.info('alive!')
        sampledGrid=self.applyFunctionToElements(self,function,*args,**kwargs)
        return sampledGrid

    def reshape(self,*args):
        self.array=self.array.reshape(*args)
        self.shape=self.array.shape
        return self

    def transpose(self,**kwargs):
        self.array=np.transpose(self.array,**kwargs)
        self.shape=self.array.shape
        return self
    
    def swapaxes(self,*args):
        self.array=self.array.swapaxes(*args)
        self.shape=self.array.shape
        return self
        
    def generateSetItemFunction(self,gridType):
        def setitem_nonUniform(self,key,value):
            array=self.array
            item=array.__getitem__(key)
            shape=item.shape
            mask=array.mask.__getitem__(key)
            keyIndexesSingleValue=isinstance(mask,np.bool_)

            if keyIndexesSingleValue:
                if not mask:
                    array.__setitem__(key,value)
                else:
                    log.info('trying to set a single masked value!')
            else:
                tempView=item.reshape(-1)
                validDataIndexes=tuple(np.arange(tempView.shape[0])[~tempView.mask])
                tempView[(validDataIndexes,)]=getArrayOfArray(value).flatten()
                array.__setitem__(key,item)

        def setitem_nonUniform1D(self,key,value):
            array=self.array
            item=array.__getitem__(key)
            shape=item.shape
            mask=array.mask.__getitem__(key)
            keyIndexesSingleValue=isinstance(mask,np.bool_)

            if keyIndexesSingleValue:
                if not mask:
                    array.__setitem__(key,value)
                else:
                    log.info('trying to set a single masked value!')
            else:
                tempView=item.reshape(-1)
                validDataIndexes=tuple(np.arange(tempView.shape[0])[~tempView.mask])
                tempView[(validDataIndexes,)]=value
                array.__setitem__(key,item)
                

        def setitem_uniform(self,key,value):
            array=self.array
            item=array.__getitem__(key)
#            log.info('item = {}'.format(item))
            keyIndexesSingleValue= (item.dtype!=np.object)
#            log.info('keyIndexesSingleValue={}'.format(keyIndexesSingleValue))
            if keyIndexesSingleValue:
                array.__setitem__(key,value)
            else:
                shape=item.shape
                tempView=item.flatten()
                tempView[:]=getArrayOfArray(value)
#                log.info('temp view={}'.format(tempView))
                array.__setitem__(key,tempView.reshape(shape))

        def setitem_uniform1D(self,key,value):
            array=self.array
            item=array.__getitem__(key)
            keyIndexesSingleValue= not isIterable(item)
            if keyIndexesSingleValue:                
                array.__setitem__(key,value)
            else:
                shape=item.shape
                tempView=item.reshape(-1)
                tempView[:]=value
                array.__setitem__(key,item)

                
        types=self.types
        dimension=self.dimension
        if gridType==types['nonUniform_dependend']:
            if dimension==0:
                setitem=setitem_nonUniform1D
            else:
                setitem=setitem_nonUniform
        else:
            if dimension==0:
                setitem=setitem_uniform1D
            else:
                setitem=setitem_uniform
        return setitem
            
    def generateGetItemFunction(self,gridType):
        def getitem_nonUniform(self,key):
            array=self.array
            keyIndexesSingleValue= isinstance(array.mask.__getitem__(key),np.bool_)
            if keyIndexesSingleValue:
                item=np.array(tuple(array.__getitem__(key)))
            else:
                item=array.__getitem__(key)
                item=np.array(tuple(item[~item.mask]))
            return item

        def getitem_nonUniform1D(self,key):
            array=self.array
            keyIndexesSingleValue= isinstance(array.mask.__getitem__(key),np.bool_)
            if keyIndexesSingleValue:
                item=array.__getitem__(key)
            else:
                item=array.__getitem__(key)
                item=np.array(item[~item.mask])
            return item
                
        def getitem_uniform(self,key):
            item=self.array.__getitem__(key)
            keyIndexesSingleValue= (item.dtype!=np.object)
#            log.info('key={}  singleValue={} item dtype={}'.format(key,keyIndexesSingleValue,item.dtype))
            if not keyIndexesSingleValue:
                item=item.flatten()
                item=np.array(tuple(item))
            return item
        
        def getitem_uniform1D(self,key):
            item=self.array.__getitem__(key)
            
            keyIndexesSingleValue= not isIterable(item)
  #          log.info('key={}  singleValue={}'.format(key,keyIndexesSingleValue))
            if not keyIndexesSingleValue:
                item=item.flatten()
            return item

        types=self.types
        dimension=self.dimension
        if gridType==types['nonUniform_dependend']:
            if dimension==0:
                getitem=getitem_nonUniform1D
            else:
                getitem=getitem_nonUniform
        else:
            if dimension==0:
                getitem=getitem_uniform1D
            else:
                getitem=getitem_uniform
        return getitem


    ###generate Apply Function along Axis###
    def generateApplyFunctionToAxis(self,gridType):   
        dimension=self.dimension
        if gridType==self.types['nonUniform_dependend']:
            if dimension==0:                
                decorator=concatenateDecorators([self.formatFunctionOutput,self.formatFunctionInput_applyMask])
                applyFunctionToAxis=self.plugDecoratorIntoTemplate(self.applyFunctionToAxis_nonUniform_template,decorator)
            else:
                decorator=concatenateDecorators([self.formatFunctionOutput,self.formatFunctionInput_nD,self.formatFunctionInput_applyMask])
                applyFunctionToAxis=self.plugDecoratorIntoTemplate(self.applyFunctionToAxis_nonUniform_template,decorator)
        else:
            if dimension==0:
                decorator=self.formatFunctionOutput
                applyFunctionToAxis=self.plugDecoratorIntoTemplate(self.applyFunctionToAxis_uniform_template,decorator)
            else:
                decorator=concatenateDecorators([self.formatFunctionOutput,self.formatFunctionInput_nD])
                applyFunctionToAxis=self.plugDecoratorIntoTemplate(self.applyFunctionToAxis_uniform_template,decorator)
        return applyFunctionToAxis

    def formatFunctionOutput(self,function):
        def newFunction(*args,**kwargs):
            output=function(*args,*kwargs)
            
            outputIsArray=isIterable(output)
            if outputIsArray:
                outputIsAnArrayOfArrys= len(output.shape)>1
                if outputIsAnArrayOfArrys:
                    output=getArrayOfArray(output)
            return output
        return newFunction

    def formatFunctionInput_nD(self,function):
        def newFunction(*args,**kwargs):
            #                log.info('input={}'.format(args[0]))
            args=(np.array(tuple(args[0])),)+args[1:]
            #                log.info('input2={}'.format(args[0]))
            output=function(*args,*kwargs)
            return output
        return newFunction

    def formatFunctionInput_applyMask(self,function):
        def newFunction(*args,**kwargs):
            args=(args[0][~args[0].mask],)+args[1:]
            output=function(*args,*kwargs)
            return output
        return newFunction
        
    def plugDecoratorIntoTemplate(self,template,decorator):            
        def applyFunctionAlongAxis(self,function,axis,*args,**kwargs):
#            log.info('function={}'.format(function))
#            log.info('template={}'.format(template))            
            output=template(decorator,function,axis,*args,**kwargs)
            return output
        return applyFunctionAlongAxis


    def applyFunctionToAxis_nonUniform_template(self,functionDecorator,function,axis,*args,**kwargs):
        function=appendArgumentsToFunction(function,*args,**kwargs)
        newFunction=functionDecorator(function)
        SampledFunction,newShape=getSampleFunctionAndShape(self,newFunction,axis)

        newArraySpecifier=generateNewArraySpecifier_nonUniform(SampledFunction,newShape)
        oldShape=self.array.shape
        newArray=getArray_nonUniform(newArraySpecifier)
        #        log.info('sampled Function={}'.format(SampledFunction))
        #        log.info('sampled Function2={}'.format(SampledFunction.reshape(-1,sampledGrid.dimension)))
        self.array=newArray
        newDimension=newArraySpecifier[1]
        newArrayIsUniform=newArraySpecifier[4]
        if newArrayIsUniform:
            newGridType=self.types['uniform_dependend']
        else:
            newGridType=self.types['nonUniform_dependend']
        self.setDimensionAndGridType(newGridType,newDimension)
        self[:]=SampledFunction.flatten()
        #       for index,gridPart in enumerate(self.array):
        #           gridPart[:len(SampledFunction[index])]=SampledFunction[index]
        self.swapaxes(axis,lastAxis)
        return self
    
    def applyFunctionToAxis_uniform_template(self,functionDecorator,function,axis,*args,**kwargs):
        function=appendArgumentsToFunction(function,*args,**kwargs)
        newFunction=functionDecorator(function)
        array=self.array
        newArray=np.apply_along_axis(newFunction,axis,array)
#        log.info('new shape={}'.format(newArray.shape))
        dimension=np.abs(len(newArray.shape)-len(array.shape))
        self.array=newArray
        self.setDimension(dimension)
        self.shape=newArray.shape
#        log.info('dimension={}'.format(self.dimension))
        return self
            

    def getSampleFunctionAndShape(self,function,axis):            
        array=self.array
        lastAxis=len(originalShape)-1
        
#        log.info('last axis={}'.format(lastAxis))
        if lastAxis!=0:
            if axis!=lastAxis:
                array=array.swapaxes(axis,lastAxis)
            shape=array.shape
            array=array.reshape(-1,originalShape[axis])
        else:
            shape=array.shape
            array=mp.expand_dims(array,0)
            
        SampledFunction=np.array(list(map(function,array)))                
#        log.info('sampled Function.shape={}'.format(SampledFunction.shape))
        return SampledFunction,shape

    def generateNewArraySpecifier_nonUniform(self,SampledFunction,shape):
        def generateMask(axesLength,newShape):
            mask=np.full(newShape,True)
            for index,axis in enumerate(mask):
                axis[:axesLength[index]]=False
            return mask
        
        shapeWithoutAxis=shape[:-1]            
        nrOfAxes=np.sum(shapeWithoutAxis)
        mask=False
        if isIterable(SampledFunction[0]):
            axesLength=np.array(tuple(map(len,SampledFunction)))
            maxAxisLength=np.max(axesLength)
            minAxisLength=np.min(axesLength)

            uniform=False
            if maxAxisLength==minAxisLength:
                uniform=True
                
            if maxAxisLength>1:
                newShape=shapeWithoutAxis+(maxAxisLength,)
                if not uniform:
                    mask=generateMask(axesLength,newShape)
                    
            elif maxAxisLength==1:
                newShape=shapeWithoutAxis+(1,)
                uniform=True
            elif maxAxisLength==0:
                newShape=shapeWithoutAxis
                    
            if isIterable(SampledFunction[0][0]):
                dimension=len(SampledFunction[0][0])
                newDtype=np.object
            else:
                dimension=0
                newDtype=SampledFunction[0].dtype
        else:
            dimension=0
            newShape=shapeWithoutAxis
            newDtype=SampledFunction.dtype
        log.info('new Datatype={}, new Dimension={}'.format(newDtype,dimension))
        return [newShape,dimension,newDtype,mask,uniform]

    def getArray_nonUniform(self,newArraySpecifier):                
        newShape,dimension,newDtype,mask,uniform=newArraySpecifier

        if dimension==0:
            newArray=np.full(newShape,-9999,dtype=newDtype)
        else:
            newArray=np.full(newShape+(dimension,),-9999,dtype=newDtype)
            newArray=getArrayOfArray(newArray)
                
        if not uniform:
            newArray=mp.array(newArray,mask=mask)
        return newArray
        
        
    
    ### generate Apply Function To Elements###
    def generateApplyFunctionToElements(self,gridType):        
        applyFunctionToElements=self.plugInArrayCreationRoutine(self.applyFunctionToElements_template,gridType)
        return applyFunctionToElements
    
    def plugInArrayCreationRoutine(self,applyFunctionToElements_template,gridType):
        if gridType==self.types['nonUniform_dependend']:
            creationRoutine=self.arrayCreation_nonUniform
        else:
            creationRoutine=self.arrayCreation_uniform
        def applyFunctionToElements(self,function,*args,**kwargs):
            sampledGrid=self.applyFunctionToElements_template(creationRoutine,function,*args,**kwargs)
            return sampledGrid
        return applyFunctionToElements    

    def arrayCreation_uniform(self,dimension,newDtype):
        shape=self.array.shape
        if dimension==0:
            newArray=np.full(shape,-9999,dtype=newDtype)
        else:
            newArray=getArrayOfArray(np.full(shape+(dimension,),-9999,dtype=newDtype))
            #            log.info('newArray={}'.format(newArray))
        return newArray
        
    def arrayCreation_nonUniform(self,dimension,newDtype):
        oldArray=self.array
        shape=oldaArray.shape
        mask=oldArray.mask
        if dimension==0:
            newArray=mp.array(np.full(shape,-9999,dtype=newDtype),mask=mask)
        else:
            unMaskedArray=getArrayOfArray(np.full(shape+(dimension,),-9999,dtype=newDtype))
            newArray=mp.array(unMaskedArray,mask=mask)
        return newArray

    def applyFunctionToElements_template(self,arrayCreationRoutine,function,*args,**kwargs):
        function=appendArgumentsToFunction(function,*args,**kwargs)
        SampledFunction=np.array(list(map(function,self[:])))
#        log.info('self shape={}'.format(self[:].shape))
#        log.info('SampledFunction shape={}'.format(SampledFunction.shape))
        #          log.info('function= {}'.format(SampledFunction))
        dimension,newDtype=self.generateDimensionAndDtypeOfNewArray_elements(SampledFunction)

        newArray=arrayCreationRoutine(dimension,newDtype)
        self.array=newArray
        self.setDimension(dimension)
        self.shape=newArray.shape
        self[:]=SampledFunction
        return self
    def generateDimensionAndDtypeOfNewArray_elements(self,SampledFunction):
        try:
            dimension=len(SampledFunction[0])                
            newDtype=SampledFunction[0].dtype
        except TypeError:
            dimension=0
            newDtype=SampledFunction.dtype
            #            log.info('new Datatype={}, new Dimension={}'.format(newDtype,dimension))
        return dimension,newDtype

class NestedArray:
    def __init__(self,array,n_ndim):        
        if isinstance(array,np.ndarray):
            self.is_masked=False
        elif isinstance(array,np.ma.core.MaskedArray):
            self.is_masked=True
        else:
            e=AssertionError('Array has to be a numpy array of type "ndarray" ore "MaskedArray". Provided array type="{}"'.format(type(array)))
            log.error(e)
            raise e

        try:
            assert array.ndim>=n_ndim, '(ndim of subarray) ={} has to be less or equal to (ndim of complete array) ={}'.format(n_ndim,array.ndim)
        except AssertionError as e:
            log.error(e)
            raise

        self._array=array
        self._n_ndim=n_ndim

        
    def __getitem__(self,key):
        return self._array.__getitem__(key)
    
    def __setitem__(self,key,value):
        self._array.__setitem__(key,value)

    def __mul__(self,value):
        return self._array.__mul__(value)                
    def __add__(self,value):
        return self._array.__add__(value)
    def __pow__(self,value):
        return self._array.__pow__(value)
                
    @property
    def array(self):
        return self._array
    @property
    def array1d(self):
        return self._array.reshape(-1,*self.n_shape)
    @property
    def total_ndim(self):
        return self._array.ndim
    @property
    def ndim(self):
        return self._array.ndim-self._n_ndim
    @property
    def n_ndim(self):
        return self._n_ndim
    @property
    def total_shape(self):
        return self._array.shape
    @property
    def shape(self):
        return self.total_shape[:self.ndim]
    @property
    def n_shape(self):
        return self._array.shape[self.ndim:]
    

    @array.setter
    def array(self,new_array):
        try:
            assert new_array.ndim>=self._n_ndim, '(ndim of new array)={} has to be greater or equal to (ndim of subarrays) ={}'.format(len(new_array.shape),self._n_ndim)
        except AssertionError as e:
            log.error(e)
            raise e
        self._array=new_array
    @n_ndim.setter
    def n_ndim(self,new_nested_ndim):
        try:
            assert self.total_ndim>=new_nested_ndim, '(new ndim of subarrays) ={} has to be less or equal to (ndim of total array) ={}'.format(new_nested_ndim,self.total_ndim)
        except AssertionError as e:
            log.error(e)
            raise e
        self._n_ndim=new_nested_ndim
        self.apply=self.generate_apply_method()

    def copy(self):
        array_copy=self._array.copy()
        return NestedArray(array_copy,self._n_ndim)
    def flatten(self):
        return NestedArray(self._array.reshape(-1,self.n_shape[0]),self._n_ndim)
    
    def apply(function):
        array=self._array
        n_ndim=self._n_ndim
        ndim=array.ndim-n_ndim
        value=func(array,n_ndim)
        new_n_ndim=value.ndim-self.ndim

        if value.shape[:ndim]==array.shape[:ndim]:
            e=AssertionError('Applying Function {} changed non-nested shape.'.format(func))
            log.error(e)
            raise e
        
        self._array=value
        self._n_ndim=new_n_ndim  

class CoordinateSystems:
    coord_systems=('spherical','cartesian')
    conversion_routines={coord_systems[0]:{coord_systems[1]:spherical_to_cartesian},coord_systems[1]:{coord_systems[0]:cartesian_to_spherical}}
class ReGrider(CoordinateSystems):
    @classmethod
    def regrid(cls,data,grid,coordinate_type,new_grid,new_coordinate_type,options={}):
        systems=cls.coord_systems        
        try:
            assert (new_coordinate_type in systems) and (coordinate_type in systems),'current and new coordinate types "{}" are not known. Known types are {}. '.format((new_coordinate_type,coordinate_type),systems)
            if (new_coordinate_type in systems) ^ (coordinate_type in systems):
                if (coordinate_type in systems):
                    log.error( 'new coordinate type "{}" is not known. Known types are {}. Continue with current = new type '.format(new_coordinate_type,systems))
                    new_coordinate_type=coordinate_type
                else:
                    log.error( 'coordinate type "{}" is not known. Known types are {}. Continue with current = new type '.format(coordinate_type,systems))
                    coordinate_type=new_coordinate_type
        except AssertionError as e:
            log.error(e)
            raise

            
        if coordinate_type == new_coordinate_type:
            if coordinate_type==cls.coord_systems[0]:
                method=cls.regrid_spher
            else:
                method=cls.regrid_cart
        elif coordinate_type==systems[1] and new_coordinate_type==systems[0]:
            method=cls.regrid_cart_to_spher
        elif coordinate_type==systems[0] and new_coordinate_type==systems[1]:
            method=cls.regrid_spher_to_cart
        #log.error('initial data shape ={}'.format(data.shape))
        #log.error('initial grid shape ={}'.format(grid.total_shape))
        #log.error('new grid shape ={}'.format(new_grid.total_shape))


        if len(data.shape) == 1:
            apply_over_axis = False
        else:
            apply_over_axis = options.get('apply_over_axis',False)
            
        #log.info('apply over axis = {}'.format(apply_over_axis))
        if isinstance(apply_over_axis,bool):         
            new_data=method(data,grid,new_grid,options)
        else:
            application_axis=apply_over_axis            
            reordered_data=np.moveaxis(data,application_axis,0)
            #log.error('reordered_data shape ={}'.format(reordered_data.shape))            
            new_data=np.array(tuple(method(tmp_data,grid,new_grid,options) for tmp_data in reordered_data))
            new_data=np.moveaxis(new_data,0,application_axis)
        return new_data

    @classmethod
    def regrid_cart(cls,data,grid,new_grid,options):
        systems=cls.coord_systems
        dims=new_grid.n_shape[0]
        #log.info('dims={}'.format(dims))
        fill_method = options.get('fill_method', 'constant')
        griddata_opt={'rescale':options.get('rescale',False),'method':options.get('interpolation','nearest'),'fill_value':options.get('fill_value',np.nan)}
        #option to fill missing data by fitting gaussian to existing data
        if fill_method =='gaussian_fit':
            gaussian=cls.generate_gaussian(cls.coord_sys)
            griddata_opt['fill_value']=np.nan
        shape=new_grid.shape
        #        log.error('dims={}'.format(dims))
        if dims == 1:
            #log.info('old_grid shape ={} new grid shape ={} data shape = {}'.format(grid.total_shape,new_grid.total_shape,data.shape))
            new_data=griddata(grid[:],data,new_grid[:], **griddata_opt)
        else:
            new_data=griddata(grid.array.reshape(-1,dims),data.reshape(-1),new_grid.array.reshape(-1,dims), **griddata_opt)
        new_data=new_data.reshape(shape)        
        if fill_method =='gaussian_fit':
            filled_data_mask=(new_data==np.nan)            
            new_data[filled_data_mask]=gaussian(new_grid[filled_data_mask])
        return new_data

    @classmethod
    def regrid_spher(cls,data,grid,new_grid,options):
        systems=cls.coord_systems
        dims=new_grid.n_shape[0]
        fill_method_dict=options.get('fill_method',{'id':'constant'})
        griddata_opt={'rescale':options.get('rescale',False),'method':options.get('interpolation','linear'),'fill_value':fill_method_dict.get('value',np.nan)}
        if fill_method_dict['id']=='gaussian_fit':
            gaussian=cls.generate_gaussian(cls.coord_sys)
            griddata_opt['fill_value']=np.nan
            
        to_cart=cls.conversion_routines[systems[0]][systems[1]]
        to_spher=cls.conversion_routines[systems[1]][systems[0]]
        grid_c = to_cart(grid)
        new_grid_c = to_cart(new_grid)
        new_data=griddata(grid_c.array.reshape(-1,dims),data.reshape(-1), new_grid_c.array.reshape(-1,dims), **griddata_opt)
        new_data=new_data.reshape(new_grid.shape)
        
        if fill_method_dict['id']=='gaussian_fit':
            filled_data_mask=(new_data==np.nan)            
            new_data[filled_data_mask]=gaussian(new_grid[filled_data_mask])
#        cls.data=new_data
        return new_data

    @classmethod
    def regrid_cart_to_spher(cls,data,grid,new_grid,options):
        log.error('grid={}'.format)
        grid=grid.copy()
        dims=grid.n_shape[0]
        fill_method_dict=options.get('fill_method',{'id':'constant'})
        griddata_opt={'rescale':options.get('rescale',False),'method':options.get('interpolation','linear'),'fill_value':fill_method_dict.get('value',np.nan)}
        
        max_dim=np.max(grid[:])
        cartesian_to_spherical(grid)
        trf_support_mask=(grid[...,0]<=max_dim)
        grid=grid[trf_support_mask]
        data=data.copy()[trf_support_mask]
        spherical_to_cartesian(grid)
        new_grid_c=new_grid.copy()
        spherical_to_cartesian(new_grid_c)
        new_data=griddata(grid.reshape(-1,dims),data.reshape(-1),new_grid_c.array.reshape(-1,dims),**griddata_opt)
        new_data=new_data.reshape(new_grid.shape)
        return new_data

    @classmethod
    def regrid_spher_to_cart(cls,data,grid,new_grid,options):
        systems=cls.coord_systems
        grid=grid.copy()
        dims=grid.n_shape[0]
        fill_method_dict=options.get('fill_method',{'id':'constant'})
        griddata_opt={'rescale':options.get('rescale',False),'method':options.get('interpolation','linear'),'fill_value':fill_method_dict.get('value',np.nan)}        
        max_r=np.max(grid[...,0])       
        gaussian=cls.generate_gaussian(data,grid,systems[0],systems[1],options)
        
        spherical_to_cartesian(grid)
        new_data=griddata(grid.array.reshape(-1,dims),data.reshape(-1),new_grid.array.reshape(-1,dims),**griddata_opt)
        new_data=new_data.reshape(new_grid.shape)
        trf_support_mask=np.prod(tuple(new_grid[...,dim]<=max_r for dim in np.arange(dims)),axis=0).astype(bool)*(new_data==np.nan)
        new_data[trf_support_mask]=gaussian(new_data[trf_support_mask])
        return new_data

    @classmethod
    def generate_gaussian(cls,data,grid,coord_system,new_coord_sys,options={}):
        systems=cls.coord_systems
        bins=options.get('radial_bins',int(np.prod(data[0,...].shape)/10))
        sigma=cls.get_gaussian_sigma(data,grid,coord_system,bins)
        amplitude=np.max(data)
        if new_coord_sys==systems[0]:
            def gaussian(points):
                radi=points[...,0]
                return amplitude*np.exp(-(radi/sigma)**2)
        else:
            def gaussian(points):
                radi=np.linalg.norm(points[:],axis=-1)
                return amplitude*np.exp(-(radi/sigma)**2)
        return gaussian

    @classmethod
    def get_gaussian_sigma(cls,data,grid,current_coord_system,bins):
        systems=cls.coord_systems
        sys=current_coord_system

        if sys==systems[1]:            
            grid_radi=np.linalg.norm(grid[:],axis=-1)
        else:
            grid_radi=grid[...,0]
            
        pos_so_averaged_data,radi_edges=np.histogram(data,bins=bins,weights=grid_radi)
        pos_radi=radi_edges[:-1]+np.diff(radi_edges)/2
        radi=np.concatenate((-pos_radi[::-1],pos_radi))
        so_averaged_data=np.concatenate((pos_so_averaged_data[::-1],pos_so_averaged_data))                
        sigma=get_gaussian_weights_1d(so_averaged_data,radi)['sigma']
        return sigma
    

class SampledFunction(CoordinateSystems):
    def __init__(self,grid,data,coord_sys='spherical'):
        self.initial_grid=grid.copy()
        try:
            assert coord_sys in self.coord_systems, 'Unknown coord_sys "{}". Has to be "cartesian" or "spherical".'.format(coord_sys)
        except AssertionError as e:
            log.error(e)
            raise e
        self.initial_coord_sys=coord_sys
        self.initial_data=data.copy()
        self.grid=grid.copy()
        self.coord_sys=self.initial_coord_sys
        self.data=data.copy()
        
    def __getitem__(self,key):
        return self.data.__getitem__(key)
    
    def __setitem__(self,key,value):
        self.data.__setitem__(key,value)

    def override_grid(self,grid):
        self.initial_grid=grid.copy()
        self.grid=grid
        
        
    #uses ReGrider class
    def regrid(self,new_grid,options={}):
        systems=self.coord_systems
        coordinate_type=self.initial_coord_sys        
        new_coordinate_type=options.get('new_coordinate_type',coordinate_type)
        try:
            assert new_coordinate_type in systems,'new coordinate type "{}" is not known. Known types are {}. Assume new = old coordinate type and continue.'.format(new_coordinate_type,systems)
        except AssertionError as e:
            log.error(e)
            new_coordinate_type=coordintae_type

        #log.info('aint before = {}'.format(self.data[:10]))
        #log.info('aint grid before = {}'.format(self.initial_grid[:10]))
        self.data=ReGrider.regrid(self.initial_data,self.initial_grid,coordinate_type,new_grid,new_coordinate_type,options=options)
        #log.info('aint grid after = {}'.format(new_grid[:10]))
        #log.info('aint after = {}'.format(self.data[:10]))
        self.grid=new_grid
        self.coord_sys=new_coordinate_type        

            

    def apply(self,function):
        self.data.apply(function)

        
class gridFactory:
    gridTypes=Grid.types
    
    def constructGrid(self,gridFunctionList,gridType):
        gridArray=self.constructGridArrayAndShape(gridFunctionList,gridType)
        grid=Grid(gridArray,gridType=gridType)
        return grid

    def constructGridArrayAndShape(self,gridFunctionList,gridType):
        types=self.gridTypes
        try:
            Type=types[gridType]
        except KeyError:
            log.error('Given gridType: {} is not known!'.format(gridType))
            raise
            
        if Type==types['uniform']:
            gridArray=self.constructGridArrayAndShape_Uniform(gridFunctionList)
        elif gridType==types['custom']:
            gridArray,gridShape=self.constructGridArrayAndShape_custom(gridFunctionList)
        else:
            gridArray,gridShape=self.constructGridArrayAndShape_Dependend(gridFunctionList,gridType)

        return gridArray

    def constuctGridArrayAndShape_custom(self,gridFunctionList):
        #not implemented yet
        return np.array([]),np.array([])
    
    def constructGridArrayAndShape_Uniform(self,gridFunctionList):
        def generateShape(dimensions):
            def repeatFunction(subList):
                numberOfRepeats=np.prod(subList[:-1])
                return np.repeat(subList[-1],numberOfRepeats)
            dimLengths=self.getDimensionSizeNumbers(dimensions)
            subListsOfDimLengths=list(map(lambda x : dimLengths[:x],np.arange(1,len(dimLengths)+1,1)))
            shape=list(map(repeatFunction,subListsOfDimLengths))
            return shape

        if isinstance(gridFunctionList[0],np.ndarray):
            dimensions=gridFunctionList
        else:
            dimensions=list(map(lambda func: func(),gridFunctionList))
        gridArray=generateMesh(*dimensions,joiningDimension=len(dimensions))
        if len(dimensions)!=1:
            gridArray=getArrayOfArray(gridArray)
        return gridArray

    def constructGridArrayAndShape_Dependend(self,gridFunctionList,gridType):
        
        def generateDimension(gridFunction,previousGridPart=[]):

            def generateDimension_Dependend(previousGridPart,gridFunction):
                elementNumbers=np.arange(len(previousGridPart))
                dimension=np.array(list(map(gridFunction,previousGridPart,elementNumbers)))
                return dimension

            def generateInitialDimension(initialGridFunction):
                dimension=initialGridFunction()
                dimension=np.expand_dims(dimension,axis=1)
                return dimension

            
            if previousGridPart==[]:
                dimension=generateInitialDimension(gridFunction)
            else:
                dimension=generateDimension_Dependend(previousGridPart,gridFunction)
            return dimension

        def appendDimensionListToGridPart_Uniform(part,dimList):
            newDimLength=len(dimList[0])
            repeatedGridPart=np.repeat(part,newDimLength,axis=0)
            newGridPart=np.concatenate((repeatedGridPart,np.expand_dims(dimList.flatten(),axis=1)),axis=1)
            return newGridPart

        def appendDimensionListToGridPart_NonUniform(part,dimList):
            def appendDimensionToGridPart(part,partLength,dim):
                dimLength=len(dim)
                repeatedGridPart=np.full((dimLength,partLength),part)
                newGridPart=np.concatenate((repeatedGridPart,np.expand_dims(dim,axis=1)),axis=1)
                newGridPart=np.split(newGridPart.flatten(),dimLength)
                return newGridPart
            
            subPartLength=len(part[0])
            partLength=len(part)
            subPartLengthIterator=repeat(subPartLength,partLength)
            newGridPart=list(map(appendDimensionToGridPart,part,subPartLengthIterator,dimList))
            newGridPart=flattenList(newGridPart)
            return newGridPart

        
        if gridType==self.gridTypes['uniform_dependend']:
            appendDimensionListToGridPart=appendDimensionListToGridPart_Uniform
        else:
            appendDimensionListToGridPart=appendDimensionListToGridPart_NonUniform

        gridShape=[]
        initialDimension=generateDimension(gridFunctionList[0])
        previousGridPart=initialDimension
        gridShape.append(np.array([ len(initialDimension)] ))
        for func in gridFunctionList[1:]:
            newDimension=generateDimension(func,previousGridPart=previousGridPart)
            gridShape.append(self.getDimensionSizeNumbers(newDimension))
            newGridPart=appendDimensionListToGridPart(previousGridPart,newDimension)
            previousGridPart=newGridPart
        gridArray=np.array(previousGridPart)
        return gridArray,gridShape

    def getDimensionSizeNumbers(self,dimension):
        sizes=np.array(list(map(len,dimension)))
        return sizes
    
class GridFactory:
    grid_types={'uniform':0,'uniform_dependent':1,'dependent':2}
    
    @classmethod
    def construct_grid(cls,grid_type,dimensions_list):
        grid_array=cls.constructGridArrayAndShape(grid_type,dimensions_list)
        grid=NestedArray(grid_array,1)
        return grid
    
    @classmethod
    def constructGridArrayAndShape(cls,grid_type,dimensions_list):
        types=cls.grid_types
        try:
            selected_type=types[grid_type]
        except KeyError:
            log.error('Given gridType: {} is not known!'.format(grid_type))
            raise
            
        if selected_type==types['uniform']:
            grid_array=cls.constructGridArray_Uniform(dimensions_list)
        else:
            grid_array=cls.constructGridArrayAndShape_Dependent(selected_type,dimensions_list)

        return grid_array
    
    @classmethod
    def constructGridArray_Uniform(cls,dimensions_list):
        first_dimension=dimensions_list[0]
        if callable(first_dimension):
            dimensions_list=tuple(dimension_func() for dimension_func in dimensions_list)
        n_lists = len(dimensions_list)
        ndim = sum([dim.ndim for dim in dimensions_list])
        
        if n_lists == ndim:
            grid_array=np.stack(np.meshgrid(*dimensions_list,indexing='ij'),ndim)
        else:
            shapes = tuple(dim.shape for dim in dimensions_list)
            shape = sum(shapes,())
            grid_array = np.zeros(shape,dtype = float)
            for n in range(n_lists):
                grid_array[...,]
            
        return grid_array

    @classmethod
    def constructGridArrayAndShape_Dependent(cls,grid_type,dimensions_list):
        
        def generate_dimension(dimension_func,grid_part=False):            
            is_first_part = isinstance(grid_part,bool)
            if callable(dimension_func):
                if is_first_part:
                    dimension=dimension_func()[:,None]
                else:
                    dimension=dimension_func(grid_part)
                if dimension.ndim == 1:
                    dimension=dimension[...,None]
            else:
                dimension = dimension_func
                if dimension.ndim == 1:
                    dimension=dimension[...,None]
                if not is_first_part:
                    array = np.zeros(grid_part.shape[:-1]+dimension.shape,dtype = dimension.dtype)
                    array[:]=dimension
                    dimension = array
            return dimension

        def append_dimension_uniform_dependent(grid_part,new_dimension):
            #dim_length=new_dimension.shape[-1]
            dim_length=new_dimension.shape[-2]
            grid_part_shape=grid_part.shape
            repeated_grid_part=np.repeat(grid_part,dim_length,-2).reshape(grid_part_shape[:-1]+(dim_length,)+grid_part_shape[-1:])
            #print('repeat shape = {}'.format(repeated_grid_part.dtype))
            #print('append shape = {}'.format(new_dimension.dtype))
            new_grid_part=np.concatenate((repeated_grid_part,new_dimension),axis=-1)
            return new_grid_part
        def append_dimension_dependent(grid_part,new_dimension):
            dim_length=new_dimension.shape[-1]
            grid_part_shape=grid_part.shape
            repeated_grid_part=mp.repeat(grid_part,dim_length,-2).reshape(grid_part_shape[:-1]+(dim_length,)+grid_part_shape[-1:])
            new_grid_part=mp.concatenate((repeated_grid_part,mp.expand_dims(new_dimension,-1)),axis=-1)
            
            vect_size=new_grid_part.shape[-1]
            new_mask=np.kron(np.sum(new_grid_part.mask,axis=-1,dtype=bool),np.full(vect_size,True))
            new_grid_part.mask=new_mask
            return new_grid_part

        
        dimensions_length=[]
        grid_shape=[]
        mask_part=np.array([1]*len(dimensions_list),dtype=int)
        first_dimension=generate_dimension(dimensions_list[0])
        grid_part=first_dimension
        grid_shape.append( (len(first_dimension),) )
        for dimension_func in dimensions_list[1:]:
            new_dimension=generate_dimension(dimension_func,grid_part=grid_part)
            if grid_type==cls.grid_types['dependent']:
                new_grid_part=append_dimension_dependent(grid_part,new_dimension)
            else:
                new_grid_part=append_dimension_uniform_dependent(grid_part,new_dimension)

            grid_part=new_grid_part
        gridArray=grid_part
        return gridArray


        
def double_dimension(grid,dimension):
    #doubles dimension of a grid and adds it in the position after the given dimension. 
    def generate_array(grid,npmp):
#        log.info('application dimension={}'.format(dimension))
        array=grid.array
        n_points=grid.shape[dimension]
        n_dimensions=grid.total_shape[-1]
#        log.info('n_dimensions={}'.format(n_dimensions))
        shape=grid.total_shape
        dimension_indexes= (...,)+(0,)*len(shape[dimension+1:-1])+(dimension,)
#        log.info('dim_indexes={}'.format(dimension_indexes))
        dim_values=array.__getitem__(dimension_indexes)
#        log.info('dim_values_2={}'.format(dim_values))

        repeat_axis=dimension        
        repeat_times=npmp.prod(shape[dimension:-1])
#        log.info('repeat times={}'.format(repeat_times))
        repeated_dim_values=npmp.repeat(dim_values[None,...,None],repeat_times,axis=dimension)
        repeated_dim_values=npmp.moveaxis(repeated_dim_values,-2,0).reshape(n_points,-1,1)
#        log.info('repeated dim values={}'.format(repeated_dim_values))        
        repeated_array=npmp.repeat(array[None,:],n_points,axis=0).reshape(n_points,-1,n_dimensions)
#        log.info('repeated_array={}'.format(repeated_array))
        
        wrong_axis_array=npmp.concatenate((repeated_dim_values,repeated_array),axis=2)
#        log.info('new_array ={}'.format(wrong_axis_array))
        dimension_mask=tuple(range(1,dimension+2))+(0,)+tuple(range(dimension+2,n_dimensions+1))
#        log.info('dimension mask ={}'.format(dimension_mask))
        wrong_shape_array=wrong_axis_array[...,dimension_mask]
#        log.info('wrong shape array={}'.format(wrong_shape_array))
        new_shape=(shape[dimension],)+shape[:dimension+1]+shape[dimension+1:-1]+(shape[-1]+1,)
#        log.info('new shape ={}'.format(new_shape))
        new_array=npmp.moveaxis(wrong_shape_array.reshape(*new_shape),0,dimension+1)
#        log.info('new array={}'.format(new_array))
        return new_array
    
    is_masked=isinstance(grid.array,np.ma.MaskedArray)
    if not is_masked:
        new_array=generate_array(grid,np)
    else:     
        new_array=generate_array(grid,np.ma)
        new_mask=(new_array,grid.ndim+1)
        new_array.mask=new_mask
    new_grid=NestedArray(new_array,1)
    return new_grid

def double_first_dimension(grid):
    def generate_array(grid,npmp):
        total_shape=grid.total_shape
        contracted_shape=(total_shape[0],np.prod(total_shape[1:-1]),total_shape[-1])
        first_dimension=grid.array.reshape(contracted_shape)[:,0,0]
        dim_length=len(first_dimension)
        
        new_dimension=npmp.kron(np.expand_dims(np.full(grid.shape,1),-1),first_dimension)
        
        repeated_grid=npmp.repeat(grid.array,dim_length,-2).reshape(total_shape[:-1]+(dim_length,)+total_shape[-1:])
#        log.info('repeated grid shape={}'.format(repeated_grid.shape))
        new_grid_array=npmp.moveaxis(npmp.concatenate((np.expand_dims(new_dimension,-1),repeated_grid),axis=-1),-2,0)
        return new_grid_array
    def generate_mask(new_array,new_grid_dim):
        new_mask=np.kron(np.prod(new_array.mask,-1,dtype=bool),np.full(new_grid_dim,True)).astype(bool)
        return new_mask

    is_masked=isinstance(grid.array,np.ma.MaskedArray)
    if not is_masked:
        new_array=generate_array(grid,np)
    else:     
        new_array=generate_array(grid,np.ma)
        new_mask=(new_array,grid.ndim+1)
        new_array.mask=new_mask
        
    new_grid=NestedArray(new_array,1)
    return new_grid

def uniformGrid_func(domain,nPoints='not specified',stepSize='not specified',noEndpoint=False):
    def uniformGridViaNPoints(domain,nPoints):
        def gridFunc(*args):
            if nPoints<=1:
                stepSize=0
            else:
                if noEndpoint:
                    stepSize=(domain[1]-domain[0])/nPoints
                else:
                    stepSize=(domain[1]-domain[0])/(nPoints-1)
#                    log.info('stepSize={}'.format(stepSize))
            gridArray=np.arange(nPoints,dtype=float)*stepSize+domain[0]
#            log.info('gridArray={}'.format(gridArray))
            return gridArray
        return gridFunc
    
    def uniformGridViaStepSize(domain,stepSize):
        def gridFunc(*args):
            if domain[0]==domain[1]:
                nPoints=1
            else:
                if noEndpoint:
                    nPoints=np.floor((domain[1]-domain[0])/stepSize)
                else:                        
                    nPoints=np.floor((domain[1]-domain[0])/stepSize)+1
            gridArray=np.arange(nPoints,dtype=float)*stepSize+domain[0]
            return gridArray
        return gridFunc
    def checkArguments(nPoints,stepSize):
        try:
            assert (nPoints!='not specified' or stepSize!='not specified')
        except AssertionError:
            log.warning( 'uniformGrid_func needs either a stepSize or nPoints as input.')

    checkArguments(nPoints,stepSize)
    gridFunc=[]
    if (nPoints!='not specified'):
        if stepSize=='not specified':
            gridFunc=uniformGridViaNPoints(domain,nPoints)
        else:
            log.warning('Number of steps as well as step size is specified. Defaulting to claculate grid for the number of Steps.')
            gridFunc=uniformGridViaNPoints(domain,nPoints)
    else:
        gridFunc=uniformGridViaStepSize(domain,stepSize)

    return gridFunc
    
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

def getGridByXYValues(xValues,yValues):
    def xfunc(*args):
        return xValues
    def yfunc(*args):
        return yValues
    gfact=gridFactory()
    newGrid=gfact.constructGrid([xfunc,yfunc],'uniform')
    return newGrid


def get_linspace_log2(domain, target_step_size,additional_steps = 0, return_parameters = False):
    try:
        assert (domain[0]<=0) and (domain[1]>0), 'domain: ( {} ) does not contain 0.'.format(domain)
    except AssertionError as e:
        log.error(e)
        traceback.print_exc()
        raise e
    domain = np.array(domain)
    length = domain[1]-domain[0]
    target_number_of_steps = length/target_step_size
    n_steps = 2**int(np.round(np.log2(target_number_of_steps)))+additional_steps
    
    partition = np.abs(np.ceil(domain/length*n_steps+0.5)).astype(int) #round up
    if partition[0] == partition[1]:
        partition[0] -= 1
    if domain[0] !=0:
        step_sizes = np.abs( (domain[0]/(partition[0]-1),domain[1]/(partition[1]-1)) )
        exact_domain_id = np.argmin(step_sizes) 
        step_size = step_sizes[exact_domain_id]
        log.info('step_size = {}'.format(step_size))
    else:
        step_size = domain[1]/(partition[1]-1)
        exact_domain_id = 1
        
    if exact_domain_id ==0:
        lin_space = domain[0] + np.arange(n_steps)*step_size
    else:
        lin_space = domain[1] - np.arange(n_steps)[::-1]*step_size
    new_domain = [lin_space.min(),lin_space.max()]
        
    parameters = {'step_size':step_size,'number_of__steps':n_steps,'domain':new_domain}
    if return_parameters:
        return lin_space,parameters
    else:
        return lin_space
        
