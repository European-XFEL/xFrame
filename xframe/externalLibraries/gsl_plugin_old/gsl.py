import logging
import numpy as np
from itertools import repeat
import ctypes as C
from ctypes import c_double,c_int,c_size_t,c_float

from xframe.externalLibraries.gsl_plugin import gslWrapper
from xframe.externalLibraries.gsl_plugin import gslStructures as gslStruct
from xframe.library.interfaces import GSLInterface

log=logging.getLogger('root')
gsl=gslWrapper.Wraper()
c_pint=C.POINTER(c_int)
c_pdouble=C.POINTER(c_double)
c_pfloat=C.POINTER(c_float)
c_psize_t=C.POINTER(c_size_t)

class FFTPrecomputedDataHandler:
    class FFTPrecomputedData:
        dataTypes={'complex':0,'real':1,'halfcomplex':2}
        
        def __init__(self,dataType):            
            dataTypes=self.dataTypes                
            try:
                if dataType==dataTypes['complex']:
                    generateDatasetFactoryArgs=[gsl.gsl_fft_complex_wavetable_alloc,gsl.gsl_fft_complex_workspace_alloc]
                    deleteDatasetFactoryArgs=[gsl.gsl_fft_complex_wavetable_free,gsl.gsl_fft_complex_workspace_free]
                elif dataType==dataTypes['real']:
                    generateDatasetFactoryArgs=[gsl.gsl_fft_real_wavetable_alloc,gsl.gsl_fft_real_workspace_alloc]
                    deleteDatasetFactoryArgs=[gsl.gsl_fft_real_wavetable_free,gsl.gsl_fft_real_workspace_free]
                elif dataType==dataTypes['halfcomplex']:
                    generateDatasetFactoryArgs=[gsl.gsl_fft_halfcomplex_wavetable_alloc]
                    deleteDatasetFactoryArgs=[gsl.gsl_fft_halfcomplex_wavetable_free]
                else:
                    raise AssertionError('FFTPrecomputedData does not know the data type = {} \n Known dataTypes are listed as class varibales starting with dataType_'.format(dataType))
            except AssertionError as error:
                log.error(error)

            self.dataType=dataType
            self.dataDict={'fftSizes':[],'precomputedData':[]}
            self.generateDataset=self.generateDataset_MethodFactory(*generateDatasetFactoryArgs)
            self.deleteSingleDataset=self.deleteDataset_MethodFactory(*deleteDatasetFactoryArgs)


        def getDataset(self,fftSize):
            dataDict=self.dataDict
            fftSizes=dataDict['fftSizes']
        
            datasetExists=fftSize in fftSizes                
            if datasetExists:
                log.info('use existing dataset. Type={}'.format(self.dataType))
                dataIndex=fftSizes.index(fftSize)
                precomputedDataset=dataDict['precomputedData'][dataIndex]
            else:
                precomputedDataset=self.generateDataset(fftSize)
                self.addDataset(fftSize,precomputedDataset)

            return precomputedDataset
        
        def deleteDatasets(self,fftSizes):
            list(map(self.deleteSingleDataset,np.asarray(fftSizes)))

        def deleteAll(self):
            self.deleteDatasets(self.dataDict['fftSizes'])
        def addDataset(self,fftSize,precomputedDataset):
            dataDict=self.dataDict                
            dataDict['fftSizes'].append(fftSize)
            dataDict['precomputedData'].append(precomputedDataset)


        def generateDataset_MethodFactory(self,*args):
            wavetable_alloc=args[0]
            isNotHalfecomplexType= len(args)==2
            if isNotHalfecomplexType:
                workspace_alloc=args[1]
            
            def generateDataset1(fftSize):
                log.info('generate Dataset for size= {}'.format(fftSize))
                size=c_size_t(fftSize)
                wavetablePointer=wavetable_alloc(size)
                workspacePointer=workspace_alloc(size)
                precomputedDataset=[wavetablePointer,workspacePointer]
                return precomputedDataset
            
            def generateDatasetHalfcomplex(fftSize):
                size=c_size_t(fftSize)
                wavetablePointer=wavetable_alloc(size)
                precomputedDataset=[wavetablePointer]
                return precomputedDataset
            
            if isNotHalfecomplexType:
                generateDataset=generateDataset1
            else:
                generateDataset=generateDatasetHalfcomplex
                
            return generateDataset

        def deleteDataset_MethodFactory(self,*args):
            wavetable_free=args[0]
            isNotHalfecomplexType= len(args)==2
            if isNotHalfecomplexType:
                workspace_free=args[1]
            
            def deleteDataset1(fftSize):
                dataDict=self.dataDict
                fftSizes=dataDict['fftSizes']
                datasetExists=fftSize in fftSizes
                
                if datasetExists:                        
                    dataIndex=fftSizes.index(fftSize)
                    precomputedData=dataDict['precomputedData']
                    precomputedDataset=precomputedData[dataIndex]
                    
                    #free allocated memory
                    wavetablePointer=precomputedDataset[0]
                    workspacePointer=precomputedDataset[1]
                    wavetable_free(wavetablePointer)
                    workspace_free(workspacePointer)

                    #delete dataDict entries
                    del fftSizes[dataIndex]
                    del precomputedData[dataIndex]                        
                else:
                    log.warning('Dataset to Fourier Size {} does not exist! Nothing was deleted.'.format(fftSize))
                    
            def deleteDatasetHalfcomplex(fftSize):
                dataDict=self.dataDict
                fftSizes=dataDict['fftSizes']
                datasetExists=fftSize in fftSizes
                
                if datasetExists:                        
                    dataIndex=fftSizes.index(fftSize)
                    precomputedData=dataDict['precomputedData']
                    precomputedDataset=precomputedData[dataIndex]
                    
                    #free allocated memory
                    wavetablePointer=precomputedDataset[0]
                    wavetable_free(wavetablePointer)

                    #delete dataDict entries
                    del fftSizes[dataIndex]
                    del precomputedData[dataIndex]                        
                else:
                    log.warning('Dataset to Fourier Size {} does not exist! Nothing was deleted.'.format(fftSize))
                    
            if isNotHalfecomplexType:
                deleteDataset=deleteDataset1                                    
            else:
                deleteDataset=deleteDatasetHalfcomplex
            return deleteDataset
            
    dataTypes=FFTPrecomputedData.dataTypes
    def __init__(self):
        dataTypes=self.dataTypes
        FFTPrecomputedData=self.FFTPrecomputedData
        precomputedDataDict={}
        precomputedDataDict[dataTypes['complex']]=FFTPrecomputedData(dataTypes['complex'])
        precomputedDataDict[dataTypes['real']]=FFTPrecomputedData(dataTypes['real'])
        precomputedDataDict[dataTypes['halfcomplex']]=FFTPrecomputedData(dataTypes['halfcomplex'])

        self.precomputedDataDict=precomputedDataDict

        
    def getDataset(self,fftSize,dataType):
        dataTypes=self.dataTypes
        self.dataTypeIsValid(dataType)
        
        dataSet=self.precomputedDataDict[dataType].getDataset(fftSize)

        if dataType==dataTypes['halfcomplex']:
            workspacePointer=self.precomputedDataDict[dataTypes['real']].getDataset(fftSize)[1]
            dataSet=[dataSet[0],workspacePointer]
            
        return dataSet
    
    def deleteDatasets(self,fftSizes,dataType):
        self.dataTypeIsValid(dataType)
        self.precomputedDataDict[dataType].deleteDatasets(fftSizes)
        
    def deleteAll(self,dataType=False):
        if isinstance(dataType,bool):
            for key,value in self.precomputedDataDict.items():
                value.deleteAll()
        else:
            self.dataTypeIsValid(dataType)
            self.precomputedDataDict[dataType].deleteAll()

    def dataTypeIsValid(self,dataType):
        try:
            if dataType in self.dataTypes.values():
                isValidType=True
            else:
                raise AssertionError('FFTPrecomputedData does not know the data type = {} \n Known dataTypes are listed int the class varibale dataTypes'.format(dataType))
        except AssertionError as error:
            log.error(error)
            raise
        return isValidType

class DHTPrecomputedData():
    def __init__(self):
        self.dataDict={}

    def generateDataset(self,dht_size,nus,maxR):
        xmax=c_double(maxR)
        precomputed_data_tuple=()
        for nu in nus:                
            precomputed_data_tuple+=(gsl.gsl_dht_new(c_size_t(dht_size),c_double(abs(nu)),xmax),)
        precomputed_data_array_type=C.POINTER(gslStruct.gsl_dht_struct)*len(nus)
        precomputed_data_array=precomputed_data_array_type(*precomputed_data_tuple)
        precomputed_data_pointer=C.cast(C.pointer(precomputed_data_array),C.POINTER(C.POINTER(gslStruct.gsl_dht_struct)))
        
        log.info('constructed dht r_points ={}, number of orders={} and maxR={}'.format(dht_size+1,len(nus),xmax))
        self.dataDict[(dht_size,nus.tobytes(),maxR)]=precomputed_data_pointer
        return precomputed_data_pointer
    
    def generate_single_dataset(self,dht_size,nu,maxR):
        xmax=c_double(maxR)
        precomputed_data_pointer=gsl.gsl_dht_new(c_size_t(dht_size),c_double(abs(nu)),xmax)
        return precomputed_data_pointer

    def getDataset(self,specifier):
        # specifier is of the form (dht_size,nus,maxR)[float,ndarray or float,float]
        nus=specifier[1]
        #log.info('nus={}'.format(nus))
        if not isinstance(nus,np.ndarray):
            nus=np.array([nus])
            nus.flags.writeable=False
        elif nus.flags.writeable:
            log.warning('Array of nus/orders needs to be readonly! Changing array readability and continue.')
            nus.flags.writeable=False
        specifier=(specifier[0],nus,specifier[2])
        precomputed_dataset=self.dataDict.get((specifier[0],nus.tobytes(),specifier[2]),False)
        dataset_does_not_exists=isinstance(precomputed_dataset,bool)
        if dataset_does_not_exists:
            precomputed_dataset=self.generateDataset(*specifier)
        return precomputed_dataset
    
    def get_single_dataset(self,specifier):
        return self.generate_single_dataset(*specifier)
        
    def deleteAll():
        dataDict=self.dataDict
        list(map(gsl.gsl_dht_free,dataDict.values()))
        dataDict={}

        
class gslMath(GSLInterface):
    def __init__(self):
        gsl.gsl_set_error_handler_off()
        self.dht_precomputed_data_handler=DHTPrecomputedData()
        self.fft_precomputed_data_handler=FFTPrecomputedDataHandler()
        self.fft_data_types=self.fft_precomputed_data_handler.dataTypes

    
    #FFT    
    def assembleFFTArguments(self,data,dataType):
        #I am assuming data is a np array of the proper datatype (np.double for real and halfcomplex and np.complex for complex fft)
        fft_data_types=self.fft_data_types        
        fftSize=data.shape[-1]
        data=data.reshape(-1,fftSize)
        if dataType==fft_data_types['complex']:
            fftSize=int(fftSize/2)
#        log.info('fft size={}'.format(fftSize))
        stride=c_size_t(1)            
        fftSizeC=c_size_t(fftSize)
        precomputedData=self.fft_precomputed_data_handler.getDataset(fftSize,dataType)
  #      log.info('{} arguments =\n{}'.format(dataType,data))
        data_pointers=[data_part.ctypes.data_as(C.POINTER(c_double)) for data_part in data]
                    
        arguments=[data_pointers,repeat(stride),repeat(fftSizeC),repeat(precomputedData[0]),repeat(precomputedData[1])]
        return arguments

    
    def fft_complex_mixedRadix_forward(self,data):
        data=self.fft_complex_pack(data)
        shape=copy(data.shape)
#        log.info('fft complex mixedradix forward data shape ={}'.format(data.shape))
        arguments=self.assembleFFTArguments(data,self.fft_data_types['complex'])
#        log.info('data in forward =\n{}'.format(arguments))
        retVal=map(gsl.gsl_fft_complex_forward,arguments[0],*arguments[1:])
        #        log.info('data out forward =\n{}'.format(data))
        retVal=retVal.reshape(shape)
        unpackedData=self.fft_complex_unpack(data)
#        log.info('data out forward =\n{}'.format(unpackedData))
        return unpackedData
    
    def fft_complex_mixedRadix_inverse(self,data):
        
        data=self.fft_complex_pack(data)
#        log.info('data in inverse =\n{}'.format(data))
        arguments=self.assembleFFTArguments(data,self.fft_data_types['complex'])
        retVal=gsl.gsl_fft_complex_inverse(*arguments)
#        log.info('retVal={}'.format(retVal))
#        log.info('data out inverse =\n{}'.format(data))
        unpackedData=self.fft_complex_unpack(data)
        return unpackedData

    
    def fft_real_mixedRadix_forward(self,data):
        arguments=self.assembleFFTArguments(data,self.fftDataTypes['real'])        
        retVal=gsl.gsl_fft_real_forward(*arguments)
        data=self.fft_halfcomplex_unpack(data)
        return data
    
    
    def fft_halfcomplex_mixedRadix_inverse(self,data):
        data=self.fft_halfcomplex_pack(data)
        arguments=self.assembleFFTArguments(data,self.fftDataTypes['halfcomplex'])
        retVal=gsl.gsl_fft_halfcomplex_inverse(*arguments)
#        log.info('data returned=\n{}'.format(data))
        return data

    def fft_complex_pack(self,data):
#        log.info('input for packing dtype={}'.format(data.dtype))        
        shape=data.real.shape
        new_shape=shape[:-1]+(shape[-1]*2,)
        packed_data=np.zeros(new_shape,dtype=float)
        packed_data[...,0::2] = data.real
        packed_data[...,1::2] = data.imag
        
#        log.info('fft packed_data shape={}'.format(packed_data.shape))
        return packed_data
    
    def fft_complex_unpack(self,data):
  #      log.info('start unpacking')
        unpacked_data=data[...,0::2]+1.j*data[...,1::2]
#        log.info('fft unpacked data shape ={}'.format(unpacked_data.shape))
        return unpacked_data

    def fft_halfcomplex_pack(self,complexArray):
        lenData=len(complexArray)   
        packedArray=np.dstack((complexArray.real,complexArray.imag)).flatten()
        halfcomplexArray=np.concatenate((np.array([packedArray[0]]),packedArray[2:lenData+1]))
        #            log.info('halfcomplexArray={}'.format(halfcomplexArray))
        return halfcomplexArray

    def fft_halfcomplex_unpack(self,data):
        dataLength=len(data)
        unpackedArray=np.zeros(dataLength*2,dtype=np.float)
        
        dataPointer=data.ctypes.data_as(C.POINTER(c_double))
        unpackedArrayPointer=unpackedArray.ctypes.data_as(C.POINTER(c_double))

        stride=c_size_t(1)
        size=c_size_t(dataLength)
        
        gsl.gsl_fft_halfcomplex_unpack(dataPointer,unpackedArrayPointer,stride,size)
        complexArray=unpackedArray[0::2]+1.j*unpackedArray[1::2]
        return complexArray


    #bessel Zero
    
    def besselZero(self,order,zeroNumber):
        order=c_double(abs(order))
        zeroNumber=c_int(zeroNumber)
        zero=gsl.gsl_sf_bessel_zero_Jnu(order,zeroNumber)
        return zero

    #Legendre Polynomial
        
    def legendrePoly(self,degree,x):
        degree=c_int(degree)
        x=c_double(x)
        value=gsl.gsl_sf_legendre_Pl(degree,x)
        return value
        
    #DHT
    def generate_dht_2D(self,n_radial_points,orders,maxR):
        #assumes data is complex numpy array
        #gsl is quite confusing here !! the dhtSize to initialize the transform is M-1 where M indicates the number of sample points where f(u_M) is supposed to be 0
        dht_size=n_radial_points
        n_orders=len(orders)
        c_n_orders=c_size_t(n_orders)
        data_specifier=(dht_size,orders,maxR)
        precomputed_data_pointer=self.dht_precomputed_data_handler.getDataset(data_specifier)

        reciprocal_coeff=np.zeros((n_orders,n_radial_points),dtype=np.complex,order='C')
        coeff=np.zeros((n_orders,n_radial_points),dtype=np.complex,order='C')

        #forward_prefactor=1.j**(-1*orders)*4*np.pi*maxR**2/np.array(tuple(map(self.besselZero,orders,np.full(n_orders,dht_size+1))))
        #inverse_prefactor=1.j**orders/(np.pi*maxR**2)

        forward_prefactor=((-1.j)**orders)*4*np.pi*(maxR/np.array(tuple(map(self.besselZero,orders,np.full(n_orders,dht_size+1)))))**2
        inverse_prefactor=(1.j**orders)/(np.pi*maxR**2)
        
        def dht_2D_forward(harmonic_coeffs):
            gsl.dht_2D(c_n_orders,precomputed_data_pointer,harmonic_coeffs,reciprocal_coeff,forward_prefactor)
            return reciprocal_coeff
        
        def dht_2D_inverse(reciprocal_harmonic_coeffs):
            gsl.dht_2D(c_n_orders,precomputed_data_pointer,reciprocal_harmonic_coeffs,coeff,inverse_prefactor)
            return coeff
        return dht_2D_forward,dht_2D_inverse
        
    def generate_dht_bare(self,n_radial_points,order,maxR,prefactor):
        dht_size=n_radial_points
        data_specifier=(dht_size,order,maxR)
        precomputed_data_pointer=self.dht_precomputed_data_handler.getDataset(data_specifier)

        complex2=c_double*2
        prefactor=gslStruct.gsl_complex(complex2(prefactor.real,prefactor.imag))
        
        coeff=np.zeros(n_radial_points,dtype=np.complex,order='C')

        def dht_bare(harmonic_coeff):
            gsl.dht_bare(precomputed_data_pointer,harmonic_coeff,coeff,prefactor)
            return coeff
        return dht_bare

    def dht_test(self,data):
        gsl.dht_test(c_size_t(data.shape[0]),c_size_t(data.shape[1]),data)
    
        
    
    def dht_forward2(self,data,nu,maxR):        
        #assumes data is complex numpy array
        #gsl is quite confusing here !! the dhtSize to initialize the transform is M-1 where M indicates the number of sample points where f(u_M) is supposed to be 0
        dhtSize=len(data)


        transformedRealData=np.zeros(dhtSize,dtype=np.float)
        transformedImagData=np.zeros(dhtSize,dtype=np.float)
        dataReal=np.copy(data.real)
        dataImag=np.copy(data.imag)

        initializedWorkspacePointer=self.dht_precomputed_data_handler.get_single_dataset((dhtSize,nu,maxR))
        gsl.gsl_dht_apply(initializedWorkspacePointer,dataReal.ctypes.data_as(C.POINTER(c_double)),transformedRealData.ctypes.data_as(C.POINTER(c_double)))
        gsl.gsl_dht_apply(initializedWorkspacePointer,dataImag.ctypes.data_as(C.POINTER(c_double)),transformedImagData.ctypes.data_as(C.POINTER(c_double)))
        transformedData=transformedRealData+1.j*transformedImagData
        return transformedData

    def dht_inverse2(self,data,nu):
        #not precisely invertible for low values of M = len(data)+1
        dhtSize=len(data)
        besselZero=self.besselZero(nu,dhtSize+1)
        transformedData=(besselZero**2)*self.dht_forward(data,nu)
        return transformedData


