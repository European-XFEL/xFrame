#from dataclasses import dataclass
import numpy as np
import numpy.ma as mp
import logging

log=logging.getLogger('root')

from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import uniformGridGetStepSizes
from xframe.library.gridLibrary import SampledFunction
from .analysisLibrary.misk import returnGrid_deleteFirstDimension



class BCoefficientData(SampledFunction):
    def __init__(self,grid,dataGrid,dimension):
        self.dimension=dimension
        super().__init__(grid,dataGrid)
    def apply(self,function):
        newDataGrid=self.data.apply(function)
        newCrossCorrelationData=BCoefficientData(self.grid,newDataGrid,self.dimension)
        return newCrossCorrelationData
    def apply_along_axis(self,function,axis):
        newDataGrid=self.data.apply_along_axis(function,axis)
#        newGrid=self.grid.swapaxes(axis,len(self.grid.shape)-1)
        newCrossCorrelationData=BCoefficientData(self.grid,newDataGrid,self.dimension)
        return newCrossCorrelationData

class FXS_Data():
    def __init__(self,ccd,incident_wavelength,invariant_extraction_grid,average_intensity=False):
        self.incident_wavelength=incident_wavelength
        self.invariant_extraction_grid=invariant_extraction_grid
        #CCData is supposed to be an instance of SampledFunction
        self.CCD=ccd
        self.aInt=average_intensity
        
        def extractAutocorrelation(self):
            ACD_data=np.moveaxis(self.CCD.diagonal(0,0,1),-1,0).copy()            
            ACD=SampledFunction(invariant_extraction_grid,ACD_data)            
            return ACD

        self.extractAutocorrelation=extractAutocorrelation
        if isinstance(self.CCD,SampledFunction):
            self.ACD=extractAutocorrelation(self)
        else:
            self.ACD=False

        self.dimension=False
        self.bCoeff=False
        self.usedHarmOrders=False
        self.projection_vector=False

'''
@dataclass
class ReciprocalProjectionData:
    average_intensity : object
    proj_matrices : object

@dataclass
class MTIP_Data():
    max_Q : float
    grid : NestedArray = False
    ReciprocalProjectionData
'''

class FTGridPair:
    def __init__(self,realGrid,reciprocalGrid):
        self.realGrid=realGrid
        self.reciprocalGrid=reciprocalGrid



class ProjectionOrders:
    def __init__(orders,ids):
        self.orders=orders
        self.ids=ids
        assert len(orders) == len(ids),'length of orders and ids must match.'

    def __getitem__(self,key):
        return self.orders.__getitem__(key)
    
    def __setitem__(self,key,value):
        self.orders.__setitem__(key,value)
