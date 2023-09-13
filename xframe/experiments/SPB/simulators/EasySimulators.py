import logging
import numpy as np

from .interfaces import DetectorInterfaceSimulation
from xframe.experiment.interfaces import SimulationInterface

log=logging.getLogger('root')



class RandomSimulator(SimulationInterface):
    def __init__(self,detector,experimentalSetup):
        try:
            assert isinstance(detector,DetectorInterfaceSimulation)
        except AssertionError:
            log.error('The Detector {0} given to {1} does not implement {2}.'.format(detector,type(self).__name__,DetectorInterfaceSimulation.__name__))
            raise
        self.detector=detector

        try:
            assert 'SampleDetectorDistance' in experimentalSetup.keys()
        except AssertionError:
            log.error('experimentalSetup does not contain the Sample Distance')
            raise
        self.experimentalSetup=experimentalSetup
        
    def getData(self,*args,**kwargs):
        amplitude=kwargs.get('Amplitude',1)
        pixelArrayShape=self.detector.getGeometry()['PixelKornerIndex'].shape
        randomImage=amplitude*np.random.rand(pixelArrayShape[0],pixelArrayShape[1])
        return randomImage

    def getGeometry(self):
        geometry=self.detector.getGeometry()
        return geometry

    def getExperimentalSetup(self):
        return self.experimentalSetup


class MockSimulator(SimulationInterface):
    def getData(self):
        pass
    def getGeometry(self):
        pass
    def getExperimentalSetup(self):
        pass
