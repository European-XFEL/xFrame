import abc

class DetectorInterfaceSimulation(abc.ABC):
    @abc.abstractmethod
    def get_geometry(self):
        '''
        should return a dict containing the keys pixelGrid,pixelKornerIndex
        '''
        pass
