import abc


class PresenterInterface(abc.ABC):
    @abc.abstractmethod
    def present(self):
        pass
    @abc.abstractmethod
    def present2(self):
        pass


class DatabaseInterfaceAnalysis(abc.ABC):
    @abc.abstractmethod
    def load(self,analysisType,restultId):
        pass
    @abc.abstractmethod
    def save(self,analysisType,restultId):
        pass
