import abc

class DatabaseInterfaceDetector(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass
    @abc.abstractmethod
    def save(self):
        pass

