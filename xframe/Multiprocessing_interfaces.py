from xframe.interfaces import DependencyMeta
import abc
class SharedMemoryInterface(abc.ABC):
    #_dict structure is as follows
    # {mem_name:{'shape': <tuple(dim1,...)>,'dtype': <np.dtype object >}}
    _external_dependency_name_ = 'shared_array'
    @abc.abstractmethod
    def allocate(self,_dict):
        pass
    @abc.abstractmethod
    def delete(self,mem_names):
        pass
    @abc.abstractmethod
    def attach(self,mem_names):
        pass
    @abc.abstractmethod
    def detach(self,mem_names):
        pass    
    @abc.abstractmethod
    def __getitem__(self,index):
        pass
    @abc.abstractmethod
    def __len__(self):
        pass
    @abc.abstractmethod
    def __iter__(self):
        pass

class SharedMemoryDependency(SharedMemoryInterface,metaclass = DependencyMeta):
    pass
    
class ClFunctionInterface(abc.ABC):
    @abc.abstractmethod
    def assemble_buffers(self,kernel_data):
        pass
    @abc.abstractmethod
    def assemble_function(self,kernel_data):
        pass
class ClProcessInterface(abc.ABC):
    @abc.abstractmethod
    def assemble(self,kernel_data):
        pass
    
class OpenClInterface(abc.ABC):
    _external_dependency_name_ = 'opencl'
    ClProcess = ClProcessInterface
    ClFunction = ClFunctionInterface
    @abc.abstractmethod
    def create_context(self):
        pass
    @abc.abstractmethod
    def create_functions(self,kernel_data):
        pass
    @abc.abstractmethod
    def create_process_on_all_gpus(self,kernel_data):
        pass
    @abc.abstractmethod
    def get_number_of_gpus(self,kernel_data):
        pass


class OpenClDependency(OpenClInterface,metaclass=DependencyMeta):
    pass

class PsutilInterface(abc.ABC):
    _external_dependency_name_ = 'psutil'
    @abc.abstractmethod
    def get_free_memory(self):
        pass

class PsutilDependency(PsutilInterface,metaclass = DependencyMeta):
    pass

