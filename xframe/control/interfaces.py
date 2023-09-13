import abc

class SharedMemoryInterface(abc.ABC):

    #_dict structure is as follows
    # {mem_name:{'shape': <tuple(dim1,...)>,'dtype': <np.dtype object >}}
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
    
        
