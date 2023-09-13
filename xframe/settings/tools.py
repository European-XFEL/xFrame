from types import SimpleNamespace

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
            print(e)
            print('Known attributes are {}'.format(list(self.keys())))
            raise
        return value
    
