from types import SimpleNamespace
import numpy as np
import traceback
import logging
log = logging.getLogger('root')

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
    

class SettingsConverter:
    @classmethod
    def list_to_str(cls,l,end=']'):
        out='['
        for part in l:
            if isinstance(part,(list,tuple)):
                out+= cls.list_to_str(part,end='],')
            else:
                out+= str(part)+','            
        return out+end
    @classmethod
    def numpy_to_list(cls,array):
        out=[]
        if array.ndim==1:
            out=array.tolist()
        elif array.ndim>1:
            for part in array:
                out += [cls.numpy_to_list(part)]
        return out
    @classmethod
    def numpy_to_str(cls,array):
        dtype=array.dtype
        if dtype==np.dtype('object'):
            raise AssertionError('Cannot convert numpy array of type object to yaml settings.')
    
        l = cls.numpy_to_list(array)
        st = cls.list_to_str(l)
        numpy_str = f'np.array({st},dtype=np.{str(dtype)})'
        return numpy_str
    @classmethod
    def numpy_type_to_str(cls,val):
        _type= type(val).__name__
        str_val=str(val)
        numpy_str = f'np.{_type}({str_val})'
        return numpy_str
    @classmethod
    def recursive_convert_list(cls,l):
        out=[]
        for _id,val in enumerate(l):
            if isinstance(val,(np.number,np.bool_)):
                out.append(val.item())
            elif isinstance(val,(type(None),str,complex,float,int,bytes,bool)):
                # order is important otherwise np.float64 will be treated as float
                out.append(val)                
            elif isinstance(val,np.ndarray):
                log.warning(f'Numpy arrays not allowed in lists or tuples when saving settings to yaml. Skipping array.')                
            elif isinstance(val,(list,tuple)):                
                out.append(cls.recursive_convert_list(val))
            else:
                log.warning(f'Could not convert element {val} of type {type(val)} to yaml settings format.Skipping.')
        return out
                                        
    @classmethod
    def recoursive_convert_settings(cls,settings):
        out_dict={}
        for key,val in settings.items():
            key = str(key)
            try:
                if isinstance(val,(type(None),str,complex,float,int,bytes,bool)):
                    out_dict[key]=val
                elif isinstance(val,(np.number,np.bool_)):
                    out_dict[key]={'command':cls.numpy_type_to_str(val)}
                elif isinstance(val,np.ndarray):
                    out_dict[key]={'command':cls.numpy_to_str(val)}
                elif isinstance(val,slice):
                    out_dict[key]={'command': str(val)}
                elif isinstance(val,(list,tuple)):
                    out_dict[key]=cls.recursive_convert_list(val)
                elif isinstance(val, (dict,DictNamespace)):
                    out_dict[key]=cls.recoursive_convert_settings(val)
                else:
                    out_dict[key]=val
                    #raise ValueError('Cannot convert {} type for settings key {}'.format(type(val),key))
            except Exception as e:
                log.warning("Failed to save key {} and value type {} with error: \n {}".format(key,type(val),e))
                traceback.print_exc()
        return out_dict

    @classmethod
    def convert(cls,settings):
        return cls.recoursive_convert_settings(settings)
