from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap,CommentedSeq
from xframe.library.pythonLibrary import DictNamespace
import numpy as np
import logging
import traceback

log=logging.getLogger('root')
yaml = YAML()

class YAML_access:
    @staticmethod
    def save(path, data:dict,convert=False,**kwargs):
        with open(path,'w') as _file:
            if convert:
                yaml.dump(SettingsConverter.convert(data),_file)
            else:
                yaml.dump(data,_file)
    @staticmethod
    def load(path,**kwargs):
        with open(path,'r') as _file:
            data = yaml.load(_file.read())
        if isinstance(data,type(None)):
            data = {}
        return data
    @staticmethod
    def format_settings_dict(settings):
        return SettingsConverter.convert(settings)

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
    def recursive_convert_list(cls,l):
        out=[]
        for _id,val in enumerate(l):
            if isinstance(val,(type(None),str,complex,float,int,bytes,bool,np.number,np.bool_)):
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
                if isinstance(val,(str,complex,float,int,bytes,bool,np.number,np.bool_)):
                    out_dict[key]=val
                elif isinstance(val,np.ndarray):
                    out_dict[key]={'command':cls.numpy_to_str(val)}
                elif isinstance(val,slice):
                    out_dict[key]={'command': str(val)}
                elif isinstance(val,(CommentedSeq,CommentedMap)):
                    out_dict[key]=val
                elif isinstance(val,(list,tuple)):
                    print(f'key {key} list {val}')
                    out_dict[key]=cls.recursive_convert_list(val)
                    print(f'parsed list {out_dict[key]}')
                elif isinstance(val, (dict,DictNamespace)):
                    out_dict[key]=cls.recoursive_convert_settings(val)
                else:
                    raise ValueError('Cannot convert {} type for settings key {}'.format(type(val),key))
            except Exception as e:
                log.warning("Failed to save key {} and value type {} with error: \n {}".format(key,type(val),e))
                traceback.print_exc()
        return out_dict

    @classmethod
    def convert(cls,settings):
        return cls.recoursive_convert_settings(settings)
