import h5py as h5
import numpy as np
import logging
import traceback
from xframe.library.gridLibrary import NestedArray
from xframe.library.pythonLibrary import FTGridPair
from xframe.library.pythonLibrary import DictNamespace
from xframe.database.interfaces import HDF5Interface
log=logging.getLogger('root')

class HDF5_DB(HDF5Interface):
    load_custom_types=False
    save_custom_types=False
    def __init__(self):
        save_custom_types={
            FTGridPair.__name__:HDF5_DB.save_FTGridPair,
            NestedArray.__name__:HDF5_DB.save_NestedArray,
            DictNamespace.__name__:HDF5_DB.save_DictNamespace
        }
        HDF5_DB.save_custom_types = save_custom_types

        load_custom_types={
            NestedArray.__name__:HDF5_DB.load_NestedArray
        }
        HDF5_DB.load_custom_types = load_custom_types
        
    @staticmethod
    def save(path,value_dict,**kwargs):        
        if isinstance(path,str):            
            with h5.File(path, 'w') as h5file:
                HDF5_DB.recursively_save_dict_to_group(h5file,'/',value_dict)
        else:
            raise FileNotFoundError('Path Error, Abort loading "{}".'.format(path))
            
    @staticmethod
    def load(path,as_h5_object=False,**kwargs):
        try:
            if isinstance(path,str):
                if not as_h5_object:
                    with h5.File(path, 'r') as h5file:
                        data=HDF5_DB.recursively_load_dict_from_group(h5file, '/')
                else:
                    data = h5.File(path, 'r')
            else:
                raise FileNotFoundError('Path Error, Abort loading "{}".'.format(path))
        except Exception as e:
            raise e
        return data
    
    @staticmethod
    def recursively_save_dict_to_group(h5_file, path, dic):
        custom_save_routines = HDF5_DB.save_custom_types
        for key,item in dic.items():
            key=str(key)
            #log.info('key={}'.format(key))
            #log.info('path={}'.format(path))
            #log.info('h5-file={}'.format(h5_file.keys()))
            try:                
                if isinstance(item,(complex,float,int,bytes,bool,np.number,np.bool_)):
                    h5_file[path].create_dataset(key,data=item)
                elif isinstance(item,(str,np.character)):
                    h5_file[path].create_dataset(key,data=item.encode('utf-8'))
                    h5_file[path+'/'+key].attrs['type']='str'
                elif isinstance(item,np.ndarray):
                    HDF5_DB.save_numpy_array(h5_file,path,key,item)            
                elif isinstance(item,h5.VirtualLayout):
                    h5_file[path].create_virtual_dataset(key, item)
                elif isinstance(item,(list,tuple)):
                    #log.info(f'Saving list or tuple !!!! at {path+"/"+key}')
                    h5_file[path].create_group(key)
                    if isinstance(item,list):
                        h5_file[path+'/'+key].attrs['type']='list'
                    else:
                        h5_file[path+'/'+key].attrs['type']='tuple'                        
                    HDF5_DB.recursively_save_dict_to_group(h5_file, path + key + '/', {str(i):elem for i,elem in enumerate(item)})     
                elif isinstance(item, dict):
                    HDF5_DB._save_dict(h5_file,path,key,item)
                elif type(item).__name__ in custom_save_routines:
                    custom_save_routines[type(item).__name__](h5_file,path,key,item)
                else:
                    raise ValueError('Cannot save {} type for key {}'.format(type(item),key))
            except Exception as e:
                log.error("Failed to save key {} and item type {} with error: \n {}".format(key,type(item),e))
    
    @staticmethod
    def recursively_load_dict_from_group(h5_file, path):
        ans = {}
        custom_load_routines=HDF5_DB.load_custom_types
        for key, item in h5_file[path].items():
            item_type = item.attrs.get('type',False)
            if isinstance(item, h5._hl.dataset.Dataset):
                if item_type == 'str':
                    ans[key] = item[()].decode('utf-8')
                elif item_type in custom_load_routines:
                    ans[key] = custom_load_routines[item_type](item)
                else:                   
                    ans[key] = item[()]
            elif isinstance(item, h5._hl.group.Group):
                if item_type=='list':
                    ans[key] = HDF5_DB._load_list(h5_file,path,key)
                elif item_type=='tuple':
                    ans[key] = HDF5_DB._load_tuple(h5_file,path,key)
                else:
                    ans[key] = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        return ans
    
    @staticmethod
    def save_numpy_array(h5_file,path,key,array,meta_dict={}):
        dtype=array.dtype
        if dtype==np.complex_:
            h5_file[path].create_dataset(key,data=array.astype('<c16'))
        elif dtype == np.bool_ :
                h5_file[path].create_dataset(key,data=array.astype(int))
        elif 'str' in dtype.name:
            h5_file[path].create_dataset(key,data=array.astype('S'))
        else:
            h5_file[path].create_dataset(key,data=array)
            for attrib in meta_dict:
                h5_file[path+key+'/'].attrs.create(attrib,meta_dict[attrib])


    @staticmethod
    def _save_dict(h5_file,path,key,item):        
        h5_file[path].create_group(key)
        #log.info('path key slash = {}'.format(path+key+'/') )
        HDF5_DB.recursively_save_dict_to_group(h5_file, path + key + '/', item)
    @staticmethod
    def _load_tuple(h5_file,path,key):
        data = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        n_elements = len(data)
        return tuple(data[str(i)] for i in range(n_elements))
    @staticmethod
    def _load_list(h5_file,path,key):
        data = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        n_elements = len(data)        
        return list(data[str(i)] for i in range(n_elements))

    ######## Custom Datattypes #######
    @staticmethod
    def save_NestedArray(h5_file,path,key,item):
        HDF5_DB.save_numpy_array(h5_file,path,key,item.array,meta_dict={'type':NestedArray.__name__,'n_ndim':item.n_ndim})
    @staticmethod
    def save_FTGridPair(h5_file,path,key,item):
        h5_file[path].create_group(key)
        new_item={'real_grid':item.realGrid,'reciprocal_grid':item.reciprocalGrid}
        HDF5_DB.recursively_save_dict_to_group(h5_file, path + key + '/', new_item)
    @staticmethod
    def save_DictNamespace(h5_file,path,key,item):
        HDF5_DB._save_dict(h5_file,path,key,item)
    @staticmethod
    def load_NestedArray(item):
        return NestedArray(item[()],item.attrs['n_ndim'])
