import numpy as np
import sys
import os
from os import path as op
import logging
import traceback
#import glob
import struct
import re
import glob
import shutil
from importlib import util
pjoin = os.path.join
pexists = os.path.exists

from xframe.settings import general as xframe_opt

# Custom data types that may need to be saved #
from xframe.library.gridLibrary import NestedArray
from xframe.library.pythonLibrary import FTGridPair,DictNamespace

# Interfaces that the default DB should satisfy #
from xframe.interfaces import DatabaseInterface

# Interfaces that will be replaced upon depencency injection of the external plugins that satisfy them. #
from xframe.database.interfaces import VTKDependency
from xframe.database.interfaces import PDBDependency
from xframe.database.interfaces import HDF5Dependency
from xframe.database.interfaces import YAMLDependency
from xframe.database.interfaces import OpenCVDependency
from xframe import Multiprocessing 

# dependency injection attributes #
# Their correct implementations will be inserted by xframe/__init__.py on runtime#
VTK_saver = VTKDependency
PDB_loader = PDBDependency
HDF5_access = HDF5Dependency
YAML_access = YAMLDependency
OpenCV_access = OpenCVDependency

log=logging.getLogger('root')

class FileAccess():
    '''
    Class that handles folders and file paths that will be used to access data.
    It's main function is to read in folder and file paths form the settings file
    and provide access to them via the 'get_path' method
    '''
    def __init__(self,folders={},files={}):
        self.folders=self.parse_folders(folders)
        self.files=files

    def set_folders_and_files(self,folders={},files={}):
        self.folders=self.parse_folders(folders)
        self.files=files
    def update_folders_and_files(self,folders={},files={}):
        self.folders.update(self.parse_folders(folders))
        self.files.update(files)
    def get_path(self,name,path_modifiers={},is_file=True,**kwargs):     
        try:
            if is_file:
                #log.info('requested path name = {}'.format(name))
                _file=self.files[name]
                location=self.folders[_file['folder']]
                file_name=_file['name']
                #log.info('location = {}'.format(location))
                #log.info('file_name = {}'.format(file_name))
                path=location+file_name
            else:
                path=self.folders[name]

            path=self.modify_path(path,modifiers=path_modifiers)
            path = os.path.expanduser(path)
        except KeyError as e:
            #traceback.print_exc()
            #log.info()
            #log.error(e)
            log.debug('No path specified in settings for name "{}"'.format(name))
            path=name
            
        return path
    def create_path_if_nonexistent(self,path):
        dirname=os.path.dirname(path)
        if not (os.path.exists(dirname) or dirname==''):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
    def modify_path(self,path,modifiers={}):
        try:
             path=path.format(**modifiers)
        except KeyError as e:
            log.error(e)
#            n_modifiers = min( (path.count('{'),path.count('}')) )
            log.error('given file path: {} contains the modifiers {}, some of them are wrong.'.format(path,modifiers))
            raise e
        return path

    def parse_folders(self,folder_dict):
        known_folders={}
        for i in range(len(folder_dict)):
            for key,item in folder_dict.items():
                if isinstance(item,str):
                    known_folders[key]=item
                elif isinstance(item,(dict,DictNamespace)):
                    path = self.recursive_folders(folder_dict[key],known_folders)
                    if isinstance(path,str):
                        known_folders[key]=path
        return known_folders
    def recursive_folders(self,folder_dict,known_folders):
        for key in folder_dict:
            if key in known_folders:            
                path=known_folders[key]
                if isinstance(folder_dict[key],(dict,DictNamespace)):
                    path_result=self.recursive_folders(folder_dict[key],known_folders)                
                    if isinstance(path_result,str):                    
                        path+=path_result
                    else:
                        path=False
                elif isinstance(folder_dict[key],str):
                    path+=folder_dict[key]
            else:
                path = False
        return path

    def create_symlink(self,data_path,sym_path,is_file = True,path_modifiers = {}):
        data_path = self.get_path(data_path,is_file=is_file,path_modifiers = path_modifiers)
        sym_path = self.get_path(sym_path,is_file=is_file,path_modifiers = path_modifiers)
        data_path_exists = os.path.exists(data_path)
        sym_path_exists = os.path.exists(sym_path)
        #log.info('data exists = {} path = {}'.format(data_path_exists,data_path))
        #log.info('sym path = {}'.format(sym_path))
        if not sym_path_exists:
            self.create_path_if_nonexistent(sym_path)
        if data_path_exists and (('/' in sym_path) or ('\\' in sym_path)):
            if sym_path_exists:
                os.remove(sym_path)
            os.symlink(data_path,sym_path)

    def copy(self,source,destination,path_modifiers = {}):
        source = self.get_path(source,path_modifiers=path_modifiers)
        destination = self.get_path(destination,path_modifiers=path_modifiers)
        shutil.copyfile(source,destination)
        
    def create_folders(self,path,path_modifiers={}):
        path = self.get_path(path,is_file=False,path_modifiers = path_modifiers)
        self.create_path_if_nonexistent(path)
class DefaultDB(FileAccess,DatabaseInterface):
    '''
    Modular loading and saving Class which can be inherited from to generate customiced loading and saving routines.
    Main methods are 'save' and 'load'.
    
    If load or save are called with the argument <name>(and sone data for save) a 4 step process starts.
    1. Try to find a method in this instance which is called load_<name>, save_<name> and execute it.
    2. else lookup if there is a stored data path to the given <name> if yes retreive that path
    3. else assume <name> itself is a complete file path.
    4. Check if there is an available plugin for the given filetype, if yes call it.

    If one inherits from default_DB point 1. in above sequence lets you easily do some pre or postprocessing if the data
    you want to retrieve is not completely in the shape you want it to be.
    
    A new access method can be added to the access_methods attribute by specifiing the Abstract class and the filetype which it is supposed to handle.

    Note that no direct dependencies on external libraries should never be placed here. Use dependency injection for that.
    See xframe/__init__.py , xframe/externalLibraries/ and xframe/database/interfaces.py
    '''
    
    def __init__(self,**folders_files):
        super().__init__(**folders_files)
        self.access_method_splitter = '://'
        self.data_type_splitter = '.'
        self.default_access_method_str = 'file'
        self.default_data_type_str = 'default'
        self.access_methods={
            'file://':{
                'h5':'HDF5_access',
                'vtk':'VTK_saver',
                'vts':'VTK_saver',
                'vtr':'VTK_saver',
                'npy':'NumpyAccess',
                'raw':'BinaryAccess',
                'txt':'TextAccess',
                'bash':'TextAccess',
                'sh':'TextAccess',
                'zsh':'TextAccess',
                'fish':'TextAccess',
                'yaml':'YAML_access',
                'matplotlib':'MatplotlibAccess',
                'cv':'OpenCV_access',
                'py':'PythonAccess',
                self.default_data_type_str:'TextAccess'
            },
            'pdb://':{
                self.default_data_type_str:'PDB_loader'
            }
        }    
    def load(self,name,**kwargs):        
        try:
            loader = self.get_method(name,prefix = 'load_')
            path = name
            file_options=self.files.get(path,{}).get('options',{})
            if isinstance(loader,bool) or kwargs.get('skip_custom_methods',False):
                #log.info(name)
                path = self.get_path(name,**kwargs)
                #log.info(path)
                if isinstance(path,bool):
                    path = name
                loader = self.get_db(path).load
                path = self.remove_access_method_from_path(path)
            #log.debug(loader)
            #log.debug(path)
            data = loader(path,**{**kwargs,**file_options})
        except Exception as e:
            if not isinstance(e,FileNotFoundError):
                traceback.print_exc()
            if not (path in locals()):
                path = "not defined. Error during path definition."
            if not (loader in locals()):
                loader = "not defined. Could not find loader."            
            log.info('Failed to load for name: {} \n path: {}\n loader: {}\n with error: {}'.format(name,path,loader,e))
            raise e
        return data
        
    def save(self,name,data,**kwargs):
        try:            
            saver = self.get_method(name,prefix = 'save_')
            path = name
            file_options=self.files.get(path,{}).get('options',{})
            if isinstance(saver,bool) or kwargs.get('skip_custom_methods',False):
                path = self.get_path(name,**kwargs)
                if isinstance(path,bool):
                    path = name
                saver = self.get_db(path).save
                path = self.remove_access_method_from_path(path)
                self.create_path_if_nonexistent(path)
            log.debug('saving "{}" to path:  {}'.format(name,path))
            saver(path,data,**{**kwargs,**file_options})
        except Exception as e:
            traceback.print_exc()
            log.error(e)
            log.error('saving for "{}" failed.'.format(name))

    def load_direct(self,name,**kwargs):
        return self.load(name,skip_custom_methods=True,**kwargs)
    def save_direct(self,name,data,**kwargs):
        self.save(name,data,skip_custom_methods=True,**kwargs)

        
    def remove_access_method_from_path(self,path):
        access_method_split = path.split(self.access_method_splitter)
        if len(access_method_split)>1:
            new_path = access_method_split[-1]
        else:
            new_path = path
        return new_path
        
    def get_db(self,path):
        a_split = self.access_method_splitter
        access_method_split = path.split(a_split)
        if len(access_method_split)>1:
            access_method = access_method_split[0]+a_split
        else:
            access_method = self.default_access_method_str+a_split

        d_split = self.data_type_splitter
        name = os.path.basename(path)
        if name[0]==self.data_type_splitter:
            name=name[1:]
        data_type_split = name.split(d_split)
        if len(data_type_split)>1:
            data_type = data_type_split[-1]
        else:
            data_type = self.default_data_type_str            
            
        db = globals()[self.access_methods[access_method][data_type]]
        #log.info(access_method_split)
        #log.info(data_type_split)
        #log.info('{},{}'.format(access_method,data_type))
        #log.info(db)
        return db
    
    def get_method(self,name,prefix=''):
        loader = False
        try:
            loader=getattr(self,prefix+name)
        except AttributeError as e:
            pass
            #log.info('No db method found for "{}". Use default.'.format(prefix+name))
        return loader

    def recursive_command_execution(self,settings_dict,parent_dict,parent_key):
        for key in settings_dict:
            if key == 'command':
                parent_dict[parent_key]=eval(settings_dict[key])
            elif isinstance(settings_dict[key],dict):
                self.recursive_command_execution(settings_dict[key],settings_dict,key)

    def assert_path_exists(self,path):
        if not os.path.exists(path):
            raise FileNotFoundError('The file "{}" does not exist.'.format(path))
        
    def load_settings(self,name,project_path='',worker_name='',settings_file_name=False,target='project',direct_path = False,ignore_file_not_found=False):
        loader = globals()[self.access_methods['file://']['yaml']].load
        dir_name = os.path.basename(os.path.dirname(project_path))
        settings_folders = [
            self.get_path('settings_'+target,path_modifiers={target:dir_name,'worker':worker_name},is_file=False),
            self.get_path('settings_direct',path_modifiers={'path':project_path,'worker':worker_name},is_file=False),
            self.get_path('settings_default_'+target,path_modifiers={target:dir_name,'worker':worker_name},is_file=False),
            self.get_path('settings_install_'+target,path_modifiers={target:dir_name,'worker':worker_name},is_file=False),      
        ]
        s_load = SettingsLoader(loader,settings_folders,ignore_file_not_found=ignore_file_not_found)
        settings,defaults,loader_sub_defaults = s_load.load_settings_files(project_path,worker_name,settings_file_name,direct_path = direct_path)
        s_parser = SettingsParser(loader_sub_defaults)
        settings_out,raw_out = s_parser.parse(settings,defaults)
        return settings_out,raw_out

    def format_settings(self,settings):
        return YAML_access.format_settings_dict(settings)
    def update_settings(self,opt):
        if isinstance(opt,DictNamespace):
            opt = opt.dict()
        self.update_folders_and_files(**opt.get('IO',{}))


# Simple acces Access classes #
class MatplotlibAccess:
    @staticmethod
    def save(path,data,dpi=300,bbox_inches='tight',_format='png',**kwargs):
        path_with_correct_filetype = op.splitext(path)[0]+'.'+_format
        data.savefig(path_with_correct_filetype,dpi=dpi,format=_format,bbox_inches=bbox_inches)
    @staticmethod
    def load(path,**kwargs):
        e = NotImplementedError('loading with the Matplotlib tool is not possible.')
        raise e        

class NumpyAccess:
    @staticmethod
    def save(path,data,**kwargs):
        np.save(path,data)
    @staticmethod
    def load(path,**kwargs):
        return np.load(path)
class BinaryAccess:
    @staticmethod
    def save(path,data,**kwargs):
        endian = kwargs.get('endian','<')            
        fmt=endian+str(data.size)+data.dtype.char            
        bin_struct = struct.pack(fmt, *data.flatten()) 
        with open( path, "wb") as file:
            file.write( bin_struct )
    @staticmethod
    def load(path,**kwargs):
        with open( path, "rb") as file:
            data_bin = file.read()
        if len(data_bin)!=0:
            dtype = np.dtype(kwargs.get('dtype',np.float32))
            endian = kwargs.get('endian','<')            
            length = int(len(data_bin)//dtype.itemsize)
            fmt=endian+str(length)+dtype.char
            data=np.asarray(struct.unpack(fmt, data_bin),dtype = dtype)
        else:
            data = numpy([])
        return data

class TextAccess:
    @staticmethod
    def save(path,data,**kwargs):
        with open(path,'wt') as txt_file:
            if isinstance(data,str):
                txt_file.write(data)
            elif isinstance(data,(list,tuple)):
                txt_file.writelines(data)
            
    @staticmethod
    def load(path,**kwargs):
        with open(path,'rt') as txt_file:
            text = txt_file.readlines()
        return text
class PythonAccess:
    @staticmethod
    def load(path,as_text=False,**kwargs):
        if as_text:
            module = TextAccess.load(path,**kwargs)
        else:
            spec = util.spec_from_file_location('file',general.settings_file)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module    
    @staticmethod
    def save(path,data,**kwargs):
        if isinstance(data,(str,list,tuple)):
            TextAccess.save(path,data,**kwargs)
        else:
            raise NotImplementedError('Saving of python modules as .py files is not supported yet.')

    
# Settings management #
class SettingsLoader:
    def __init__(self,file_loader,settings_folders,ignore_file_not_found=False):
        self.load_file = file_loader
        self.settings_folders = settings_folders
        self.ignore_file_not_found = ignore_file_not_found
        
    def get_settings_path(self,folders,settings_file_name):
        settings_paths = [pjoin(f+settings_file_name+'.yaml') for f in folders]
        settings_path = False
        for folder in folders:
            path = pjoin(folder,settings_file_name+'.yaml')
            if pexists(path):
                settings_path = path                
                break
        
        if isinstance(settings_path,bool):
            message = f'Non of the possible possible settings files exist! They are {settings_paths}'
            if self.ignore_file_not_found:
                message+='Try to start with unchanged default settings.'
            else:
                raise FileNotFoundError(message)
        return settings_path,folder
    
    def get_default_settings_path(self,folder,input_version=False,name = '',warn = False):
        regexpr = xframe_opt.default_settings_regexpr
        if len(name)>0:
            search_expression = pjoin(folder,'default_'+name+'_'+'[0-9]*.yaml')
        else:
            search_expression = pjoin(folder,'default_[0-9]*.yaml')
        default_paths = sorted(glob.glob(search_expression))
        version_number_dict = {re.search(regexpr,p):p for p in default_paths}
        if len(default_paths)>0:
            latest_path = default_paths[-1]
            default_path = latest_path
            if not isinstance(input_version,bool):
                if str(input_version) in version_number_dict:
                    default_path = version_number_dict[str(input_version)]
                elif warn:
                    log.warning(f'Default settings for specified settings_version {input_version} not found. Try to use latest known default file at: {latest_path}. Worker name {worker_name}')
        else:
            if warn:
                log.info(f'No defaults found at {search_expression} . Trying to continue with empty default.')
            default_path = False
        return default_path

    def generate_load_sub_defaults(self,settings_version):
        folder = str(self.selected_folder)        
        def load(name):
            path = self.get_default_settings_path(folder,input_version=settings_version,name = name)
            if isinstance(path,str):
                return self.load(path)
            else: 
                return {}
        return load

    def load_settings_files(self,path,worker_name,settings_file_name,target='project',direct_path = False,fail_silent=False):
        if not isinstance(direct_path,str):
            settings_folders = self.settings_folders
            if isinstance(settings_file_name,str):
                settings_path,folder_path = self.get_settings_path(settings_folders,settings_file_name)                
            else:
                settings_path = False
                folder_path=self.settings_folders[-1]
            self.selected_folder=folder_path            
            if isinstance(settings_path,str):
                settings = self.load_file(settings_path)
            else:
                settings = {}
            settings_version = settings.get(xframe_opt.settings_version_key,False)
            default_path = self.get_default_settings_path(settings_folders[0],input_version=settings_version,warn = True)
            if isinstance(default_path,str):            
                default_settings = self.load_file(default_path)
            else:
                default_settings = {}
            self.default_settings_path = default_path
            self.settings_path = settings_path
            if not isinstance(default_path,bool):
                default_version = re.search(xframe_opt.default_settings_regexpr,os.path.basename(default_path)).group()
            else:
                default_version = False
        else:
            settings = self.load_file(direct_path)
            self.selected_folder=os.path.dirname(direct_path)            
            default_settings = {}
            default_version = False
            settings_path = direct_path
            default_path = ''
        load_sub_defaults = self.generate_load_sub_defaults(default_version)
        settings['_settings_path']=settings_path
        settings['_default_settings_path']=default_path
        return settings,default_settings,load_sub_defaults

class SettingsParser:
    def __init__(self,load):
        self.load = load
        self.parse_order = ['_only_if','_copy','_if']

    def recursive_command_execution(self,settings_dict,parent_dict,parent_key):
        for key in settings_dict:
            if key == 'command':
                #log.info(settings_dict[key])
                parent_dict[parent_key]=eval(settings_dict[key])
            elif isinstance(settings_dict[key],dict):
                self.recursive_command_execution(settings_dict[key],settings_dict,key)

    
    def parse_path(self,current_path,relative_path):
        '''
        processes relative path definition at current place in defaults file.
        '''
        if relative_path[0]!="/":
            if len(current_path) >0:
                tmp_path = relative_path.split('../')
                backwards_index = -len(tmp_path)+1
                if backwards_index ==0:
                    backwards_index = None
                path = current_path[:backwards_index]+tmp_path[-1].split('/')
            else:
                path = [relative_path]
        else:
            path = relative_path[1:].split('/')
        return path

    def get_value(self,dataset,current_path,relative_path):
        '''
        get value in out at position given as a relative path.
        '''
        keys = self.parse_path(current_path,relative_path)
        #print('get path dataset: ',dataset)
        #print('get path current: ',current_path)
        #print('get path keys: ', keys)
        value = dataset[keys[0]]
        for key in keys[1:]:
            value = value[key]
        return value
    
    def set_dict_entry(self,dataset,value,current_path,relative_path):
        '''
        get value in out at position given as a relative path.
        '''
        keys = self.parse_path(current_path,relative_path)
        #print('get path dataset: ',dataset)
        #print('get path current: ',current_path)
        #print('get path keys: ', keys)
        sub_dict = dataset[keys[0]]
        for key in keys[1:-1]:
            sub_dict = sub_dict[key]
        sub_dict[keys[-1]]=value
        
    def get_default_value(self,out,current_path,sub_default):
        '''
        In the default settings file get value for an entry that has no sub settings.
        '''
        default_value = sub_default['_value']
        #log.info(f'default value = {default_value}')
        if isinstance(default_value,(dict,DictNamespace)):
            if '_copy' in default_value:
                opt = default_value['_copy']
                relative_path = opt
                value=self.get_value(out,current_path,relative_path)
            #mode,opt = list(default_value.items())[0] 
            #if mode == '_copy':
            #    relative_path = opt
            #    value=self.get_value(out,current_path,relative_path)
            else:
                return default_value
                #log.error(f'Wrong format in default entry {current_path+[sub_default.dict()]} under _value only "command:" entry is allowed as sub directory but it is {default_value.dict()}')
        else:
            value = default_value
        return value

                
    def check_values(self,*args,**kwargs):
        '''
        future option to check data types against specification in defaults file.
        '''
        return True
    

    ## dynamic default modifiers ##
    def _if(self,out,default,sub_default,current_path):
        value_path=sub_default['_if']['x']
        #log.info(f' current_path = {current_path[:-1]}, \n value_path = {value_path} ')
        x = self.get_value(out,current_path[:-1],value_path)
        #log.info(f'value in if is x = " {x} "')
        conditions = sub_default['_if']['condition']
        if not isinstance(conditions,list):
            conditions = [conditions]
        true_index = len(conditions)
        for _id,c in enumerate(conditions):
            if eval(c):
                true_index = _id
                break
        #print('true index = ',true_index)
        value = sub_default['_if']['values'][true_index]
        
        return {"_value":value},False
        
    def _only_if(self,out,default,sub_default,current_path):
        value_path=sub_default['_only_if']['x']
        x = self.get_value(out,current_path[:-1],value_path)
        condition = sub_default['_only_if']['condition']
        skip = not bool(eval(condition))
        return sub_default,skip
    
    def _copy(self,out,default,sub_default,current_path):
        value_path = sub_default.pop('_copy')
        #print('copy value path: ',value_path)
        try:
            value = self.get_value(default,current_path[:-1],value_path)
        except Exception as e:
            log.error(f'current path {current_path}, value path = {value_path}, \n sub_default = {sub_default},\n default = {default}')
            raise e
        self.set_dict_entry(default,value,current_path,f'../{current_path[-1]}')
        #log.info(f'copyed part: \n \n {value}')
        #d,skip = self.parse_default_sketch(out,default,value,current_path)
        #log.info(f'processed copyed part: \n \n {d}')
        return value,False


    ## parse defaults ##
    def parse_default_sketch(self,out,default,sub_default_sketch,current_path):
        skip = False
        for method_key in self.parse_order:
            try:
                #log.info(f'method key: {method_key}')
                #log.info(f'sketch: {sub_default_sketch}')
                if method_key in sub_default_sketch:
                    #log.info(f'found: {method_key}')
                    try:
                        method = self.__getattribute__(method_key)
                    except AttributeError:
                        continue
                    sub_default_sketch,skip = method(out,default,sub_default_sketch,current_path)
                    if skip:
                        break
            except Exception as e:
                log.error(f'error in step: {method_key}, error: {e}, Traceback:\n')
                traceback.print_exc()
        return sub_default_sketch,skip
    def apply_defaults(self,data,current_path=False,sub_data = False):
        if (not isinstance(current_path,list)) or (not isinstance(sub_data,list)):
            sub_data=data
            current_path = []
            
        defaults,settings,out = sub_data
        global_out = data[-1]
        global_default = data[0]
        #print('def: ' ,defaults)
        if '_import' in defaults:
            import_name = defaults.pop('_import')
            imported_defaults = DictNamespace.dict_to_dictnamespace(self.load(import_name))
            defaults.update(import_defaults)
        if not isinstance(defaults,(dict,DictNamespace)):
            log.info(defaults)
        for key,d in defaults.items():
            try:
                d,skip = self.parse_default_sketch(global_out,global_default,d,current_path+[key])
                if skip or (key[0]=='_'):
                    continue
                exists = key in settings
                d_contains_sub_settings = ('_value' not in d)
                if exists:
                    value = settings[key]
                    s_contains_sub_settings = isinstance(value,(dict,DictNamespace))
                    if s_contains_sub_settings and d_contains_sub_settings:
                        self.apply_defaults(data,current_path = current_path + [key],sub_data = [d,settings[key],out[key]])
                    elif (not s_contains_sub_settings) and (not d_contains_sub_settings):
                        value_valid = self.check_values(d,value)
                        if not value_valid:
                            out[key]=d['_value']
                else:
                    if d_contains_sub_settings:
                        out[key]={}
                        self.apply_defaults(data,current_path = current_path + [key],sub_data = [d,{},out[key]])
                    else:
                        out[key] = self.get_default_value(global_out,current_path,d)
            except Exception as e:
                log.error(f'Error in Application of default settings at in path {current_path} at key {key}')
                traceback.print_exc()
                raise e
        return out
    
    def parse(self,settings_dict,default_settings_dict):
        raw_settings = DictNamespace.dict_to_dictnamespace(settings_dict)
        raw_defaults = DictNamespace.dict_to_dictnamespace(default_settings_dict)
        self.recursive_command_execution(settings_dict,{'value':0},'value')
        self.recursive_command_execution(default_settings_dict,{'value':0},'value')
        settings = DictNamespace.dict_to_dictnamespace(settings_dict)
        default_settings = DictNamespace.dict_to_dictnamespace(default_settings_dict)
        settings_out = settings.copy()
        settings_out = self.apply_defaults([default_settings,settings,settings_out])
        raw_out = raw_settings.copy() 
        raw_out = self.apply_defaults([raw_defaults,raw_settings,raw_out])
        return settings_out,raw_out
            
