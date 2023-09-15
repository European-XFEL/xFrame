import importlib
import os
import sys
import datetime
import traceback
import re
import glob
import shutil
import argparse
import logging

import xframe
#from xframe.library.pythonLibrary import xprint
def xprint(m):
    print(m)
    

def import_settings():
    from xframe import settings
    #xframe.__setattr__('settings',settings)
    
def setup_logging():
    from xframe.settings import general
    from xframe import log
    log = log.setup_custom_logger('root',general.loglevel)
    globals().update({'log':log})

# xframe modules to load at import 
modules_to_import  = ['Multiprocessing','database','control.Control',['library','lib'],'presenters']
def startup_imports():
    for module in modules_to_import:
        try:
            if isinstance(module,(list,tuple)):
                module_name = 'xframe.'+module[0]
                xframe.__setattr__(module[1], importlib.import_module(module_name))
            else:
                splitted_name = module.split('.')
                module_name = 'xframe.'+module
                if len(splitted_name) > 1:                
                    xframe.__setattr__(splitted_name[-1], importlib.import_module(module_name))
                else:
                    xframe.__setattr__(module,importlib.import_module(module_name))
            #log.info('loaded {}'.format(module))
                    
        except Exception as e:
            traceback.print_exc()
            log.error('Caught exeption while loading {} with content: {}'.format(module_name,e))



def create_dependency_dict():
    external_dependency_dict = {
        ##### math stuff #####
        'gsl':[(('gsl_plugin','GSLPlugin','pygsl'),('xframe.lib.math','gsl'),())],
        'flt':[(('flt_plugin','LegendreTransform','flt'),('xframe.lib.math','leg_trf'),())],
        'shtns':[(('shtns_plugin','sh','shtns'),('xframe.lib.math','shtns'),())],
        'persistent_homology':[(('persistent_homology','PersistentHomologyPlugin','None'),('xframe.lib.math','PeakDetector'),())],
        'pysofft':[(('soft_plugin','Soft','pysofft'),('xframe.lib.math','Soft'),())],
        ##### multiprocessing & GPU #####
        'shared_array':[('SharedArray',('xframe.lib.python','sa'),()),
                        ('SharedArray',('xframe.Multiprocessing','sa'),())],
        'opencl':[(('openCL_plugin','OpenClPlugin','pyopencl'),('xframe.Multiprocessing','openCL_plugin'),())],
        'psutil':[(('psutil_plugin','PsutilPlugin','psutil'),('xframe.Multiprocessing','psutil_plugin'),())],
        ###### database plugins ######
        'vtk':[(('vtk_plugin','vtkSaver','vtk'),('xframe.database.database','VTK_saver'),())],
        'pdb':[(('pdb_plugin','ProteinDB','pdb_eda'),('xframe.database.database','PDB_loader'),())],
        'hdf5':[(('hdf5_plugin','HDF5_DB','h5py'),('xframe.database.database','HDF5_access','as_object'),())],
        'yaml':[(('yaml_plugin','YAML_access','ruamel_yaml'),('xframe.database.database','YAML_access'),())],
        'opencv':[(('cv_plugin','CV_Plugin','opencv-python'),('xframe.database.database','OpenCV_access'),()),
                  (('cv_plugin','CV_Plugin','opencv-python'),('xframe.presenters.openCVPresenter','CV_Plugin'),())],
        ###### presenter plugins ######        
        'matplotlib':[(('matplotlib_plugin','matplotlib','matplotlib'),('xframe.presenters.matplotlibPresenter','matplotlib'),('xframe.presenters.matplotlibPresenter.depencency_injection_hook_mpl',))],
        'matplotlib_toolkit':[(('mpl_toolkits_plugin','mpl_toolkits','matplotlib'),('xframe.presenters.matplotlibPresenter','matplotlib'),('xframe.presenters.matplotlibPresenter.dependency_injection_hook_mpl_toolkits',))],
        ###### cmd ######
        'click':[(('click_plugin','ClickPlugin','click'),(False,False,'as_object'),())],
    }
    return external_dependency_dict
    
def _inject_dependency(dependency_name,reload=False):
    log=logging.getLogger('root')
    log.info(f'Dependency injecting {dependency_name}.')
    dependency_list = xframe._external_dependency_lookup.get(dependency_name,False)
    if isinstance(dependency_list,bool):
        log.warning(f'xFrame does not know the dependency {dependency_name}, skipping. Known external dependencies are {xframe._external_dependency_lookup.keys()}')
        return
    ## Failing to import does not generate an error message or raise an error.
    ## If later a routine is called that would require a
    ## depencency it is called against an abstract calss that throws a corresponding not imported error.
    for import_source ,import_destination,injection_hooks in dependency_list:
        if isinstance(import_source,(list,tuple)):
            import_name='xframe.externalLibraries.'+ import_source[0]
            tmp = importlib.import_module(import_name)
            dependency = tmp.__getattribute__(import_source[1])                
        else:
            #import_name='xframe.externalLibraries.'+ import_source
            dependency = importlib.import_module(import_source)
        if reload:
            dependency = importlib.reload(dependency)
        if len(import_destination)==3:
            if import_destination[2]=='as_object':
                dependency = dependency()
        if not isinstance(import_destination[0],bool):
            setattr(eval(import_destination[0]),import_destination[1],dependency)
            if len(injection_hooks)>0:
                for hook in injection_hooks:
                    eval(hook)()
    return dependency
            
# External packages to be dependency injected.
# has to be callde after startup_imports
def dependency_injection():
    if not xframe.settings.general.lazy_denpendency_injection:
        for key in xframe._external_dependency_name_.keys():
            if key != 'pysofft':
                _inject_dependency(key)

def dependency_inject_SOFT():
    # separate since it uses numba and requires a bit of time to import.
    try:
        from xframe.externalLibraries.soft_plugin import Soft
        xframe.lib.math.Soft=Soft
    except Exception as e:
        log.error(f'loading PySOFT dependency failed with error {e}')

        
def setup_default_database():
    xframe.database.default = xframe.database.database.DefaultDB(**xframe.settings.general.dict()['IO'])


def lookup_projexp(path_list):
    _dict = {}
    #print(f'{path_list}')
    for path in path_list:
        try:
            names = next(os.walk(path))[1]
            for name in names:
                if (name[0]!='_') and (name not in _dict):
                    _dict[name] = os.path.join(path,name+'/')
        except StopIteration as e:
            pass
    return _dict
## look for projects
def lookup_projects():
    opt = xframe.settings.general
    db = xframe.database.default

    project_paths = [db.get_path(fname,is_file=False) for fname in opt.project_folders]
    project_dict = lookup_projexp(project_paths)
    if len(project_dict) == 0:
        log.warning('No xFrame projects found at {}. Check the general settings project path option.'.format(project_paths))
    return project_dict    
def setup_projects():
    xframe.known_projects = lookup_projects()
    xframe.__setattr__('projects',importlib.import_module('xframe.projects'))

## look for experiments
def lookup_experiments():
    opt = xframe.settings.general
    db = xframe.database.default
    #log.info('looking for experiments at {}'.format(opt.experiment_paths))    
    experiment_paths = [db.get_path(fname,is_file=False) for fname in opt.experiment_folders]
    experiment_dict = lookup_projexp(experiment_paths)
    if len(experiment_dict) == 0:
        log.warning('No xFrame experiment found at {}. Check the general settings experiment path option.'.format(experiment_paths))
    return experiment_dict    

def setup_experiments():
    xframe.known_experiments = lookup_experiments()
    xframe.__setattr__('experiments',importlib.import_module('xframe.experiments'))


def lookup_workers(project_path):
    from xframe.settings import general as opt
    file_paths = glob.glob(os.path.join(project_path,'*.py'))
    file_names = tuple(os.path.basename(p) for p in file_paths)
    #print(f'worker candidates = {file_names}')
    #print(f'worker regexpr = {opt.worker_regexpr}')
    pattern = re.compile(opt.worker_regexpr)
    worker_names = [os.path.splitext(name)[0] for name in file_names if pattern.match(name)]
    #print(f'workers = {worker_names}')
    return worker_names

def _parse_project_name(name,worker_name=False):
    known_projects = xframe.known_projects
    if not name in known_projects:
        raise AssertionError('Project_Name {} was not found at {}. Known projects are {}'.format(name,[path + name for path in xframe.settings.general.project_paths],known_projects.keys()))
    project_path = known_projects[name]
    if isinstance(worker_name,str):        
        worker_path = os.path.join(project_path,worker_name+'.py')
        worker_exists = os.path.exists(worker_path)
    else:
        worker_exists = False
    if not worker_exists:
        raise AssertionError('Worker {} for Project {} was not found at {}'.format(worker_name,name,worker_path))            
    return project_path,name,worker_name

def _parse_experiment_name(name):
    known_experiments = xframe.known_experiments
    if not name in known_experiments:
        raise AssertionError('Experiment "{}" was not found at {}. Known experiment modules are "{}"'.format(name,[path + name for path in xframe.settings.general.experiment_paths],known_experiments,keys()))
    experiment_path = known_experiments[name]
    return experiment_path,name

def _load_db(name,db_name,opt,target='project'):
    if target=='experiment':
        db_module_pypath = 'xframe.experiments.'+name+'._database_'  
    else:
        db_module_pypath = 'xframe.projects.'+name+'._database_'
        
    try:      
        db = getattr(importlib.import_module(db_module_pypath),db_name)(**opt)
    except (AttributeError, ModuleNotFoundError) as e:
        log.info('Could not instanciate database class {} in project {} use default database instead. Recived the following error during initialization: \n {}'.format(db_name,name,e))
        log.debug(traceback.format_exc())
        db = xframe.database.database.DefaultDB(**opt)
    return db
   

def select_project(name=False,worker_name=False,project_settings=False,ignore_file_not_found=False):
    #sys.path.append(opt.project_paths)
    settings = xframe.settings
    database = xframe.database
    
    proj_settings_name = project_settings
    if isinstance(name,bool):
        project_settings = {}
        project_settings_raw = {}
        project_path = ''
        name = ''
        worker_name = ''
    else:        
        project_path,name,worker_name = _parse_project_name(name,worker_name)

        project_settings,project_settings_raw = database.default.load('settings',project_path = project_path,worker_name = worker_name,settings_file_name=project_settings,ignore_file_not_found=ignore_file_not_found)

        project_settings_raw = project_settings_raw.dict()
    settings._update_settings_on_project_select(project_settings,project_settings_raw)
    #settings.project = project_settings
    #settings.raw_project= project_settings_raw
    if 'IO' in project_settings:
        project_IO_settings=project_settings.dict().pop('IO')
    else:
        project_IO_settings = {}
    project_db = _load_db(name,settings.general.default_project_db_name,project_IO_settings)
    database.project = project_db
    xframe._project_worker_module_name = _get_worker_module_name(name,worker_name)
    
def select_experiment(exp_name=False,exp_settings=False,ignore_file_not_found=False):
    settings = xframe.settings
    database = xframe.database
    controller = xframe.controller
    
    exp_settings_name = exp_settings
    if isinstance(exp_name,str):
        exp_path,exp_name = _parse_experiment_name(exp_name)
        exp_settings,raw_exp_settings=database.default.load('settings',project_path = exp_path,settings_file_name=exp_settings,target='experiment',ignore_file_not_found=ignore_file_not_found)
        raw_exp_settings = raw_exp_settings.dict()
        settings._update_settings_on_experiment_select(exp_settings,raw_exp_settings)
        #settings.experiment = exp_settings
        #settings.raw_experiment = raw_exp_settings
        if 'IO' in exp_settings:
            exp_IO_settings = exp_settings.dict().pop('IO')
        else:
            exp_IO_settings = {}
        exp_db = _load_db(exp_name,settings.general.default_experiment_db_name,exp_IO_settings,target='experiment')
        database.experiment = exp_db
        xframe._experiment_module_name = _get_worker_module_name(exp_name,is_experiment=True)

def import_selected_project(update_worker=True):
    settings = xframe.settings
    database = xframe.database
    controller = xframe.controller    

    if update_worker:
        settings.raw_project = database.default.format_settings(settings.project)

    module_name=xframe._project_worker_module_name
    worker_module_imported = module_name in sys.modules
    worker_instance_exists = isinstance(xframe.project_worker,xframe.interfaces.ProjectWorkerInterface)
    needs_worker_creation = (not worker_module_imported) or (not worker_instance_exists) or update_worker
    #xprint(f'importing {module_name} needs worker update = {needs_worker_creation}')
    if needs_worker_creation:        
        try:
            if module_name in sys.modules:
                #xprint('reloading')
                worker_module = importlib.reload(sys.modules[module_name])
            else:
                #xprint('importing')
                worker_module = importlib.import_module(module_name)
            #xprint('instanciating')
            worker_instance = getattr(worker_module,settings.general.default_project_worker_name)()
            xframe.project_worker =  worker_instance
            controller.project_worker = worker_instance
        except Exception as e:
            log.error(f'Could not import project worker {module_name},\n with error {e}\n {traceback.format_exc()}\n Terminating.')
            sys.exit()
            
def import_selected_experiment(update_worker=True):
    if isinstance(xframe._experiment_module_name,str):
        settings = xframe.settings
        database = xframe.database
        controller = xframe.controller    
        
        if update_worker:
            settings.raw_experiment = database.default.format_settings(settings.experiment)
    
        moduel_name = xframe._experiment_module_name
        module_imported = module_name in sys.modules
        instance_exists = isinstance(xframe.experiment_worker,xframe.interfaces.ExperimentWorkerInterface)
        needs_experiment_creation = (not module_imported) or (not instance_exists) or update_worker
        
        if needs_experiment_creation:        
            try:
                if module_name in sys.modules:
                    exp_module = importlib.reload(sys.modules[module_name])
                else:
                    exp_module = importlib.import_module(module_name)            
                exp_instance = getattr(exp_module,settings.general.default_experiment_worker_name)()
                xframe.experiment_worker =  exp_instance
                controller.experiment_worker = exp_instance
            
            except Exception as e:
                log.error(f'Could not import experiment {moduel_name},\n with error {e}\n {traceback.format_exc()}\n Terminating.')
                sys.exit()

                
def _get_worker_module_name(name,worker,is_experiment=False):
    if is_experiment:
        worker_module_name = 'xframe.experiments.'+name+'.'+xframe.settings.general.default_experiment_module_name
    else:
        worker_module_name = 'xframe.projects.'+name+'.'+worker
    return worker_module_name
    


def select_and_run(project='project', project_worker='project_worker',project_settings='settings_name',exp_name=False,exp_settings=False, update_worker=False,oneshot=False):
    select_project(name = project, worker_name = project_worker, project_settings = project_settings)
    select_experiment(exp_name = exp_name, exp_settings = exp_settings)
    import_selected_project(update_worker=update_worker)
    import_selected_experiment(update_worker=update_worker)
    xframe.lib.python.measureTime(xframe.controller.run)(oneshot=oneshot)

def run(update_worker = False,oneshot=False):
    import_selected_project(update_worker=update_worker)
    import_selected_experiment(update_worker=update_worker)
    xframe.controller.run(oneshot=oneshot)


class DefaultProjectArgparser:
    project_description=''
    def __init__(self):
        self.settings_parser = argparse.ArgumentParser(add_help=False)
        self.settings_parser.add_argument('settings',metavar='SETTINGS_NAME',default=False,type=str,help='Name of settings file to be used')
    def add_parser(self,subparser,project_name,project_path):
        worker_names = lookup_workers(project_path)
        #log.info()
        description = self.project_description
        project_parser = subparser.add_parser(project_name, description=description)
        project_subparser = project_parser.add_subparsers(help='Available Workers',dest = 'worker')
        for worker in worker_names:
            add_worker_parser = getattr(self,'add_'+worker+'_argparser',self.add_default_worker_parser)
            #print(add_worker_parser)
            add_worker_parser(project_subparser,worker)
    
    def add_default_worker_parser(self,subparser,worker):
        subparser.add_parser(worker, description='No description available', help=f'No help available',parents=[self.settings_parser])

    def process_arguments(self,args):
        pass

class DefaultProjectClick:
    project_description = ''
    short_description = ''
    worker_help = {}
    def add_click(self,group,click,project_name,project_path):
        parsed_results= {}
        self.worker_names = lookup_workers(project_path)
        self.project_name = project_name
        description = self.project_description
        add_project_click = getattr(self,'add_'+project_name,self.add_default_project_click)
        worker_group = add_project_click(group,click)
        for worker in self.worker_names:
            try:
                add_worker_parser = getattr(self,'add_worker_'+worker,self.add_default_worker_click)
                add_worker_parser(worker_group,click,worker)
            except Exception as e:
                log.error(e)
    
    def add_default_worker_click(self,group,click,worker_name,help='',short_help='',epilog=''):
        @group.command(worker_name,help=help,short_help=short_help,epilog=epilog)
        @click.argument('settings_name',required=False)
        def worker(settings_name=False,**kwargs):
            log.info(f'start: {worker_name}')
            #xprint(f'selecting worker {worker_name}')
            if settings_name is None:
                #print('no settings provided')
                select_project(name=self.project_name,worker_name=worker_name)
            else:
                #print('settings provided')
                select_project(name=self.project_name,worker_name=worker_name,project_settings=settings_name)
            xframe.library.pythonLibrary.measureTime(run)(oneshot=False)
    def add_default_project_click(self,group,click):
        @group.group(self.project_name,chain=True,help = self.project_description, short_help= self.short_description)
        def project():
            pass
        return project
    def process_arguments(self,args):
        pass

def setup_home(home):
    db = xframe.database.default
    if os.path.expanduser(home) != db.get_path('home',is_file=False):
        if home[-1]!='/':
            home+='/'
        new_home_exists = os.path.exists(os.path.expanduser(home))
        if not new_home_exists:
            response = 'a'
            while response not in 'yYnN':                
                response = input(f'Folder "{home}" does not exist. Do you want to create the new folder [y/n]? ')
            if response in 'nN':
                xframe.lib.python.xprint('abborting!')
                sys.exit()
        log.info(f'Changing home from {db.get_path("home",is_file=False)} to {home}')
        settings_entry = f'home_folder = "{os.path.expanduser(home)}"\n'
        
        #Creating config file in .xframe pointing to the new home directory
        if os.path.exists(db.get_path('config')):
            lines = db.load('config',as_text= True)
            home_folder_found  =False
            for _id,line in enumerate(lines):
                if line[:11]=='home_folder':
                    lines[_id]= settings_entry
                    home_folder_found = True
                    break
            if not home_folder_found:
                lines = [settings_entry]+lines
            config= lines
        else:
            config = [settings_entry]        
        db.save('config',config)
        xframe.settings.general.home_folder = home
        xframe.settings.general.IO.folders.home=home
        db.update_folders_and_files(**xframe.settings.general.IO.dict())        

    log.info('symlinking projects and experiments from install folder to home.')
    db.create_folders('home')
    distributed_projects = lookup_projexp([db.get_path('install_projects',is_file=False)])
    distributed_experiments = lookup_projexp([db.get_path('install_experiments',is_file=False)])
    log.info(f'distributed_projects = {distributed_projects}')
    for project,install_path in distributed_projects.items():
        dest = os.path.join(db.get_path('projects',is_file=False),project)
        log.info(f'source = {install_path} target = {dest}')
        db.create_symlink(install_path,dest,is_file=False)        
    for experiment,install_path in distributed_experiments.items():
        dest = os.path.join(db.get_path('experiments',is_file=False),experiment)
        log.info(f'source = {install_path} target = {dest}')
        db.create_symlink(install_path,dest,is_file=False)
    #db.create_symlink('experiments_install','experiments',is_file=False)
    #db.create_symlink('projects_install','projects',is_file=False)
    log.info('symlinking default settings.')
    for project,project_path in distributed_projects.items():
        for worker in lookup_workers(project_path):            
            project_path_modifier = {'project':project,'worker':worker}
            db.create_folders('settings_project',path_modifiers=project_path_modifier)
            settings_install_path = db.get_path('settings_install_project',path_modifiers=project_path_modifier,is_file=False)
            home_settings_path = db.get_path('settings_project',path_modifiers=project_path_modifier,is_file=False)
            default_settings_files = glob.glob(os.path.join(settings_install_path,'default_*.yaml'))
            for default in default_settings_files:
                target_file = os.path.join(home_settings_path,os.path.basename(default))
                db.create_symlink(default,target_file)
                log.info(f'source = {default} target = {target_file}')
    log.info('copy tutorial settings')
    for project,project_path in distributed_projects.items():
        for worker in lookup_workers(project_path):            
            project_path_modifier = {'project':project,'worker':worker}
            db.create_folders('settings_project',path_modifiers=project_path_modifier)
            settings_install_path = db.get_path('settings_install_project',path_modifiers=project_path_modifier,is_file=False)
            home_settings_path = db.get_path('settings_project',path_modifiers=project_path_modifier,is_file=False)
            tutorial_settings_files = glob.glob(os.path.join(settings_install_path,'tutorial*.yaml'))
            for tutorial in tutorial_settings_files:
                target_file = os.path.join(home_settings_path,os.path.basename(tutorial))
                db.copy(tutorial,target_file)
                log.info(f'source = {tutorial} target = {target_file}')
                #print(f'source = {default} target = {target_file}')
    for exp,exp_path in xframe.known_experiments.items():
        xframe.database.default.create_folders('user_experiment_settings',path_modifiers={'experiment':exp})

        exp_modifier = {'experiment':exp}

        db.create_folders('settings_experiment',path_modifiers=exp_modifier)
        settings_install_path = db.get_path('settings_install_experiment',path_modifiers=exp_modifier,is_file=False)
        home_settings_path = db.get_path('settings_experiment',path_modifiers=exp_modifier,is_file=False)
        
        default_settings_files = glob.glob(os.path.join(settings_install_path,'default_*.yaml'))
        for default in default_settings_files:
            target_file = os.path.join(home_settings_path,os.path.basename(default))
            log.info(f'source = {default} target = {target_file}')
            db.create_symlink(default,target_file)
    xprint(f'Created xframe home at: {os.path.expanduser(home)}')

    


        
