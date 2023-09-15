import os
from importlib import util

#install_folder = os.path.abspath(os.path.dirname(__file__)+'/../')+'/'
install_folder = util.find_spec('xframe').submodule_search_locations[0]+'/'
#print(f'xframe is installed at {install_folder}')

default_folder=os.path.expanduser('~/.xframe/')
home_folder=os.path.expanduser('~/.xframe/')
from importlib import util
###Loading User provided config ###
user_config_file = os.path.join(home_folder,'config.py')
if os.path.exists(user_config_file):
    spec = util.spec_from_file_location('file',user_config_file)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
home_folder = os.path.expanduser(home_folder)

lazy_denpendency_injection = True

n_control_workers = 0 #8
max_parallel_processes = 200
RAM = 512*1024**3 # in Byte, This value is only used if psutil is not installed
cache_aware = True
L1_cache = 64 #32 # Cache size in kB
L2_cache = 512 #256 # Cache size in kB
loglevel = 'WARNING' #['WARNING','INFO','DEBUG']

default_project_db_name= 'ProjectDB'
default_experiment_db_name= 'ExperimentDB'
default_project_worker_name = 'ProjectWorker'
default_experiment_worker_name = 'ExperimentWorker'
default_experiment_module_name = 'experiment'

default_settings_regexpr = '[-+]?\d*\.*\d+'
worker_regexpr = '^[A-Za-z0-9].*\.py'
settings_version_key = 'settings_version'

load_projects = 'all' # or list of project names
load_experiments = 'all' # or list of experiment names

project_folders=['projects','install_projects']
experiment_folders=['experiments','install_experiments']
log_file = os.path.join(home_folder,'log.txt')
IO = {"folders":
      {"default_home":'~/.xframe/',
       "home":home_folder,
       'install':install_folder,
       'direct':'{path}',
       'experiments':{'home':'experiments/'},
       'projects':
       {'home':'projects/'},
       'default_experiments':
       {'default_home':'experiments/'},
       'default_projects':
       {'default_home':'projects/'},
       'install_experiments':
       {'install':'experiments/'},
       'install_projects':
       {'install':'projects/'},       
       'settings_install_project':
       {'install_projects':'{project}/settings/{worker}/'},
       'settings_install_experiment':
       {'install_experiments':'{experiment}/settings/'},
       'settings_project':
       {'home':'settings/projects/{project}/{worker}/'},
       'settings_experiment':
       {'home':'settings/experiments/{experiment}/'},
       'settings_default_project':
       {'default_home':'settings/projects/{project}/{worker}/'},
       'settings_default_experiment':
       {'default_home':'settings/experiments/{experiment}/'},
       'settings_direct':
           {'direct':'settings/{worker}/'}
       },
      'files':{
          'config':{
              'name':'config.py',
              'folder':'default_home'
          },
          'log':{
              'name':'log.txt',
              'folder':'home'
          },
          'settings_experiment':{
              'name': '{name}.yaml',
              'folder': 'settings_experiment'
          },
          'settings_default_experiment':{
              'name': '{name}.yaml',
              'folder': 'settings_default_experiment'
          },
          'settings_install_experiment':{
              'name': '{name}.yaml',
              'folder': 'settings_install_experiment'
          },
          'settings_project':{
              'name': '{name}.yaml',
              'folder': 'settings_project'
          },
          'settings_default_project':{
              'name': '{name}.yaml',
              'folder': 'settings_default_project'
          },
          'settings_install_project':{
              'name': '{name}.yaml',
              'folder': 'settings_install_project'
          },
          'settings_direct':{
              'name': '{name}.yaml',
              'folder': 'direct'
          }
      }
    }
