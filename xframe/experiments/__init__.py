import traceback
import os
import sys
import xframe
import logging
from importlib import util
log = logging.getLogger('root')

experiments_to_load = xframe.settings.general.load_experiments
def create_init_file_if_non_existing(experiment_path):
    init_path = os.path.join(experiment_path,'__init__.py')    
    if not os.path.exists(init_path):
        with open(init_path, 'w') as fp:
            fp.write('')
    
def import_experiment(name,path):
    create_init_file_if_non_existing(path)
    py_import_path = f'xframe.experiments.{name}'
    spec = util.spec_from_file_location(py_import_path, path+'__init__.py')
    module = util.module_from_spec(spec)
    sys.modules[py_import_path] = module
    spec.loader.exec_module(module)
    globals().update({name:module})

for experiment_name,experiment_path in xframe.known_experiments.items():
    try:
        if isinstance(experiments_to_load,(list,tuple)):
            if experiment_name in experiments_to_load:
                import_experiment(experiment_name,experiment_path)
        else:        
            import_experiment(experiment_name,experiment_path)
    except Exception as e:
        log.error(f'Error during import of experiment at {experiment_path}, Traceback:\n'+traceback.format_exc())
