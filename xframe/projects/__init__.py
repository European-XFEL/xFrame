import traceback
import os
import sys
import xframe
import logging
from importlib import util
log = logging.getLogger('root')

projects_to_load = xframe.settings.general.load_projects
def create_init_file_if_non_existing(project_path):
    init_path = os.path.join(project_path,'__init__.py')
    
    if not os.path.exists(init_path):
        with open(init_path, 'w') as fp:
            fp.write('')
    
def import_project(name,path):
    create_init_file_if_non_existing(path)
    py_import_path = f'xframe.projects.{name}'
    spec = util.spec_from_file_location(py_import_path, path+'__init__.py')
    module = util.module_from_spec(spec)
    sys.modules[py_import_path] = module
    spec.loader.exec_module(module)
    globals().update({name:module})

for project_name,project_path in xframe.known_projects.items():
    try:
        if isinstance(projects_to_load,(list,tuple)):
            if project_name in projects_to_load:
                import_project(project_name,project_path)
        else:        
            import_project(project_name,project_path)
    except Exception as e:
        log.error(f'Error during import of project at {project_path}, Traceback:\n'+traceback.format_exc())

        
