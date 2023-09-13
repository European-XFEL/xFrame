__version__ = "0.1.0"

#make sure  numpy does not try to multiprocess over xframes multiprocessing
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

from xframe import startup_routines

startup_routines.import_settings()
startup_routines.setup_logging()
import logging
log = logging.getLogger('root')
startup_routines.startup_imports()
_external_dependency_lookup = startup_routines.create_dependency_dict()

startup_routines.dependency_injection()
dependency_inject_SOFT = startup_routines.dependency_inject_SOFT
startup_routines.setup_default_database()


# Initialize attributes #
from xframe.interfaces import ExperimentWorkerInterface,ProjectWorkerInterface
from xframe.control.Control import Controller
#The following variables will be set on runtime
_project_worker_module_name = False  # will be a string
_experiment_module_name = False      # will be a string
project_worker = ProjectWorkerInterface
experiment_worker = ExperimentWorkerInterface
controller = Controller()

startup_routines.setup_projects()
startup_routines.setup_experiments()
select_project = startup_routines.select_project
select_experiment = startup_routines.select_experiment
import_selected_project = startup_routines.import_selected_project
import_selected_experiment= startup_routines.import_selected_experiment
run = startup_routines.run
select_and_run = startup_routines.select_and_run


