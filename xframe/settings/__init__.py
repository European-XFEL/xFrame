import os
import sys
from importlib import util
from . import general
from .tools import DictNamespace

project = DictNamespace.dict_to_dictnamespace({}) 
_project_at_run = False
_project_at_select = False
raw_project = DictNamespace.dict_to_dictnamespace({})

experiment = DictNamespace.dict_to_dictnamespace({}) 
_experiment_at_run = False
_experiment_at_select = False
raw_experiment = DictNamespace.dict_to_dictnamespace({}) 

# only to make it the same type as the other settings #
gd = {key:general.__getattribute__(key) for key in dict(general.__dict__).keys() if key[0]!="_"}
general = DictNamespace.dict_to_dictnamespace(gd)

def _update_settings_on_project_select(opt,raw_opt):
    global project
    global raw_project
    global _project_at_run
    global _project_at_select
    project = opt
    raw_project = raw_opt
    _project_at_run = False
    _project_at_select = opt.dict()
    
def _update_settings_on_experiment_select(opt,raw_opt):
    global experiment
    global raw_experiment
    global _experiment_at_run
    global _experiment_at_select
    experiment = opt
    raw_experiment = raw_opt
    _experiment_at_run = False
    _experiment_at_select = opt.dict()

def _save_settings_on_controller_run():
    global _project_at_run 
    _project_at_run = project.dict()
    global _experiment_at_run
    _experiment_at_run = experiment.dict()

def get_settings_to_save():    
    project_save=raw_project
    if isinstance(_project_at_select,dict) and isinstance(_project_at_run,dict):
        project_settings_where_modified = _project_at_select != _project_at_run
        if project_settings_where_modified:
            project_save = _project_at_run
    experiment_save=raw_experiment
    if isinstance(_experiment_at_select,dict) and isinstance(_experiment_at_run,dict):
        experiment_settings_where_modified = _experiment_at_select != _experiment_at_run
        if experiment_settings_where_modified:
            experiment_save = _experiment_at_run
    return project_save,experiment_save
