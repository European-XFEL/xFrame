import pytest
import os
import sys
import subprocess
import numpy as np
import importlib
import numpy as np
from tlib import copy,save,load,create_path_if_nonexistent
import xframe
config_path = os.path.expanduser('~/.xframe/config.py')
config_backup_path = os.path.expanduser('~/.xframe/backup_config.py')

tmp_config_path = ''
_path = ''
xfel_path='/gpfs/exfel/exp/SPB/'

'''
Disclaimer: This module houses integration tests for the fxs module.
Currently this means that the functions:

correlate
simulate_ccd
extract
reconstruct
average

will be executed in sequence for sample structure with small grid size.
After each execution we check whether the output files exist and contain roughly sensible data.
No detailed checks on the output data is performed.
'''

@pytest.fixture(scope='module')
def tmp_path(tmpdir_factory):
    path = tmpdir_factory.mktemp('xtest'+str(np.random.rand()))
    return path

@pytest.fixture(scope='module',autouse=True)
def set_temp_home(tmp_path):    
    config_file_exists = os.path.exists(config_path)
    global tmp_config_path
    tmp_config_path = f'{tmp_path}/.xframe/'
    if config_file_exists:
        initial_config = load(config_path)
        print(f'initial config content = {initial_config}')
        copy(config_path,config_backup_path)
    xframe.change_home(tmp_config_path)
    yield
    if config_file_exists:
        copy(config_backup_path,config_path)
    else:
        os.remove(config_file)
    print(f'config after test = {load(config_path)}')
    
def home_files_exist():
    paths=[
        'projects/fxs',
        'projects/tutorial',
        'experiments/SPB',
        'settings/projects/fxs/correlate/tutorial.yaml',
        'settings/projects/fxs/correlate/default_0.01.yaml',
        'settings/projects/fxs/simulate_ccd/tutorial.yaml',
        'settings/projects/fxs/simulate_ccd/default_0.01.yaml',
        'settings/projects/fxs/extract/tutorial.yaml',
        'settings/projects/fxs/extract/default_0.01.yaml',
        'settings/projects/fxs/reconstruct/tutorial.yaml',
        'settings/projects/fxs/reconstruct/default_0.01.yaml',
        'settings/projects/fxs/average/tutorial.yaml',
        'settings/projects/fxs/average/default_0.01.yaml',
        'settings/experiments/SPB/tutorial.yaml',
        'settings/experiments/SPB/default_0.01.yaml',
    ]
    global tmp_config_path
    for path in paths:
        _path= os.path.join(tmp_config_path,path)
        exists = os.path.exists(_path)
        if not exists:
            assert _path == None


def check_settings(file_path):
    '''
    Checks if file contents are same as converted(replacing numpy arrays and types)
    '''
    loaded_settings = xframe.database.default.load(file_path)
    converter = xframe.settings.tools.SettingsConverter
    program_settings = converter.convert(xframe.settings.project.dict())
    raw_settings = converter.convert(xframe.settings.raw_project)
    assert (loaded_settings == program_settings) or (loaded_settings == raw_settings)
    
def check_dict_content(_dict,template):
    for key,temp in template.items():
        assert key in _dict, f'Key "{key}" could not be found.'
        val = _dict[key]
        if isinstance(temp,dict):
            check_dict_content(val,temp)
        elif isinstance(temp,(tuple,list)):
            if isinstance(val,xframe.lib.grid.NestedArray):
                val=val.array
            if isinstance(val,(tuple,list)):
                check_list_content(val,temp)
            else:
                assert isinstance(val,np.ndarray),key+' : '+str(_dict.keys())
                assert val.dtype == temp[0],key+' : '+str(_dict.keys())
                assert val.shape == temp[1],key+' : '+f'wrong shape,array = {val}'
                assert np.isnan(val).any() == False,key+' : '+str(_dict.keys())
        else:
            assert isinstance(val,temp),key+' : ' + f'is not of type {temp} but {type(val)}'
            
def check_list_content(_list,template):
    for val,temp in zip(_list,template):
        if isinstance(val,(list,tuple)):
            check_list_content(val,temp)
        else:
            if isinstance(temp,(tuple,list)):
                assert isinstance(val,np.ndarray), f'expected numpy array but got {type(val)}'
                assert val.dtype == temp[0], f'wrong dtype expected {temp[0]} got {val.dtype}'
                assert val.shape == temp[1], f'wrong shape expected {temp[1]} got {val.shape}'
                assert np.isnan(val).any() == False, f'array contains nan values!'
            else:
                assert isinstance(val,temp),f'wrong dtype expected {temp} got {type(val)}'                            


def has_access_to_gpfs():
    return not os.path.exists(xfel_path)

@pytest.mark.skipif(has_access_to_gpfs(),reason="Need access to exfel GPFS")
def test_get_data(set_temp_home):
    xframe.select_experiment('SPB','tutorial')
    xframe.import_selected_experiment()
    exp = xframe.experiment
    run = 162
    selection = exp.DataSelection(run,n_frames=2000)
    data_generator = exp.get_data(selection)

    for chunk in data_generator:
        print(chunk.keys())
        del(chunk)

@pytest.mark.skipif(has_access_to_gpfs(),reason="Need access to exfel GPFS")
def test_filters(set_temp_home):
    xframe.select_experiment('SPB','tutorial')
    opt = {
        "filter_sequence": ["norm","lit_pixels"],
        "filters":{
            "lit_pixels":{
                "class": "LitPixels",
                "lit_threshold": 1e5,
                "limits": [0.04,None]
            },
            "norm":{
                "class": "NormalizationFilter"
            }               
        }
    }
    xframe.settings.experiment.update(opt)    
    xframe.import_selected_experiment()
    exp = xframe.experiment
    run = 162
    selection = exp.DataSelection(run,n_frames=2000)
    data_generator = exp.get_data(selection)

    for chunk in data_generator:
        print(chunk.keys())
        # the assert valu is specific to run 162 of exp /gpfs/exfel/exp/SPB/202202/p003046/
        assert chunk['data'].mean()<100,'Normalization did not work'
        del(chunk)
        
@pytest.mark.skipif(has_access_to_gpfs(),reason="Need access to exfel GPFS")
def test_rois(set_temp_home):
    opt = {
        "filter_sequence": ["norm","lit_pixels"],
        "filters":{
            "lit_pixels":{
                "class": "LitPixels",
                "lit_threshold": 1e5,
                "limits": [0.04,None]
            },
            "norm":{
                "class": "NormalizationFilter",
                "ROIs": ['rect1','donut','asic070']
            }               
        },
        "ROIs":{
            'rect1':{
                'class':'Rectangle',
                'parameters':{
                    'center': [0.3,0.02],
                    'x_len': 0.2,
                    'y_len': 0.2
                }
            },
            'donut':{
                'class':'Annulus',
                'parameters':{
                    'center': [0,0],
                    'inner_radius': 0.07,
                    'outer_radius': 0.12
                }
            },
            'asic070':{
                'class': "Asic",
                'parameters':{
                    'asics': [[0,7,0]]
                }
            }            
        }
    }
    xframe.settings.experiment.update(opt)    
    xframe.import_selected_experiment()
    exp = xframe.experiment
    run = 162
    selection = exp.DataSelection(run,n_frames=2000)
    data_generator = exp.get_data(selection)

    for chunk in data_generator:
        print(chunk.keys())
        # the assert valu is specific to run 162 of exp /gpfs/exfel/exp/SPB/202202/p003046/
        del(chunk)
        
@pytest.mark.skipif(has_access_to_gpfs(),reason="Need access to exfel GPFS")
def test_geometry(set_temp_home):
    xframe.select_experiment('SPB','tutorial')
    xframe.import_selected_experiment()
    exp = xframe.experiment
    geom = exp.get_geometry
    grid = exp.get_pixel_grid_reciprocal
    agipd = exp.detector
    agipd.base = (0,0,100)
    assert (grid != exp.get_pixel_grid_reciprocal()).all(), 'pixel grid was not updated.'
    module0 = agipd.modules[0].detection_plane
    
