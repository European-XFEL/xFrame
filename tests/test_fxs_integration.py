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
            
@pytest.fixture(scope='module')
def patterns():
    patterns = np.random.rand(*TestsCorrelate.pattern_shape).astype(np.float32)
    paths = [os.path.join(tmp_config_path,TestsCorrelate.pattern_folder,f'{i}.raw') for i in range(TestsCorrelate.pattern_shape[0])]
    for p,d in zip(paths,patterns):
        xframe.database.default.save(p,d)
    pattern_list = [p+'\n' for p in paths]
    xframe.database.default.save(os.path.join(tmp_config_path,TestsCorrelate.pattern_list_path),pattern_list)
      
@pytest.fixture(scope='module')
def run_correlate(patterns):
    pattern_shape = TestsCorrelate.pattern_shape    
    settings_path= os.path.join(tmp_config_path,'settings/projects/fxs/correlate/test.yaml')
    settings = [
        "structure_name: '3d_test_corr'\n",
        "image_dimensions:\n",
        f"  command: '{pattern_shape[1:]}'\n",
        "n_processes: 32\n",
        f"max_n_patterns: {pattern_shape[0]}\n",
        f"detector_origin: {[(pattern_shape[1]-1)/2]*2}\n",
        "phi_range:\n",
        f"  command: '(0,2*np.pi,{pattern_shape[1]*2},\"exact\")'\n",
        f"fc_n_max: {pattern_shape[1]-1}\n",
        "IO:\n",
        "  folders:\n",
        "    in_base:\n",
        "      data: 'correlate/'\n"
    ]
    save(settings_path,settings)   
    xframe.select_project('fxs','correlate','test')
    xframe.run()
        
class TestsCorrelate:
    pattern_shape = (200,16,16)
    pattern_folder = 'data/fxs/correlate/patterns/'
    pattern_list_path = 'data/fxs/correlate/patterns_list.txt'
    def test_files_exist(self,run_correlate):
        date=xframe.database.project.get_time_string()
        files = [
            'data/fxs/ccd/3d_test_corr.h5',
            f'data/fxs/ccd/archive/3d_test_corr/{date}/run_0/settings.yaml',
            f'data/fxs/ccd/archive/3d_test_corr/{date}/run_0/ccd.h5',
        ]
        for fpath in files:
            path= os.path.join(tmp_config_path,fpath)
            exists = os.path.exists(path)
            assert exists,f'{path} does not exist'
    def test_file_contents(self,run_correlate):
        date=xframe.database.project.get_time_string()
        settings_path = os.path.join(tmp_config_path,f'data/fxs/ccd/archive/3d_test_corr/{date}/run_0/settings.yaml')
        check_settings(settings_path)
        self.check_h5_file_contents()
    def check_h5_file_contents(self):
        date=xframe.database.project.get_time_string()
        h5_path = os.path.join(tmp_config_path,f'data/fxs/ccd/archive/3d_test_corr/{date}/run_0/ccd.h5')
        data = xframe.database.default.load(h5_path)
        ccd_shape=(self.pattern_shape[1]//2,)*2+(self.pattern_shape[1]*2,)        
        data_structure = {'radial_points':[float,(self.pattern_shape[1]//2,)],
                          'angular_points':[float,(self.pattern_shape[1]*2,)],
                          'xray_wavelength':float,
                          'average_intensity':[float,(self.pattern_shape[1]//2,)],
                          'cross_correlation':{
                              'I1I1':[float,ccd_shape]
                          }
                          }
        check_dict_content(data,data_structure)


@pytest.fixture(scope='module')
def run_simulate_ccd():
    radial_points = TestsCorrelate.pattern_shape[1]//2
    max_order = TestsCorrelate.pattern_shape[1]
    settings_path= os.path.join(tmp_config_path,'settings/projects/fxs/simulate_ccd/test.yaml')
    settings = [
        "structure_name: '3d_test'\n",
        "dimensions: 3\n",
        "grid:\n",
        "  max_q: 0.017\n",
        f"  n_radial_points: {radial_points}\n",
        f"  max_order: {max_order}\n",
        "shapes:\n",
        "  types:\n",
        "    command: \"['sphere']*2\"\n",
        "  centers:\n",
        "    command: \"[(140.0,np.pi/2,phi*2*np.pi/2) for phi in range(2) ]\"\n",
        "  sizes:\n",
        "    command: \"[70]*2\"\n",
        "  densities: [25,50]\n",
        "cross_correlation:\n",
        "  method: 'back_substitution'\n",
        "  xray_wavelength: 1.23984\n",
    ]
    save(settings_path,settings)   
    xframe.select_project('fxs','simulate_ccd','test')
    xframe.run()
    
class TestsSimulateCCD:    
    def test_files_exist(self,run_simulate_ccd):
        date=xframe.database.project.get_time_string()
        files = [
            'data/fxs/ccd/3d_test.h5',
            f'data/fxs/ccd/archive/3d_test/{date}/run_0/settings.yaml',
            f'data/fxs/ccd/archive/3d_test/{date}/run_0/ccd.h5',
            f'data/fxs/ccd/archive/3d_test/{date}/run_0/model_density.vts',
        ]
        for fpath in files:
            path= os.path.join(tmp_config_path,fpath)
            exists = os.path.exists(path)
            assert exists,f'{path} does not exist'
    def test_files_contents(self,run_simulate_ccd):
        date=xframe.database.project.get_time_string()
        settings_path = os.path.join(tmp_config_path,f'data/fxs/ccd/archive/3d_test/{date}/run_0/settings.yaml')
        check_settings(settings_path)
        self.check_h5_file_contents()
    def check_h5_file_contents(self):
        date=xframe.database.project.get_time_string()
        h5_path = os.path.join(tmp_config_path,f'data/fxs/ccd/archive/3d_test/{date}/run_0/ccd.h5')
        data = xframe.database.default.load(h5_path)
        pattern_shape = TestsCorrelate.pattern_shape
        n_angular_points = 2*pattern_shape[1] 
        ccd_shape=(pattern_shape[1]//2,)*2+(n_angular_points,)
        data_structure = {'radial_points':[float,(pattern_shape[1]//2,)],
                          'angular_points':[float,(n_angular_points,)],
                          'xray_wavelength':float,
                          'average_intensity':[float,(pattern_shape[1]//2,)],
                          'cross_correlation':{
                              'I1I1':[float,ccd_shape]
                          }
                          }
        check_dict_content(data,data_structure)    

    
@pytest.fixture(scope='module')
def run_extract(run_simulate_ccd):
    radial_points = TestsCorrelate.pattern_shape[1]//2
    max_order = TestsCorrelate.pattern_shape[1]-1
    settings_path= os.path.join(tmp_config_path,'settings/projects/fxs/extract/test.yaml')
    settings = [
        "structure_name: 'test'\n",
        "dimensions: 3\n",
        f"max_order: {max_order}\n",
        "IO:\n",
        "  files:\n"
        "    ccd:\n",
        "      name: 3d_test.h5\n",
        "    invariants:\n",
        "      options:\n",
        "        plot_first_invariants: False\n",
        "        plot_first_invariants_from_proj_matrices: False\n",
    ]
    save(settings_path,settings)   
    xframe.select_project('fxs','extract','test')
    xframe.run()    
class TestExtract:
    def test_files_exist(self,run_extract):
        date=xframe.database.project.get_time_string()
        files = [
            'data/fxs/invariants/3d_test.h5',
            f'data/fxs/invariants/archive/3d_test/{date}/run_0/extraction_settings.yaml',
            f'data/fxs/invariants/archive/3d_test/{date}/run_0/proj_data.h5',
        ]
        for fpath in files:
            path= os.path.join(tmp_config_path,fpath)
            exists = os.path.exists(path)
            assert exists,f'{path} does not exist'
    def test_files_contents(self,run_extract):
        date=xframe.database.project.get_time_string()
        settings_path = os.path.join(tmp_config_path,f'data/fxs/invariants/archive/3d_test/{date}/run_0/extraction_settings.yaml')
        check_settings(settings_path)
        self.check_h5_file_contents()
    def check_h5_file_contents(self):
        date=xframe.database.project.get_time_string()
        h5_path = os.path.join(tmp_config_path,f'data/fxs/invariants/archive/3d_test/{date}/run_0/proj_data.h5')
        data = xframe.database.default.load(h5_path)
        pattern_shape = TestsCorrelate.pattern_shape
        n_angular_points = pattern_shape[1]*2
        max_order = pattern_shape[1]
        n_radial_points = pattern_shape[1]//2 
        ccd_shape=(pattern_shape[1]//2,)*2+(n_angular_points,)
        bl_shape=(max_order,)+(n_radial_points,)*2
        data_structure = {'dimensions': np.int64,
                          'xray_wavelength':float,
                          'max_order': np.int64,
                          'average_intensity':[float,(n_radial_points,)],
                          'data_angular_points':[float,(n_angular_points,)],
                          'data_radial_points':[float,(n_radial_points,)],
                          'deg_2_invariant':{'I1I1':[complex,bl_shape]},
                          'deg_2_invariant_masks':{'I1I1':[bool,bl_shape]}
                          }
                          
        check_dict_content(data,data_structure)    
        pr_matrices=data['data_projection_matrices']['I1I1']
        for l,mat in enumerate(pr_matrices):
            assert mat.dtype== complex, f'projection matrix of order {l} is not of dtype complex but {mat.dtype}.'
            assert np.isnan(mat).any() == False, f'projection matrix of order {l} contains Nans.'

            mat_shape = (n_radial_points,min(2*l+1,n_radial_points))
            assert mat.shape == mat_shape ,f'projection matrix of order {l} has wring shape. {mat.shape} instead of {mat_shape}'

        
@pytest.fixture(scope='module')
def run_reconstruct(run_extract):
    radial_points = TestsCorrelate.pattern_shape[1]//2
    max_order = TestsCorrelate.pattern_shape[1]-1
    settings_path= os.path.join(tmp_config_path,'settings/projects/fxs/reconstruct/test.yaml')
    settings = [
        "structure_name: 'test'\n",
        "dimensions: 3\n",
        f"particle_radius: 250\n",
        "grid:\n",
        "  n_radial_points: 8\n",
        "  max_order: 15\n",
        "projections:\n",
        "  reciprocal:\n",
        "    use_order_ids:\n",
        "      command: 'np.arange(16)'\n",
        "multi_process:\n",
        f"  n_parallel_reconstructions: {TestReconstruct.n_reconstructions}\n",
        "fourier_transform:\n",
        "  reciprocity_coefficient: 2.0\n",
        "main_loop:\n",
        "  sub_loops:\n",
        "    main:\n",
        "      iterations: 2\n",
        "IO:\n",
        "  files:\n",
        "    reconstructions:\n",
        "      options:\n",
        "        plot_first_used_invariants: False\n",
        "        plot_reconstructed_deg2_invariants: False\n"
    ]
    save(settings_path,settings)   
    xframe.select_project('fxs','reconstruct','test')
    xframe.run()
    
class TestReconstruct:
    n_reconstructions=10
    def test_files_exist(self,run_reconstruct):
        date=xframe.database.project.get_time_string()
        files = [
            f'data/fxs/reconstructions/3d_test/{date}/run_0/settings.yaml',
            f'data/fxs/reconstructions/3d_test/{date}/run_0/data.h5',
            f'data/fxs/reconstructions/3d_test/{date}/run_0/vtk/',
        ]
        for fpath in files:
            path= os.path.join(tmp_config_path,fpath)
            exists = os.path.exists(path)
            assert exists,f'{path} does not exist'        
    def test_files_contents(self,run_reconstruct):
        date=xframe.database.project.get_time_string()
        settings_path = os.path.join(tmp_config_path,f'data/fxs/reconstructions/3d_test/{date}/run_0/settings.yaml')
        check_settings(settings_path)
        self.check_h5_file_contents()
    def check_h5_file_contents(self):
        date=xframe.database.project.get_time_string()
        h5_path = os.path.join(tmp_config_path,f'data/fxs/reconstructions/3d_test/{date}/run_0/data.h5')
        data = xframe.database.default.load(h5_path)
        pattern_shape = TestsCorrelate.pattern_shape
        n_angular_points = pattern_shape[1]*2
        max_order = pattern_shape[1]-1
        n_radial_points = pattern_shape[1]//2
        from xframe.projects.fxs.projectLibrary.harmonic_transforms import HarmonicTransform
        grid_param = HarmonicTransform('complex',{'dimensions':3,'max_order':max_order}).grid_param
        rec_shape = (n_radial_points,len(grid_param['thetas']),len(grid_param['phis']))
        bl_shape = (max_order+1,)+(n_radial_points,)*2

        results_structure = {
            'error_dict':{
                'main':[float,(400,)],
                'real':{
                    'l2_projection_diff':[float,(400,)]
                },                
            },
            'final_error':float,
            'initial_density':[complex,rec_shape],
            'initial_support':[bool,rec_shape],
            'last_deg2_invariant':[complex,bl_shape],
            'last_real_density':[complex,rec_shape],
            'last_reciprocal_density':[complex,rec_shape],
            'last_support_mask':[bool,rec_shape],
            'real_density':[complex,rec_shape],
            'reciprocal_density':[complex,rec_shape],
            'support_mask':[bool,rec_shape],
            'n_particles':[float,(400,1)],
            'loop_iterations':np.int64,
            'fxs_unknowns':[[complex,(min(2*i+1,n_radial_points),2*i+1)] for i in range(max_order+1)]         
        }
        
        data_structure = {
            'configuration':{
                'internal_grid':{
                    'real_grid':[float,rec_shape+(3,)],
                    'reciprocal_grid':[float,rec_shape+(3,)]
                },
                'reciprocity_coefficient':float,
                'xray_wavelength':float,
            },
            'projection_matrices':[[float,(n_radial_points,1)]]+[[complex,(n_radial_points,min(2*i+1,n_radial_points))] for i in range(1,max_order+1)],
            'reconstruction_results': {str(i):results_structure for i in range(self.n_reconstructions)},
        }
                          
        check_dict_content(data,data_structure)

    
@pytest.fixture(scope='module')
def run_average(run_reconstruct):
    settings_path= os.path.join(tmp_config_path,'settings/projects/fxs/average/test.yaml')
    settings = [
        "structure_name: 'test'\n",
        "reconstruction_files:\n",
        "  - 3d_test/{today}/run_0/data.h5\n",        
        "multi_process:\n",
        "  use: True\n",
        "  n_processes: 2\n",
        "find_rotation:\n",
        "  r_limit_ids:\n",
        "    command: 'np.arange(4)'\n",        
    ]
    save(settings_path,settings)   
    xframe.select_project('fxs','average','test')
    xframe.run()
    
class TestAverage:
    def test_files_exist(self,run_average):
        date=xframe.database.project.get_time_string()
        files = [
            f'data/fxs/averages/3d_test/{date}/run_0/settings.yaml',
            f'data/fxs/averages/3d_test/{date}/run_0/average_results.h5',
            f'data/fxs/averages/3d_test/{date}/run_0/PRTF.png',
            f'data/fxs/averages/3d_test/{date}/run_0/vtk/real_average.vts',
            f'data/fxs/averages/3d_test/{date}/run_0/vtk/reciprocal_average.vts',
        ]
        for fpath in files:
            path= os.path.join(tmp_config_path,fpath)
            exists = os.path.exists(path)
            assert exists,f'{path} does not exist'
    def test_files_contents(self,run_average):
        date=xframe.database.project.get_time_string()
        settings_path = os.path.join(tmp_config_path,f'data/fxs/averages/3d_test/{date}/run_0/settings.yaml')
        check_settings(settings_path)
        self.check_h5_file_contents()
    def check_h5_file_contents(self):
        date=xframe.database.project.get_time_string()
        h5_path = os.path.join(tmp_config_path,f'data/fxs/averages/3d_test/{date}/run_0/average_results.h5')
        data = xframe.database.default.load(h5_path)
        pattern_shape = TestsCorrelate.pattern_shape
        n_angular_points = pattern_shape[1]*2
        max_order = pattern_shape[1]-1
        n_radial_points = pattern_shape[1]//2
        from xframe.projects.fxs.projectLibrary.harmonic_transforms import HarmonicTransform
        grid_param = HarmonicTransform('complex',{'dimensions':3,'max_order':max_order}).grid_param
        rec_shape = (n_radial_points,len(grid_param['thetas']),len(grid_param['phis']))
        bl_shape = (max_order+1,)+(n_radial_points,)*2
        input_structure = {
            'real_density':[complex,rec_shape],
            'reciprocal_density':[complex,rec_shape],
            'support_mask':[bool,rec_shape],
        }
        
        data_structure = {
            'input_meta':{
                'grids':{
                    'real_grid':[float,rec_shape+(3,)],
                    'reciprocal_grid':[float,rec_shape+(3,)]
                },
                'average_scaling_factors_per_file': [float,(1,)],
                'projection_matrices':[[[float,(n_radial_points,1)]]+[[complex,(n_radial_points,min(2*i+1,n_radial_points))] for i in range(1,max_order+1)]],
            },
            'average':{
                'intensity_from_densities':[float,rec_shape],
                'intensity_from_ft_densities':[float,rec_shape],
                'normalized_real_density':[complex,rec_shape],
                'real_density':[complex,rec_shape],
                'reciprocal_density':[complex,rec_shape]
            },
            'average_ids':[np.int64]*TestReconstruct.n_reconstructions,
            'input':{
              str(i): input_structure for i in range(TestReconstruct.n_reconstructions)
            },            
        }                                  
        check_dict_content(data,data_structure)
