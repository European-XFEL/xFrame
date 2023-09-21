import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward,cartesian_to_spherical,spherical_to_cartesian
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward

import logging
#from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import SimpleRegridder2D,AgipdRegridderSimple

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter

log=logging.getLogger('root')

init = True
if init:
    from xframe.startup_routines import dependency_injection
    dependency_injection()
    from xframe.startup_routines import load_recipes
    from xframe.control.Control import Controller
    analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='saxs_65_-bg',exp_name='fxs3046_online',exp_settings = 'default_proc')
    controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
    controller.chooseJobSingleProcess()
    from xframe import database
    from xframe import Multiprocessing

from xframe import database
db = database.analysis


    
comm = Multiprocessing.comm_module
g = comm.get_geometry()

base_path = '/gpfs/exfel/theory_group/user/berberic/fxs3046/test/regrid2/'

f_data = db.load('/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_64/2022-09-02_23:12:32/bg_data_proc.h5')


mean_dict = f_data['mean_2d']
bg_data={}
bg_data['mean'] = mean_dict['mean']
log.info('mean total 2 ={}'.format(np.mean(bg_data['mean'])))
bg_data['mask'] = mean_dict['mask']
#bg_data['settings_dir'] = self.opt['out_settings_path']
bg_data['run'] = 64
bg_data['n_frames']  =  275379
db.save('background',bg_data)
