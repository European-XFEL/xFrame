import numpy as np
import os
import glob
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

init = False
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
    experiment = Multiprocessing.comm_module.get_experiment()

db = default_DB()

calib = experiment.calibrator
raw_base = '/gpfs/exfel/exp/SPB/202202/p003046/raw/'
runs = np.arange(160,170)
diod_path = 'INSTRUMENT/SPB_RR_SYS/ADC/UTC1-1:channel_0.output/data/rawData'
diod_train_path = 'INSTRUMENT/SPB_RR_SYS/ADC/UTC1-1:channel_0.output/data/trainId'
for run in runs:
    try:
        #calib.set_ids_for_run(run)
        raw_folder = raw_base + 'r{run:04d}/'.format(run = run)
        dataset_names = raw_folder + 'RAW-R{run:04d}-DIGI01-*.h5'.format(run = run)
        dset_paths = glob.glob(dataset_names)
        
        diod_list = []
        train_ids = []
        for path in dset_paths:
            print(path)
            with db.load(path,as_h5_object = True) as f:    
                d = f[diod_path][:]
                train_ids_part = f[diod_train_path][:]
                nonzero_mask = train_ids_part!=0
                max_value = np.max(d ,axis =1)
                diod_on = max_value>1000
                diod_list.append(diod_on[nonzero_mask])
                train_ids.append(train_ids_part[nonzero_mask])
        diod_array = np.concatenate(diod_list)    
        train_id_array = np.concatenate(train_ids)
        lookup = np.argsort(train_id_array)
        train_id_array = train_id_array[lookup]
        diod_array = diod_array[lookup]
        data = {'trainIds':train_id_array,'diod_on':diod_array}
        db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_diod/r{run:04d}.h5'.format(run=run),data)
    except Exception as e:
        print(e)
    
