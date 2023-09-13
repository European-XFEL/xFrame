import numpy as np
import os
import glob
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward,cartesian_to_spherical,spherical_to_cartesian
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward,masked_mean
from xframe.presenters.matplolibPresenter import plot1D,hist2D,heat2D



import logging
#from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import SimpleRegridder2D,AgipdRegridderSimple

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter

log=logging.getLogger('root')

init  = True
if init:
    from xframe.startup_routines import dependency_injection
    dependency_injection()
    from xframe.startup_routines import load_recipes
    from xframe.control.Control import Controller
    analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='saxs_65_-bg',exp_name='fxs3046_online',exp_settings = 'default_proc')
    from xframe import database
    #controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
    #controller.chooseJobSingleProcess()
    #from xframe import database
    #from xframe import Multiprocessing
    #experiment = Multiprocessing.comm_module.get_experiment()
    #aw = controller.controlWorker.analysis_worker
    #aw.geometry = aw.load_geometry()
    #aw.agipd_regridder = False
    #r = controller.controlWorker.analysis_worker.set_and_get_agipd_regridder()

    

db = database.experiment
db.create_vds(1,False)
