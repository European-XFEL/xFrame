import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import numpy as np
import scipy
import logging
import traceback
from xframe import log
os.chdir(os.path.expanduser('~/Programs/xframe'))
from xframe.startup_routines import load_recipes
from xframe.startup_routines import dependency_injection_no_soft
from xframe.library import mathLibrary
from xframe.externalLibraries.flt_plugin import LegendreTransform
mathLibrary.leg_trf = LegendreTransform
from xframe.externalLibraries.shtns_plugin import sh
mathLibrary.shtns = sh
from xframe.control.Control import Controller
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_to_deg2_invariant
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_remove_0_order
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_mask
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_apply_precision_filter
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_to_deg2_invariant_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import bl_3d_pseudo_inverse_worker
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_cc_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import density_to_deg2_invariants_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_eigenvalues
from xframe.plugins.MTIP.analysisLibrary import fxs_invariant_tools as i_tools
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict_zernike_spherical
from xframe.plugins.MTIP.analysisLibrary.harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_zernike_spherical_ft
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import ReciprocalProjection_old,ReciprocalProjection
from xframe import settings
from xframe.library import mathLibrary as mLib
from xframe import Multiprocessing
from xframe.library import physicsLibrary as pLib
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import heat2D_multi 
log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_100')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi

ccd = db.load('ccd')
cc = ccd['cross_correlation']

new_cc,_locals =ccd_remove_0_order(cc)
#locals().update(_locals)
metadata = {}
proj_opt = settings.analysis.projections.reciprocal.pre_init.cross_correlation
max_order = settings.analysis.projections.reciprocal.pre_init.max_order
metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':proj_opt.n_particles}        
new_bl = cross_correlation_to_deg2_invariant(new_cc,3,metadata)[0]
bl = cross_correlation_to_deg2_invariant(cc,3,metadata)[0]


ls = 0,2,4,6,8
db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/cc0.matplotlib',heat2D_multi.get_fig([[bl[l] for l in ls],[new_bl[l] for l in ls]],scale = 'log',shape = [2,len(ls)]))
