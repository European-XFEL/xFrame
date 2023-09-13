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
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_remove_0_order,cross_correlation_low_pass
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
log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_100')[0] 
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi

ccd = db.load('ccd')
ccd_1 = db.load_ccd('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_model_0.h5')
cc = ccd['cross_correlation']
cc_1 = ccd_1['cross_correlation']

#new_cc,_locals =ccd_remove_0_order(cc)
#locals().update(_locals)
metadata = {}
proj_opt = settings.analysis.projections.reciprocal.pre_init.cross_correlation
extract_odd_orders = settings.analysis.projections.reciprocal.pre_init.extract_odd_orders
max_order = settings.analysis.projections.reciprocal.pre_init.max_order
#mask_opt = {'type': 'donatelli','donatelli':{'threshold': -1}}
#metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':proj_opt.n_particles}        
#new_bl = cross_correlation_to_deg2_invariant(new_cc,3,metadata)[0]
metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':1,'extract_odd_orders':False}    
bl = cross_correlation_to_deg2_invariant(cc_1,3,metadata)[0]

max_orders = [5,10,15,20,25,99]
#max_orders = [5,10]
n_orders = len(max_orders)
order = 2
grid = GridFactory.construct_grid('uniform',[ccd['qs'],ccd['qs']])
bl_2_l = [np.abs(bl[order]).real]
bl_2_0_f = [np.abs(bl[order]).real]
bl_2_t = [np.abs(bl[order]).real]
bl_2_f = [np.abs(bl[order]).real]
metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':proj_opt.n_particles,'extract_odd_orders':False}    
cc_mask = cross_correlation_mask(ccd['data_grid'],metadata)
for max_order in max_orders:
    new_cc = np.copy(cc)
    new_cc[~cc_mask]=0
    new_cc,_locals = cross_correlation_low_pass(cc,max_order)   
    #metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':proj_opt.n_particles,'extract_odd_orders':True}    
    #bl_2_t.append(np.abs(cross_correlation_to_deg2_invariant(cc,3,metadata)[0][order]).real)
    #bl_2_0_f.append(np.abs(cross_correlation_to_deg2_invariant(new_cc,3,metadata)[0][order]).real)
    bl_new = cross_correlation_to_deg2_invariant(cc_new,3,metadata)[0][order]
    bl_old = cross_correlation_to_deg2_invariant(cc,3,metadata)[0][order]
    bl_2_f.append(np.abs(bl_old).real)
    bl_2_l.append(np.abs(bl_new).real)
    


#ls = 0,2,4,6,8
#db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/bl_without_0_100_mO11.matplotlib',heat2D_multi.get_fig([[np.abs(bl[l]).real for l in ls],[np.abs(new_bl[l]).real for l in ls]],scale = 'log',shape = [2,len(ls)],size = (30,10),vmin= 1e14, vmax = 1e24,cmap = 'plasma'))

layout_l = {'title':'$B_2$ low-pass-filtered','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
layout_f = {'title':'$B_2$','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
layout = tuple(zip(*[[layout_f]*(n_orders+1),[layout_l]*(n_orders+1)]))

db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/b0_comparison_low_pass.matplotlib',heat2D_multi.get_fig(tuple(zip(*[bl_2_f,bl_2_l])),layout=layout, grid = grid,scale = 'log',shape = [len(max_orders)+1,2],size = (30,10*(len(bl_2_f)+1)),vmin= 1e14, vmax = 1e24,cmap = 'plasma'))
