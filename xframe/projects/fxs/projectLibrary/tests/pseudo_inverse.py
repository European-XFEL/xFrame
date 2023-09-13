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
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_mask
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_apply_precision_filter
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_to_deg2_invariant_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import bl_3d_pseudo_inverse_worker
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_cc_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import density_to_deg2_invariants_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_eigenvalues
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_legendre_matrices
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
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_100')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi


def init_transforms(maxQ):
    max_order = 99
    n_radial_points = 128
    opt=settings.analysis
    ht_opt={'dimensions':3,'max_order':max_order,'indices':'lm','anti_aliazing_degree':2,'n_phi':0,'n_theta':0}
    #ht_opt={'dimensions':opt.dimensions,**opt['grid'],'type':opt['fourier_transform']['type'],'pi_in_q':opt['fourier_transform']['pi_in_q'],'reciprocal_cut_off':maxQ}
    # harmonic transforms and grid
    cht=HarmonicTransform('complex',ht_opt)
    weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points)
    grid_opt={'dimensions':3,'type':'Zernike_dict','max_q':maxQ,'n_radial_points':n_radial_points,**cht.grid_param,'pi_in_q':True}
    grid_pair = get_grid(grid_opt)
    maxR = grid_pair.real[:,0,0].max()
    ft,ift = generate_zernike_spherical_ft(maxR,weight_dict,cht,'complex',True,use_gpu=True)
    orders = np.arange(max_order+1)
    return locals()


ccd = db.load('ccd')

#mask_dict_true={'type':'direct','direct':{'mask':np.full(cc_1.shape,True,dtype = bool)}}
#bl,_mask = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,{**ccd,'orders':orders,'mask':mask_dict_true,'mode':'pseudo_inverse'})

phis = ccd['phis']
phi_mask = phis<np.pi
phis = phis[phi_mask]
qs = ccd['qs']
thetas = ccd['thetas']
orders = np.arange(0,50)
q_ids = np.arange(128)

leg_m = ccd_legendre_matrices(q_ids,q_ids,phis,thetas,orders)
leg_m_inv = np.linalg.pinv(leg_m)
bl_initial = bl[:,q_ids,q_ids]


cc_initial = np.sum(bl_initial[...,None]*np.moveaxis(leg_m,-1,0),axis=0)


n_m_phi = 200
leg_m = ccd_legendre_matrices(q_ids,q_ids,phis[n_m_phi:-n_m_phi],thetas,orders)
leg_m_inv = np.linalg.pinv(leg_m)
bl_recovered = np.sum(cc_initial[:,None,n_m_phi:-n_m_phi]*leg_m_inv,axis=-1)

cc_mask = np.full((len(qs),len(qs),len(phis)),True)
cc_mask[...,:n_m_phi]=False
cc_mask[...,-n_m_phi:]=False
cc_compatible = np.zeros(cc_mask.shape,dtype=float)
cc_compatible[q_ids,q_ids]=cc_initial

bl_recovered_2 = bl_3d_pseudo_inverse_worker(q_ids,q_ids,cc_compatible,phis,thetas,orders,cc_mask)

diff = np.abs(bl_recovered[1:]-bl_initial.T[1:])
#diff = np.abs(bl_recovered_2[1:]-bl_recovered[1:])
error = diff/np.abs(bl_initial.T[1:])
#error = diff/np.abs(bl_recovered[1:])
error[diff==0] = 0
print(error[-1,:])
print(error.max())

