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
from xframe.library.gridLibrary import GridFactory,ReGrider,NestedArray
from xframe.externalLibraries.flt_plugin import LegendreTransform
mathLibrary.leg_trf = LegendreTransform
from xframe.externalLibraries.shtns_plugin import sh
mathLibrary.shtns = sh
from xframe.control.Control import Controller
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_to_deg2_invariant
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
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_zernike_spherical_ft,zernike_radial_step_cutoff_relation, zernike_radial_cutoff_relation
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import ReciprocalProjection_old,ReciprocalProjection
from xframe import settings
from xframe.library import mathLibrary as mLib
from xframe import Multiprocessing
from xframe.library import physicsLibrary as pLib
log=log.setup_custom_logger('root','INFO')
import pdb_eda

dependency_injection_no_soft()
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)

#f = db.load('pdb://2X53')
#db.save('test.vtr',[f['density']],grid = f['grid'][:], grid_type = 'cartesian')

a = pdb_eda.fromPDBid('3A12')
d = a.densityObj
h = d.header

crs_grid = GridFactory.construct_grid('uniform',[np.arange(n) for n in h.ncrs]).array
#crs_grid = np.transpose(crs_grid,axes = tuple(h.map2xyz)+(3,))
flattened_grid = crs_grid.reshape(-1,3)
flattened_grid += np.array(h.crsStart)[None,:]
flattened_grid = flattened_grid[:,h.map2xyz]/np.array(h.xyzInterval)[None,:]
xyz_shape = tuple(np.array(h.ncrs)[h.map2xyz])
xyz_grid = (np.dot(h.orthoMat,flattened_grid.T).T).reshape(tuple(h.ncrs)+(3,))
xyz_grid = np.moveaxis(xyz_grid,(0,1,2),h.map2xyz)
density = np.moveaxis(d.density, (0,1,2),h.map2xyz)

particle_size = 340
oversampling = 4
max_r = particle_size/2*oversampling


max_order = 99
n_radial_points = 128
opt=settings.analysis
ht_opt={'dimensions':3,'max_order':max_order,'indices':'lm','anti_aliazing_degree':2,'n_phi':False,'n_theta':False}
#ht_opt={'dimensions':opt.dimensions,**opt['grid'],'type':opt['fourier_transform']['type'],'pi_in_q':opt['fourier_transform']['pi_in_q'],'reciprocal_cut_off':maxQ}
# harmonic transforms and grid
cht=HarmonicTransform('complex',ht_opt)
weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points)

max_q = zernike_radial_cutoff_relation(max_r,n_radial_points)

grid_opt={'dimensions':3,'type':'Zernike_dict','max_q':max_q,'n_radial_points':n_radial_points,**cht.grid_param,'pi_in_q':True}
grid_pair = get_grid(grid_opt)

#new_data = ReGrider.regrid(density,NestedArray(xyz_grid,1),'cartesian',NestedArray(grid_pair.real,1),'spherical',options={'fill_value': 0.0,'interpolation':'linear'})
db.save('test.vtr',[density],grid = xyz_grid, grid_type = 'cartesian')
