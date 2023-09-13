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
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_associated_legendre_matrices
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
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi,plot1D
from xframe import Multiprocessing
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular

log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
from xframe.library.mathLibrary import gsl
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_10p_spher')[0] 
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)


ccd = db.load('ccd')
ccd_1 = db.load_ccd('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_model_0.h5')
#ccd_1 = db.load_ccd('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_pentagon_10particles_rectSim.h5')
cc = ccd['cross_correlation']
cc_1 = ccd_1['cross_correlation']

pre_init_opt = settings.analysis.projections.reciprocal.pre_init
proj_opt = pre_init_opt.cross_correlation

def init_transforms():
    cc_data = ccd_1['cross_correlation'][...,:800]
    maxQ = np.max(ccd_1['radial_points'])*2
    max_order = 99
    n_radial_points = 128#300
    opt=settings.analysis
    ht_opt={'dimensions':3,'max_order':max_order,'indices':'lm','anti_aliazing_degree':2,'n_phi':False,'n_theta':False}
    #ht_opt={'dimensions':opt.dimensions,**opt['grid'],'type':opt['fourier_transform']['type'],'pi_in_q':opt['fourier_transform']['pi_in_q'],'reciprocal_cut_off':maxQ}
    # harmonic transforms and grid
    cht=HarmonicTransform('complex',ht_opt)
    weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points,expansion_limit = 2000)
    grid_opt={'dimensions':3,'type':'Zernike_dict','max_q':maxQ,'n_radial_points':n_radial_points,**cht.grid_param,'pi_in_q':True}
    grid_pair = get_grid(grid_opt)
    maxR = grid_pair.real[:,0,0].max()
    ft,ift = generate_zernike_spherical_ft(maxR,weight_dict,cht,'complex',True,use_gpu=True)    
    return locals()

#locals().update(init_transforms())

def generate_density_spheres(grid,radius = 100, norm = 'standard',cart_centers= np.array([[0,200,0],[0,-200,0],[200,0,0],[-200,0,0],[400,0,0],[0,0,300]],dtype = np.float64),random_orientation = False):
    centers = mLib.cartesian_to_spherical(cart_centers)
    #centers = [(100,0,np.pi),(100,np.pi,np.pi),(100,0,0),(200,np.pi,0),(300,np.pi,np.pi/2)]
    sphere_funcs = [mLib.SampleShapeFunctions.get_disk_function(radius,center = center,norm=norm,random_orientation = random_orientation) for center in centers]
    density = sphere_funcs.pop(0)(grid)
    for sphere_func in sphere_funcs:
        density += sphere_func(grid)
    return density


def init2():
    r = 65
    phis = np.arange(5)*2*np.pi/5
    centers = 2*r*np.stack((np.sin(phis),np.cos(phis),np.zeros(5)),axis = -1)
    centers = list(centers)
    centers.append(np.array([0.0,0.0,0.0]))
    centers = np.array(centers)
    centers2 = np.array([centers[0],centers[2],centers[3]])
    density1 = generate_density_spheres(grid_pair.real,radius = r,cart_centers=centers)
    density2 = generate_density_spheres(grid_pair.real,radius = r,cart_centers=centers2)
    density = 25*(density1+density2)
    #bls_exact = density_to_deg2_invariants_3d(density,cht,ft)
    
    return locals()

#locals().update(init2())

def plot2():
    orders = np.arange(0,10,2)
    #orders = np.arange(10,100,10)
    #orders = [2]
    qs = grid_pair.reciprocal[:,0,0,0]
    qq_grid = GridFactory.construct_grid('uniform',[qs,qs])
    bls = bls_exact
    
    fig = heat2D_multi.get_fig(list(np.abs(bls[orders])),grid = qq_grid,shape = (1,len(orders)),size=(5*len(orders)+2*5,5),vmin = 1e+14, vmax = 1e+26,scale='log',cmap='plasma')
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl_exact0-10.matplotlib',fig)

    fig = heat2D_multi.get_fig(list(np.abs(mLib.nearest_positive_semidefinite_matrix(bls[orders]))),grid = qq_grid,shape = (1,len(orders)),size=(5*len(orders)+2*5,5),vmin = 1e+14, vmax = 1e+26,scale='log',cmap='plasma')
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl_exact_psd0-10.matplotlib',fig)

    intensity_exact = (np.abs(ft(density))**2).real
    Ilm = cht.forward(intensity_exact)
    fig = plot1D.get_fig(np.abs(Ilm[2][:,0]),y_scale='log',grid = qs)
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/Ilm_exact.matplotlib',fig)
    #bl2_new = mLib.nearest_positive_semidefinite_matrix(bls[...,2])
    #fig = heat2D.get_fig(np.abs(bl2_new.real),scale='log')
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl_exact2n.matplotlib',fig)
    #bl2_new = (bls[...,2]+bls[...,2].T)/2
    #fig = heat2D.get_fig(np.abs(bl2_new.real),scale='log')
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl_exact2nn.matplotlib',fig)
    #
    #s,v = np.linalg.eigh(bl2_new)
    #s[s<s.max()*1e-15]=0
    #bl2_new2 = v*s[None,:] @ v.T 
    #fig = heat2D.get_fig(np.abs(bl2_new2.real),scale='log')
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl_exact2nnn.matplotlib',fig)

#plot2()



#rd = ft(density)
#I = np.abs(rd*rd.conj()).real
#Ilm = cht.forward(I)
#phases = rd/np.sqrt(I)
#
#multiplicies = [1,0,1/100,1/10,1/2,2,10,100]
#intensities = []
#for m in multiplicies:
#    temp = [np.copy(Il)*m for Il in Ilm]
#    temp[0] = np.copy(Ilm[0])
#    intensities.append(cht.inverse(temp))
#
#db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/N_particles_shift.vtk',[d.copy() for d in intensities],grid=grid_pair.reciprocal,grid_type='spherical')

