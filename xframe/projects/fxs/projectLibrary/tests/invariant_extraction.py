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
log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)


def test_cc_bl_pseudo_inverse():
    ccd= db.load('ccd')
    cc_data = ccd['cross_correlation']
    
    orders = opt.fourier_transform.pos_orders[opt.projections.reciprocal.used_order_ids]
    
    bl = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,{**ccd,'orders':orders,'mode':'pseudo_inverse','cc_mask_threshold':-1})
    cc_inverted = deg2_invariant_to_cc_3d(bl,ccd['xray_wavelength'],ccd['data_grid'],orders)
    angles = ccd["phis"]<np.pi
    
    rel_diff = np.abs(cc_data-cc_inverted)[...,angles]/np.abs(cc_data[...,angles])
    print('Pseudo inverse ruslan_data: mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
    return locals()


def test_cc_bl_legendre():
    ccd= db.load('ccd')
    cc_data = ccd['cross_correlation']
    
    orders = np.arange(0,100,2)
    
    bl2 = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,{**ccd,'orders':orders,'mode':'legendre_approx','cc_mask_threshold':-1})
    cc_inverted2 = deg2_invariant_to_cc_3d(bl2,ccd['xray_wavelength'],ccd['data_grid'],orders)

    angles = ccd["phis"]<np.pi
    
    rel_diff = np.abs(cc_data-cc_inverted2)[...,angles]/np.abs(cc_data[...,angles])
    print('Legendre_approx ruslan_data: mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
    return locals()

#locals().update(test_cc_bl_legendre())
#locals().update(test_cc_bl_pseudo_inverse())

def test_bl_pseudo_inverse():
    ccd= db.load('ccd')
    cc_data = ccd['cross_correlation']
    
    orders = opt.fourier_transform.pos_orders[opt.projections.reciprocal.used_order_ids]
    
    bl = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,{**ccd,'orders':orders,'cc_mask_threshold':-1})
    cc_inverted = deg2_invariant_to_cc_3d(bl,ccd['xray_wavelength'],ccd['data_grid'],orders)
    bl2 = cross_correlation_to_deg2_invariant(cc_inverted,3,{**ccd,'orders':orders,'cc_mask_threshold':-1})
    angles = ccd["phis"]<np.pi

    nonzero = np.abs(bl)!=0
    rel_diff = np.abs(bl-bl2)[nonzero]/np.abs(bl[nonzero])
    print('Pseudo inverse ruslan_data: mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
    return locals()
#locals().update(test_bl_pseudo_inverse())

#rel_diff_leg = np.abs((cc_data[...,:800] - cc_inverted2)/cc_data[...,:800])
#rel_diff_pseudo = np.abs((cc_data[...,:800] - cc_inverted)/cc_data[...,:800])

def test_bl_legendre():
    ccd= db.load('ccd')
    cc_data = ccd['cross_correlation']
    angles = ccd["phis"]<np.pi
    
    orders = opt.fourier_transform.pos_orders[opt.projections.reciprocal.used_order_ids][::2]
    
    bl = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,{**ccd,'orders':orders,'mode':'legendre_approx','cc_mask_threshold':-1})
    cc_inverted = deg2_invariant_to_cc_3d(bl,ccd['xray_wavelength'],ccd['data_grid'],orders)
    rel_diff = np.abs(cc_data-cc_inverted)[...,angles]/np.abs(cc_data[...,angles])
    print('Legendre_approx ruslan_data: mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
    bl2 = cross_correlation_to_deg2_invariant(cc_inverted,3,{**ccd,'orders':orders,'mode':'legendre_approx','cc_mask_threshold':-1})

    nonzero = np.abs(bl)!=0
    rel_diff = np.abs(bl-bl2)[nonzero]/np.abs(bl[nonzero])
    print('Pseudo inverse ruslan_data: mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
    return locals()
#locals().update(test_bl_legendre())

def init_transforms():
    ccd= db.load('ccd')
    cc_data = ccd['cross_correlation'][...,:800]
    maxQ = np.max(ccd['radial_points'])
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
    return locals()


def generate_density_spheres(grid,radius = 100, norm = 'standard',cart_centers= np.array([[0,200,0],[0,-200,0],[200,0,0],[-200,0,0],[400,0,0],[0,0,300]],dtype = np.float64),random_orientation = False):
    centers = mLib.cartesian_to_spherical(cart_centers)
    #centers = [(100,0,np.pi),(100,np.pi,np.pi),(100,0,0),(200,np.pi,0),(300,np.pi,np.pi/2)]
    sphere_funcs = [mLib.SampleShapeFunctions.get_disk_function(radius,center = center,norm=norm,random_orientation = random_orientation) for center in centers]
    density = sphere_funcs.pop(0)(grid)
    for sphere_func in sphere_funcs:
        density += sphere_func(grid)
    return density

def test_cc_simulated(density, ht, ft, grid,max_order):
    orders = np.arange(max_order+1)
    radial_points = grid.reciprocal[:,0,0,0]
    angular_points = np.arange(1000)/1000*2*np.pi
    cc_data_grid = {'qs':radial_points,"phis":angular_points}    
    ccd= db.load('ccd')
    ccd['data_grid'] = cc_data_grid
    ccd['phis'] = cc_data_grid['phis']
    Bl = density_to_deg2_invariants_3d(density,ht,ft)
    print('bl shape = {}'.format(Bl.shape))
    cc = deg2_invariant_to_cc_3d(Bl,ccd['xray_wavelength'],cc_data_grid,orders)
    print('cc finished')
    Bl2 = cross_correlation_to_deg2_invariant(cc,3,{**ccd,'orders':orders,'mode':'legendre_approx','cc_mask_threshold':-1})
    print('bl2 finished')
    cc2 = deg2_invariant_to_cc_3d(Bl2,ccd['xray_wavelength'],ccd['data_grid'],orders)
    return locals()
#locals().update(init_transforms())
#density = generate_density_spheres(grid_pair.real,radius = 200,norm='inf',cart_centers = np.array([[-200,0,0],[200,0,0]],dtype=np.float64),random_orientation = True)
#locals().update(test_cc_simulated(density,cht,ft,grid_pair,max_order))
#angles = ccd["phis"]<np.pi
#rel_diff = np.abs((cc2-cc)/cc2)[...,angles]
#print('mean = {} std = {}'.format(np.mean(rel_diff),np.std(rel_diff)))
#locals().update(test_cc_bl_pseudo_inverse())

#a=test_cc_bl_pseudo_inverse()
#a=test_cc_bl_legendre()
    
def projection_matrix():
    try:
        ccd= db.load('ccd')
        cc = ccd['cross_correlation']
        sym_cc = i_tools.cross_correlation_enforce_symmetry(cc)
        ccd['cross_correlation'] = sym_cc
        orders = opt.fourier_transform.pos_orders[opt.projections.reciprocal.used_order_ids]
        metadata={**ccd,'orders':orders,'cc_mask_threshold':-1}
        cc_mask = i_tools.cross_correlation_mask(metadata['qs'],metadata['thetas'],metadata['phis'],threshold = metadata['cc_mask_threshold'])
        bl = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,metadata)
        #bl[1::2,...]=0
        p = deg2_invariant_to_projection_matrices(3,bl)
        ReciprocalProjection_old.pre_init_from_ccd(ccd,Multiprocessing.comm_module,{**metadata,'max_order':orders.max(),'dimensions':3})
        p2 = ReciprocalProjection_old.data_projection_matrices

        if ~np.any([(new-old).any() for new,old in zip(p,p2)]):
            print('test passed!')
        
    except Exception as e:
        traceback.print_exc()
    return locals()

#locals().update(projection_matrix())


def projection_matrix_new():
    try:
        ccd= db.load('ccd')
        cc = ccd['cross_correlation']
        sym_cc = i_tools.cross_correlation_enforce_symmetry(cc)
        ccd['cross_correlation'] = sym_cc
        orders = opt.fourier_transform.pos_orders[opt.projections.reciprocal.used_order_ids]
        metadata={**ccd,'orders':orders,'cc_mask_threshold':-1}
        cc_mask = i_tools.cross_correlation_mask(metadata['qs'],metadata['thetas'],metadata['phis'],threshold = metadata['cc_mask_threshold'])
        bl = cross_correlation_to_deg2_invariant(ccd['cross_correlation'],3,metadata)
        #bl[1::2,...]=0
        p = deg2_invariant_to_projection_matrices(3,bl)
        ReciprocalProjection.pre_init_from_ccd(ccd)
        p2 = ReciprocalProjection.data_projection_matrices

        if ~np.any([(new-old).any() for new,old in zip(p,p2)]):
            print('test passed!')
        
    except Exception as e:
        traceback.print_exc()
    return locals()

#locals().update(projection_matrix_new())

def projection_reciprocal_projection():
    try:
        ccd= db.load('ccd')
        cc = ccd['cross_correlation']
        locals().update(init_transforms())
        sym_cc = i_tools.cross_correlation_enforce_symmetry(cc)
        ccd['cross_correlation'] = sym_cc
        ReciprocalProjection.pre_init_from_ccd(ccd)
        p2 = ReciprocalProjection.data_projection_matrices        
    except Exception as e:
        traceback.print_exc()
    return locals()

#locals().update(projection_reciprocal_projection())
#locals().update(init_transforms())
#pr = ReciprocalProjection(grid_pair.reciprocal)

def test_leg_pseudo():
    sys._getframe(1).f_locals.update(init_transforms())
    density = generate_density_spheres(grid_pair.real,radius = 200,norm='inf',cart_centers = np.array([[-200,0,0],[200,0,0]],dtype=np.float64),random_orientation = False)
    Bl = density_to_deg2_invariants_3d(density,cht,ft).real
    
    orders = np.arange(100)
    wavelength = pLib.energy_to_wavelength(10000)*1e10
    qs = grid_pair.reciprocal[:,0,0,0]
    phis = np.arange(1000)*2*np.pi/1000
    data_grid = {'qs':qs,'phis':phis,'thetas':pLib.ewald_sphere_theta_pi(wavelength,qs)}
    cc = deg2_invariant_to_cc_3d(Bl,wavelength,data_grid,orders)
    bl_pse = cross_correlation_to_deg2_invariant(cc,3,{**data_grid,'data_grid':data_grid,"xray_wavelength":wavelength,'orders':orders,'mode':'pseudo_inverse','cc_mask_threshold':-1})
    bl_leg = cross_correlation_to_deg2_invariant(cc,3,{**data_grid,"data_grid":data_grid,"xray_wavelength":wavelength,'orders':orders,'mode':'legendre_approx','cc_mask_threshold':-1})

    bl_nonzero = Bl!=0
    rel_pse = np.abs(Bl-bl_pse)/np.abs(Bl)
    rel_leg = np.abs(Bl-bl_leg)/np.abs(Bl)    
    return locals()


locals().update(test_leg_pseudo())

#a=test_cc_bl_pseudo_inverse()
#a=test_cc_bl_legendre()
    
