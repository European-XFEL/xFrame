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
from xframe.library.mathLibrary import nearest_positive_semidefinite_matrix
from scipy import signal

log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_100')[0] 
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi

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


def init():
    ccd = db.load('ccd')
    ccd_1 = db.load_ccd('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_model_0.h5')
    #sys._getframe(1).f_locals.update(init_transforms())
    #density = generate_density_spheres(grid_pair.real,radius = 200,norm='inf',cart_centers = np.array([[-200,0,0],[200,0,0]],dtype=np.float64),random_orientation = True)
    #density += generate_density_spheres(grid_pair.real,radius = 100,norm='inf',cart_centers = np.array([[0,300,0],[0,-100,0]],dtype=np.float64),random_orientation = False)
    #Bl = density_to_deg2_invariants_3d(density,cht,ft).real
    cc = ccd['cross_correlation']
    cc_1 = ccd_1['cross_correlation']
    grid = GridFactory.construct_grid('uniform',[ccd['qs'],ccd['qs']])

    max_order = 99
    proj_opt = settings.analysis.projections.reciprocal.pre_init.cross_correlation
    metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':1,'extract_odd_orders':False}
    #cc_1,_locals = cross_correlation_low_pass(cc_1,max_order)
    bl_single = cross_correlation_to_deg2_invariant(cc_1,3,metadata)[0]
    
    #new_cc,_locals = cross_correlation_low_pass(cc,max_order)
    new_cc,_locals = cross_correlation_low_pass(cc,99)
    #sos_cc = signal.butter(1, 700, 'lp', fs=1600, output='sos')
    #new_cc = signal.sosfilt(sos_cc,cc,axis = -1)
    #new_cc = cc
    
    cc_3 = np.copy(new_cc)
    low_pass_order = 30
    sos = signal.butter(1, low_pass_order, 'lp', fs=128, output='sos')
    new_cc = signal.sosfilt(sos,new_cc,axis = 0)
    new_cc = signal.sosfilt(sos,new_cc,axis = 1)
    #new_cc = cc
    
    metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':proj_opt.n_particles,'extract_odd_orders':False}
    metadata1 = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':1,'extract_odd_orders':False}
    bl_new = cross_correlation_to_deg2_invariant(new_cc,3,metadata)[0]
    bl3 = cross_correlation_to_deg2_invariant(cc_3,3,metadata)[0]

    
    A=nearest_positive_semidefinite_matrix(bl_new)
    AA=nearest_positive_semidefinite_matrix(bl3)
    A2 = signal.sosfilt(sos, AA,axis = -1)
    A2 = signal.sosfilt(sos, A2,axis = -2)
    bl_low_pass = signal.sosfilt(sos, bl3,axis = -1)
    bl_low_pass = signal.sosfilt(sos, bl3,axis = -2) 
    A3 = nearest_positive_semidefinite_matrix(bl_low_pass)
    #A4 = signal.sosfilt(sos, bl_single,axis = -1)
    #A4 = signal.sosfilt(sos, A4,axis = -2)
    A4 = nearest_positive_semidefinite_matrix(bl_single)

    

    #A2 = nearest_positive_semidefinite_matrix(
    #    cross_correlation_to_deg2_invariant(
    #        deg2_invariant_to_cc_3d(A,ccd['xray_wavelength'],ccd['data_grid'],np.arange(100))
    #        ,3,metadata1)[0]
    #)
    return locals()

#locals().update(init())

def plot(orders):
    b1 = tuple(np.abs(bl_single[orders]))
    b100 = tuple(np.abs(bl_new[orders]))
    bA = tuple(np.abs(A[orders]))
    bA2 = tuple(np.abs(A2[orders]))
    bA3 = tuple(np.abs(A3[orders]))
    bA4 = tuple(np.abs(A4[orders]))
    datasets = [b1,b100,bA,bA2,bA3,bA4]

    n_orders = len(orders)
    
    layout_1 = {'title':'S','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_100 = {'title':'M l-cc-phi-q','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_A = {'title':'M l-cc-phi-q psd','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_A2 = {'title':'M l-cc-phi psd l-bl-q','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_A3 = {'title':'M  l-cc-phi l-bl-q psd','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_A4 = {'title':'S psd','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout = tuple(zip(*[[layout_1]*(n_orders+1),
                         [layout_100]*(n_orders+1),
                         [layout_A]*(n_orders+1),
                         [layout_A2]*(n_orders+1),
                         [layout_A3]*(n_orders+1),
                         [layout_A4]*(n_orders+1),
                         ]))

    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/b0_comparison_psd.matplotlib',heat2D_multi.get_fig(tuple(zip(*datasets)),layout=layout, grid = grid,scale = 'log',shape = [n_orders,len(datasets)],size = (10*(len(datasets)+1),10*(n_orders)),vmin= 1e14, vmax = 1e24,cmap = 'plasma'))

grid = GridFactory.construct_grid('uniform',[ccd['qs'],ccd['qs']])    
orders = [2,4,6,8]
#plot(orders)

A = np.random.rand(4,10,10)
AA = A[2]

B = (A + np.swapaxes(A,-1,-2)) / 2
l,v = np.linalg.eigh(B)
l[l<0]=0
A2 =v*l[...,None,:] @ np.swapaxes(v,-1,-2)

B2 = (AA + AA.T) / 2
l2,v2 = np.linalg.eigh(B2)
l2[l2<0]=0
A3 =v2 @ np.diag(l2) @ v2.T

