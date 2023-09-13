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
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward,circularHarmonicTransform_complex_inverse,circularHarmonicTransform_real_forward,circularHarmonicTransform_real_inverse
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
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi,plot1D

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
    metadata = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':1,'extract_odd_orders':False,'bl_enforce_psd':False}
    #cc_1,_locals = cross_correlation_low_pass(cc_1,max_order)
    qs = ccd['data_grid']['qs']
    #bl_single = cross_correlation_to_deg2_invariant(cc_1,3,**metadata)[0]
    #bl_single = nearest_positive_semidefinite_matrix(bl_single)

    low_pass_order = 120
    #sos = signal.butter(1, low_pass_order, 'lp', fs=128, output='sos')
    
    metadata_100 = {**ccd,'orders':np.arange(max_order+1),'mask':proj_opt.mask,'n_particles':100,'extract_odd_orders':False,'bl_enforce_psd':False}
    n_repeats = 1
    c_orders = np.concatenate((np.arange(cc.shape[-1]//2+1),-np.arange(1,cc.shape[-1]//2 + cc.shape[-1]%2)))
    c_odd_order_mask = c_orders%2
    new_cc = np.copy(cc)
    bls = []
    errors = []
    n_qs = cc.shape[0]
    n_phis = cc.shape[-1]
    periodic_cc=np.zeros((n_qs*2,n_qs*2,n_phis),dtype = cc.dtype)
    try:
        for n in range(n_repeats):
            cc_backup = np.copy(new_cc)
            cc_nonzero = cc_backup !=0
            if n < 0:
                periodic_cc[:n_qs,:n_qs]=new_cc
                periodic_cc[n_qs:,:n_qs]=new_cc[::-1,...]
                periodic_cc[:n_qs,n_qs:]=new_cc[:,::-1,:]
                periodic_cc[n_qs:,n_qs:]=new_cc[::-1,::-1,:]
                coeff_q1 = np.fft.rfft(periodic_cc,axis = 0)
                coeff_q1[-1]=0
                coeff_q1[-2]=0
                periodic_cc = np.fft.irfft(coeff_q1,n_qs*2,axis = 0)
                coeff_q2 = np.fft.rfft(periodic_cc,axis = 1)
                #coeff_q2[:,-2:]=0
                periodic_cc = np.fft.irfft(coeff_q2,n_qs*2,axis = 1)
                new_cc = periodic_cc[:n_qs,:n_qs,:n_phis]
                #new_cc = signal.sosfilt(sos,new_cc,axis = 0)
                #new_cc = signal.sosfilt(sos,new_cc,axis = 1)
            else:                
                pass
            cc_coeff = circularHarmonicTransform_real_forward(new_cc)        
            #cc_coeff[...,max_order+1:-max_order] = 0
            cc_coeff[...,max_order+1:] = 0
            #cc_coeff[...,c_odd_order_mask] = 0
            cc_coeff[...,1::2] = 0        
            new_cc = circularHarmonicTransform_real_inverse(cc_coeff,new_cc.shape[-1])
            cc_error = np.mean(np.abs((cc_backup - new_cc)[cc_nonzero])**2/np.abs(cc_backup[cc_nonzero])**2)
            print('cc_error = {}'.format(cc_error))

            if n == 0:
                bl = cross_correlation_to_deg2_invariant(new_cc,3,**metadata_100)[0]
            else:
                bl = cross_correlation_to_deg2_invariant(new_cc,3,**metadata)[0]                
            bl_backup = np.copy(bl)
            bl_nonzero = bl_backup !=0

            if n < -1:
                bl = signal.sosfilt(sos,bl,axis = -1)
                bl = signal.sosfilt(sos,bl,axis = -2)
            else:                
                pass
            bl = nearest_positive_semidefinite_matrix(bl)
            bl_error = np.mean(np.abs((bl_backup - bl)[bl_nonzero])**2/np.abs(bl_backup[bl_nonzero])**2)
            print('bl_error = {}'.format(bl_error))
            new_cc = deg2_invariant_to_cc_3d(bl,ccd['xray_wavelength'],ccd['data_grid'],np.arange(max_order+1))
            bls.append(bl)
            errors.append((cc_error,bl_error))
    except Exception as e :
        traceback.print_exc()
        print(e)
        bls = np.array(bls)
        errors = np.array(errors).T
    return locals()

locals().update(init())

def plot(orders):
    bl= np.abs(bls)
    error = errors
    datasets = [bl[0,orders],bl[0,orders],bl[0,orders]]

    n_orders = len(orders)
    
    layout_1 = {'title':'S','0 iteration':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_100 = {'title':'20 iteration','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout_A = {'title':'40 itertion','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    #layout_A2 = {'title':'M l-cc-phi psd l-bl-q','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    #layout_A3 = {'title':'M  l-cc-phi l-bl-q psd','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    #layout_A4 = {'title':'S psd','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':30}
    layout = tuple(zip(*[[layout_1]*(n_orders+1),
                         [layout_100]*(n_orders+1),
                         [layout_A]*(n_orders+1),
                         #[layout_A2]*(n_orders+1),
                         #[layout_A3]*(n_orders+1),
                         #[layout_A4]*(n_orders+1),
                         ]))

    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/bl_loop_comparison.matplotlib',heat2D_multi.get_fig(tuple(zip(*datasets)),layout=layout, grid = grid,scale = 'log',shape = [n_orders,len(datasets)],size = (10*(len(datasets)+1),10*(n_orders)),vmin= 1e14, vmax = 1e24,cmap = 'plasma'))
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/bl_loop_errors.matplotlib',plot1D.get_fig(np.abs(error),y_scale = 'log'))

ccd = db.load('ccd')
cc = ccd['cross_correlation']
grid = GridFactory.construct_grid('uniform',[ccd['qs'],ccd['qs']])

db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/bl_loop_comparison_2.h5',{'bl':np.array(bls),'errors':np.array(errors).T})
data = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/test/cc_remove_zero_component/bl_loop_comparison_2.h5')
bls = data['bl']
errors = data['errors']
orders = [2,4,6,8,10]    
plot(orders)


