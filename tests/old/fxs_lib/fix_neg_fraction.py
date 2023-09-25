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
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import ReciprocalProjection
from xframe import settings
from xframe.library import mathLibrary as mLib
from xframe import Multiprocessing
from xframe.library import physicsLibrary as pLib
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi,plot1D
from xframe import Multiprocessing
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular
from scipy.stats import unitary_group

log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
from xframe.library.mathLibrary import gsl
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_10p_spher')[0] 
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
controller.chooseJob()
ccd_1 = db.load('ccd')


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
    return ft,ift,cht,grid_pair

#locals().update(init_transforms())

#projd = db.load('reciprocal_proj_data',path_modifiers={'name':'3d_model_0','max_order':str(99)})

#p_matrices = list(projd['data_projection_matrices'])
#for order in range(projd['max_order']+1):
#    shape = p_matrices[order].shape
#    if 2*order+1 > shape[1]:
#        tmp = np.zeros((shape[0],2*order+1))
#        tmp[:,:shape[1]] = p_matrices[order]
#        p_matrices[order] = tmp 
scan_range = [1,20.1,0.5]
names = [
    '3d_pentagon_100_mO99_inverse_binned_psd20_arc_wrong_n_particles',
    '3d_pentagon_10p_spher_mO99_inverse_binned_psd20_arc_wrong_n_particles',
    '3d_pentagon_10p_spher_mO99_inverse_binned_psd20_custom_wrong_n_particles',
    '3d_model_0_mO99_cn_inverse_not_binned_psd20_custom',
    '3d_pentagon_10p_spher_mO99_inverse_binned_psd20_custom',
    '3d_pentagon_10p_noise_mO99_inverse_arc_no-psd',
    '3d_pentagon_Unknown_p_noise_mO99_inverse_arc_no-psd',
    '3d_model_3_nosym_mO99_psd_eigh'
]
cc_names = [
    'pentagon_100_particles',
    '3d_pentagon_10particles_spherSim',
    '3d_pentagon_10particles_spherSim',
    '3d_model_0',
    '3d_pentagon_10particles_spherSim',
    '3d_pentagon_10particles_noise',
    '3d_pentagon_?particles_noise',
    '3d_model_3'
]
shift = 0
n_points = 40
scales = np.logspace(np.log10(1/(10-shift+1)),np.log10(10+shift-1),n_points) #np.arange(*scan_range)
scales = np.linspace(1,10,n_points)
scales_step = scales[1]-scales[0]
radius = 250.0
select  = 3
name = names[select]
cc_name = cc_names[select]

reconstrution_path = '11_3_2022/run_4/reconstruction_data.h5'
maxwell_path = db.get_path('maxwell_base',is_file=False)
path = maxwell_path + 'reconstructions/' + '12_11_2021/run_1/reconstruction_data.h5'
data = db.load(path)
r_density = data['reconstruction_results']['14']['reciprocal_density']
Intensity = np.abs(r_density)**2

r_grid = data['configuration']['internal_grid']['reciprocal_grid'][:]
grid = data['configuration']['internal_grid']['real_grid'][:]
def init_2():
    pd = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/reciprocal_proj_data/'+name+'.h5')
    ccd = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/'+cc_name+'.h5')
    max_order = pd['max_order']
    p_matrices = [pd['data_projection_matrices'][str(o)] for o in range(max_order +1)]
    p_matrices[0] = p_matrices[0][:,None] 
    qs = pd['data_radial_points']
    nq = len(qs)
    cns = i_tools.deg2_invariant_to_cn_3d(pd['deg_2_invariant'],qs,pd['xray_wavelength'])
    log.info('finished')
    bl = np.moveaxis(pd['deg_2_invariant'],0,-1)
    cc = ccd['intra']['ccf_2p_q1q2']
    hcc = ccd['intra']['fc_2p_q1q2']
    
    aint = pd['average_intensity']
    
    ht_opt={'dimensions':3,'max_order':max_order,'indices':'lm','anti_aliazing_degree':2,'n_phi':0,'n_theta':0}
    cht=HarmonicTransform('complex',ht_opt)

    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:p_matrices[i].shape[-1]]=p_matrices[i]
    I00 = np.abs(p_matrices[0].flatten()).real
    I = cht.inverse([I00[:,None]]+I_lm[1:]).real
    return locals()
locals().update(init_2())
I_lm = cht.forward(Intensity)
I_lm[0]*=10
I_lm[1:]=[Ilm*np.sqrt(10) for Ilm in I_lm[1:]]
#g = GridFactory.construct_grid('uniform',[np.arange(128),np.linspace(0,np.pi,128),np.linspace(0,2*np.pi,256)])
scan_space = (1,5,100)
average_intensity = aint
n_orders = np.arange(5,90,5)
n_particles, grads, neg_volume_fraction = i_tools.estimate_number_of_particles(I_lm,r_grid[:,0,0,0],scan_space,average_intensity = False,n_orders=n_orders)


def plot():
    labels = [str(o)for o in n_orders]
    scales = np.linspace(*scan_space)
    #log.info('num particles is approximatelly = {}'.format(scales[np.argmax(abs(grads))]**2))
    layout = {'title':'Gradient of the Volume of negative Intensity \n Number of particles $= {}$ '.format('?'),'x_label':'square root of the number of particles $\sqrt{x}$ as in $I_{0,0}/\sqrt{x}$', 'y_label':'Gradient of Volume of negative Intensity','text_size':10}
    fig = plot1D.get_fig(neg_volume_fraction,grid = scales,x_scale='lin',layout = layout,labels = labels)
    ax = fig.get_axes()[0]
    #ax.vlines(np.sqrt(n_particles),grads.min(),grads.max())
    path = db.get_path('reciprocal_proj_data',path_modifiers={'name':settings.analysis.name,'max_order':99,})
    new_path = os.path.dirname(path)+'/' + os.path.basename(path).split('.')[0] + '_n_particles_aint.matplotlib'
    db.save(new_path,fig,dpi=400)

    labels = [str(o)for o in n_orders]
    scales = np.linspace(*scan_space)
    #log.info('num particles is approximatelly = {}'.format(scales[np.argmax(abs(grads))]**2))
    layout = {'title':'Gradient of the Volume of negative Intensity \n Number of particles $= {}$ '.format('?'),'x_label':'square root of the number of particles $\sqrt{x}$ as in $I_{0,0}/\sqrt{x}$', 'y_label':'Gradient of Volume of negative Intensity','text_size':10}
    fig = plot1D.get_fig(np.gradient(grads,axis = 1),grid = scales,x_scale='lin',layout = layout,labels = labels)
    ax = fig.get_axes()[0]
    #ax.vlines(np.sqrt(n_particles),grads.min(),grads.max())
    path = db.get_path('reciprocal_proj_data',path_modifiers={'name':settings.analysis.name,'max_order':99,})
    new_path = os.path.dirname(path)+'/' + os.path.basename(path).split('.')[0] + '_n_particles_inflection_aint.matplotlib'
    db.save(new_path,fig,dpi=400)
