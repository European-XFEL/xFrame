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
select  = -3
name = names[select]
cc_name = cc_names[select]

def scan_n_particles(scales,dataset_name,scan_range,radius,shift,**kwargs):
    mtip = controller.job.mtip
    mtip.generate_phasing_loop()
    grid_pair = mtip.grid_pair
    r = mtip.grid_pair.realGrid[:,0,0,0]
    q = mtip.grid_pair.reciprocalGrid[:,0,0,0]
    #ft,ift,cht,grid_pair = init_transforms()
    #projd = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/reciprocal_proj_data/'+dataset_name+'.h5')
    p_matrices = mtip.rprojection.projection_matrices #list(projd['data_projection_matrices'][str(i)] for i in range(projd['max_order']+1))
    #p_matrices[0]= p_matrices[0].reshape(-1,1)
    max_order = len(p_matrices)-1
    ft =mtip.process_factory.operatorDict['fourier_transform']
    cht = mtip.process_factory.operatorDict['harmonic_transform']
    icht = mtip.process_factory.operatorDict['inverse_harmonic_transform']

    spherical_formfactor = pLib.spherical_formfactor(q,radius=radius)
    neg = []
    n_pixel_in_radius = np.sum(r<=radius)
    base_guess = np.ones(grid_pair.realGrid.shape,dtype = float)
    base_guess[r>radius]=0
    I_base_guess = np.zeros_like(base_guess)
    I_base_guess[:] = spherical_formfactor[:,None,None]
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    for repeats in range(1):        
        neg_fractions = []
#        log.info('repeat count = {}'.format(repeats))
        for scale in scales:
            #log.info('creating density')
            #density_guess = np.array(base_guess)
            #density_guess[:n_pixel_in_radius]+= 0.1*np.random.rand(n_pixel_in_radius,*base_guess.shape[1:])
            #log.info('finished density of shape = {}'.format(density_guess.shape))
            #I_guess = np.abs(ft(density_guess))**2
            I_guess = np.array(I_base_guess)
            I_lm_guess = cht(I_guess)
            I_lm_guess[1:] = [ np.zeros_like(i) for i in I_lm_guess[1:]] 
            unknowns = mtip.process_factory.operatorDict['approximate_unknowns'](I_lm_guess)
            #log.info(unknowns[-1].shape)
            #I_lm = [ np.sum(p_matrices[order][:,:,None] * unknowns[order][None,:,:],axis = 1) for order in range(max_order+1)]
            I_lm = [ np.zeros_like(i) for i in I_lm_guess]
            for i in range(len(I_lm)):
                p_matrix = p_matrices[i]
                I_l = I_lm[i]
                I_l[:,:p_matrix.shape[-1]]=p_matrix
                
            I_lm[1:] = [lm/shift for lm in I_lm[1:]]
            I_lm[0]/=scale
            I = icht(I_lm)
            neg_fractions.append(np.sum(I.real<0)/np.prod(I.shape))
        neg.append(np.array(neg_fractions))
    return np.mean(neg,axis = 0)

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
    
    return locals()
locals().update(init_2())

thetas = pLib.ewald_sphere_theta(pd['xray_wavelength'], pd['data_radial_points'])
qq_matrix = np.squeeze(i_tools.ccd_associated_legendre_matrices(thetas,99,0))

bl_new = bl.copy()
scales = np.linspace(1,100,200)
bl0 = aint[:,None]*aint[None,:]*4*np.pi
cc0 =(qq_matrix[...,0]*bl[...,0])[...,None]/scales[None,None,:]**2+np.sum(qq_matrix[...,1:]*bl_new[...,1:],axis = -1)[:,:,None]
#cc0 =(qq_matrix[...,0]*bl0)[...,None]/np.sqrt(scales[None,None,:])+np.sum(qq_matrix[...,1:]*bl_new[...,1:],axis = -1)[:,:,None]
#cc0 = np.moveaxis(cc0,-1,0)[:,20:,20:]
qs=pd['data_radial_points'][20:]

neg_sum = []
temp = np.zeros_like(cns[20:,20:])
temp[...,:100] = cns[20:,20:]
tot = len(qs)**2 * cc.shape[-1]
for s in scales:
    #temp[...,0]= (qq_matrix[...,0]*pd['deg_2_invariant'][0]/s**2+np.sum(qq_matrix[...,1:]* np.moveaxis(pd['deg_2_invariant'],0,-1)[...,1:],axis = -1))[20:,20:]
    #temp[...,0]= (qq_matrix[...,0]*pd['deg_2_invariant'][0]/s**2+2*np.sum(qq_matrix[...,1:]* np.moveaxis(pd['deg_2_invariant'],0,-1)[...,1:],axis = -1))[20:,20:]
    temp[...,0]= (qq_matrix[...,0]*bl0/s**2+8*np.sum(qq_matrix[...,1:]* np.moveaxis(pd['deg_2_invariant'],0,-1)[...,1:],axis = -1))[20:,20:]
    cc = np.fft.irfft(temp,1600) #mLib.circularHarmonicTransform_real_inverse(temp,200)
    temp[...,1:]=cns[20:,20:,1:]*8
    neg_sum.append(np.sum(cc<0)/tot)

#cc0 = np.moveaxis(cc0,-1,0)[:,20:,20:]
#qs= qs[20:]
neg_mask = cc0<0
#neg_sum = np.array([ np.sum(-1*cc0[s,neg_mask[s]]*((qs**2)[:,None]*(qs**2)[None,:])[neg_mask[s]]) for s in range(len(scales))])
#neg_sum = np.sum(neg_mask*((qs**2)[:,None]*(qs**2)[None,:])[None,...],axis = (1,2))
#neg_sum = np.sum(neg_mask[20:,20:],axis = (0,1))
neg_sum = np.array(neg_sum)
#neg_sum = np.sum(neg_mask,axis=(0,1))
grad = np.gradient(neg_sum,scales[1]-scales[0])

pl_name = 'bl_c0_test'
def print_neg():
    expected=100
    #expected=shift
    grid = scales
    layout = {'title':'{} \n Assumed particle radius ={}'.format(name.replace('_','\_'),radius),'x_label':'scaling factor', 'y_label':'\% of negative pixels in I'}#,))*len(orders)
    fig = plot1D.get_fig(neg_sum,grid = grid,x_scale='lin',layout = layout)
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/n_particles/{}_new3.matplotlib'.format(pl_name),fig)
    layout = {'title':'{} \n Assumed particle radius ={}'.format(name.replace('_','\_'),radius),'x_label':'scaling factor', 'y_label':' diff '}#,))*len(orders)
    #diff = (neg[:-2]-neg[2:])/(scales[:-2]-scales[2:])
    fig = plot1D.get_fig(grad,grid = scales[:],x_scale='lin',layout = layout)
    ax = fig.get_axes()[0]
    #ax.vlines(expected,grad.min(),grad.max())
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/n_particles/{}_diff_new4.matplotlib'.format(pl_name),fig)
print_neg()
