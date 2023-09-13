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
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_donatelli_n_particles_test_')[0]
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_10p_spher')[0] 
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
controller.chooseJob()
ccd_1 = db.load('ccd')


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
    '3d_pdb_2YHM_mO99_inverse_binned_psd20_custom'
]
name = names[1]

def init():
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
    estimate_N = mtip.process_factory.operatorDict['estimate_number_of_particles']
    approx_unknowns = mtip.process_factory.operatorDict['approximate_unknowns']
    mtip_projection = mtip.process_factory.operatorDict['mtip_projection']
    update_n_particles = mtip.process_factory.operatorDict['update_number_of_particles']
    return locals()
#locals().update(init())

density_guess = np.zeros(grid_pair.realGrid.shape,dtype = float)
density_guess[:40] = 1
density_guess[:40] += 0.01*np.random.rand(40,*density_guess.shape[1:])
r_guess = ft(density_guess)
I_guess = (r_guess*r_guess.conj()).real
Ilm_guess = cht(I_guess)
N = estimate_N(Ilm_guess)
print(N)
u = approx_unknowns(Ilm_guess)
Ilm_new = mtip_projection(Ilm_guess,u)
update_n_particles(N)
u2 = approx_unknowns(Ilm_guess)
Ilm_new2 = mtip_projection(Ilm_guess,u2)

I_new=icht(Ilm_new)
I_new2=icht(Ilm_new2)




def print_neg(neg,name,radius,shift):
    expected=np.sqrt(10)
    grid = scales
    layout = {'title':'{} \n Assumed particle radius ={}'.format(name.replace('_','\_'),radius),'x_label':'scaling factor', 'y_label':'\% of negative pixels in I'}#,))*len(orders)
    neg_model = np.exp(-1/(grid)**1.12*11.3)*0.07098
    fig = plot1D.get_fig(neg,grid = grid,x_scale='lin',layout = layout)
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/n_particles/{}_rad_{}_new3.matplotlib'.format(name,radius),fig)
    layout = {'title':'{} \n Assumed particle radius ={}'.format(name.replace('_','\_'),radius),'x_label':'scaling factor', 'y_label':' diff '}#,))*len(orders)
    #diff = (neg[:-2]-neg[2:])/(scales[:-2]-scales[2:])
    acc = 8
    #diff = mLib.second_derivative(neg,scales_step,acc)
    diff = np.gradient(neg,scales_step)
    diff_model = np.gradient(neg_model,scales_step)
    fig = plot1D.get_fig(diff,grid = scales[:],x_scale='lin',layout = layout)
    ax = fig.get_axes()[0]
    ax.vlines(expected,diff.min(),diff.max())
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/n_particles/{}_rad_{}_diff_new4.matplotlib'.format(name,radius),fig)
    
    diff2 = np.gradient(diff,scales_step)
    diff2_model = np.gradient(diff_model,scales_step)
    #fig = plot1D.get_fig(diff,grid = grid[acc//2:-acc//2],x_scale='log',layout = layout)
    fig = plot1D.get_fig(diff2,grid = scales[:],x_scale='lin',layout = layout)
    ax = fig.get_axes()[0]
    ax.vlines(expected,diff.min(),diff.max())
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/n_particles/{}_rad_{}_diff2_new4.matplotlib'.format(name,radius),fig)

#print_neg(neg,name,radius,shift)

def plot():
    qs = ccd['data_grid']['qs']
    qq_grid = GridFactory.construct_grid('uniform',[qs,qs])
    orders = np.arange(2,12,2)
    orders2 = np.arange(12,22,2)
    bls_plot = np.swapaxes(bls,0,-1)#/100**2
    bls_plot[0]#/=100
    cns_plot = np.swapaxes(cns,0,-1)#/100**2
    cns_plot_1 = np.swapaxes(cns_1,0,-1)#/100**2
    cns_new_plot = np.swapaxes(cns_new,0,-1)#/100**2
    cns_diff = np.abs(cns_plot - cns_new_plot)/np.abs(cns_plot)
    cns_diff_1_100 = np.abs(cns_plot_1-cns_plot)/np.abs(cns_plot_1)
    bls_plot_psd = np.zeros_like(bls_plot)
    q_start = 25
    bls_plot_psd[:,q_start:,q_start:] = mLib.nearest_positive_semidefinite_matrix(bls_plot[:,q_start:,q_start:])
    layout = {'title':'','x_label':'$q_1$', 'y_label':'$q_2$','title_size':40,'text_size':20}#,))*len(orders)
    #bls_plot[:,30:,30:] = mLib.nearest_positive_semidefinite_matrix(bls_plot[:,30:,30:])
    fig = heat2D_multi.get_fig((
        list(np.abs(cns_plot_1[orders])),
        list(np.abs(bls_plot[orders])),
        list(np.abs(bls_plot_psd[orders]))
    ),grid = qq_grid,shape = (3,len(orders)),size=(5*len(orders)+2*5,3*5),vmin = 1e+14, vmax = 1e+24,scale='log',cmap='plasma',layout = layout)
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl0-10_10p_mO70_rect.matplotlib',fig)
    db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/bl0-20_10p_mO99_rect.matplotlib',fig)
    #fig = heat2D_multi.get_fig((
    #    list(np.abs(cns_diff_1_100[orders])[:,1:,1:])
    #    ,),grid = qq_grid[1:,1:],shape = (1,len(orders)),size=(5*len(orders)+2*5,1*5),cmap='plasma')
    fig = heat2D_multi.get_fig((list(np.abs(cns_diff_1_100[orders])),list(np.abs(cns_diff_1_100[orders2]))),grid = qq_grid,shape = (2,len(orders)),size=(5*len(orders)+2*5,2*5),vmin = 1e-2, vmax = 1e2,scale='log',cmap='plasma',layout = layout)
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/cn0-10_p1_p10_mO30_diff_rect.matplotlib',fig)
    #db.save('/gpfs/exfel/theory_group/user/berberic/MTIP/test/bl_from_cm/cn0-20_p1_p10_mO99_diff_spher.matplotlib',fig)
    #fig = heat2D_multi.get_fig((
    #    list(np.abs(cns_diff[orders]))
    #),grid = qq_grid,shape = (1,len(orders)),size=(5*len(orders)+2*5,1*5),scale='log',cmap='plasma')    

#plot()
