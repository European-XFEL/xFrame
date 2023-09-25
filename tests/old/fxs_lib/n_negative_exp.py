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
    '3d_pdb_2YHM_mO99_inverse_binned_psd20_custom'
]
shift = 1
n_points = 40
scales = np.logspace(np.log10(1/(10-shift+1)),np.log10(10+shift-1),n_points) #np.arange(*scan_range)
scales = np.linspace(1,20,n_points)
scales_step = scales[1]-scales[0]
radius = 250.0
name = names[0]

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

def init_2(dataset_name):
    pd = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/reciprocal_proj_data/'+dataset_name+'.h5')
    max_order = pd['max_order']
    p_matrices = [pd['data_projection_matrices'][str(o)] for o in range(max_order +1)]
    p_matrices[0] = p_matrices[0][:,None] 
    qs = pd['data_radial_points']
    nq = len(qs)
    h_dict = {'dimensions':3,'max_order':99,'anti_aliazing_degree':2,'n_phi':False,'n_theta':False}
    cht = HarmonicTransform('complex',h_dict)
    grid = GridFactory.construct_grid('uniform',[qs,cht.grid_param['thetas'],cht.grid_param['phis']])
    integrator = mLib.SphericalIntegrator(grid[:])
    a_int = pd['average_intensity']*2*np.sqrt(np.pi)
    return locals()
#locals().update(init_2(name))

def scan_n_particles_new2():    
    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:p_matrices[i].shape[-1]]=p_matrices[i]

    neg_volume = np.array(tuple( integrator.integrate(cht.inverse([a_int[:,None]/s]+I_lm[1:])<0)  for s in scales))
    
    return locals()

def scan_n_particles_new2():    
    I_lm = [ np.zeros((nq, 2*o+1),dtype=complex) for o in range(max_order+1)]
    for i in range(len(I_lm)):        
        I_l = I_lm[i]
        I_l[:,:p_matrices[i].shape[-1]]=p_matrices[i]
    I = cht.inverse([a_int[:,None]]+I_lm[1:]).real
    imax = I.max()
    I/=imax
    y00=1/(2*np.sqrt(np.pi))
    a = a_int*y00/imax
    d=1
    IN = I[None,...]+(1/scales[:,None,None,None]-1)*a[None,:,None,None]
    gd1 = 2*d*IN*(a[None,:]/scales[:,None]**2)[...,None,None]
    gd2 = np.exp(-d*IN**2)
    g= gd1*gd2
    grads = [integrator.integrate(gi) for gi in g]
    #gd = 2*d*IN*(a[None,:]/scales[:,None]**2)[...,None,None]*np.exp(-d*IN**2)
    #dnIns = 2*d*Ins*a_int[None,:,None]*y00/s[:,None,None]**2*np.exp(-d*Ins**2)
    return locals()
locals().update(scan_n_particles_new2())

#neg = Multiprocessing.comm_module.request_mp_evaluation(scan_n_particles,argArrays = [scales],const_args=[names[0],scan_range,radius,shift],callWithMultipleArguments=True,n_processes = 40,splitMode='modulus')
#q_min_id=0
#neg = Multiprocessing.comm_module.request_mp_evaluation(scan_n_particles_new2,argArrays = [scales],const_args=[name,scan_range,radius,shift,q_min_id],callWithMultipleArguments=True,n_processes = 70,splitMode='modulus')


def print_neg(neg,name,radius,shift):
    expected=np.sqrt(10)
    #expected=shift
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



def init(): 
    thetas = ccd['data_grid']['thetas']
    phis = ccd['data_grid']['phis']
    max_order = 99
    orders = np.arange(max_order+1)
    
    metadata = {**ccd,**proj_opt,'orders':np.arange(max_order+1),'extract_odd_orders':False,'bl_enforce_psd':False}          
    cc_mask = cross_correlation_mask(ccd['data_grid'],metadata)
    #cc_mask = np.full(cc.shape,True)
    new_cc_1,new_cc_mask_1,new_phis_1 = binned_mean(np.full(cc_mask.shape,True),cc_1,max_order,phis)
    new_cc_1 = interpolate(new_cc_1,new_cc_mask_1,new_phis_1)
    cns_1 = mLib.circularHarmonicTransform_real_forward(new_cc_1)
    new_cc,new_cc_mask,new_phis = binned_mean(cc_mask,cc,max_order,phis)
    new_cc = interpolate(new_cc,new_cc_mask,new_phis)
    cns = mLib.circularHarmonicTransform_real_forward(new_cc)
    #pl,ls,ms = mLib.gsl.legendre_sphPlm_array(max_order,max_m,np.cos(thetas),return_orders = True,sorted_by_l = True)
    qq_matrix = ccd_associated_legendre_matrices(ccd['data_grid']['thetas'],max_order,max_order)
    #qq_matrix = q_matrix[None,:]*q_matrix[:,None]/(2*orders+1)[None,None,None,:]
    inverses = np.linalg.inv(qq_matrix[...,2::2,2::2])
    q_ids = np.arange(len(thetas))
    
    bls = np.zeros_like(cns)
    bls[...,2::2] = np.sum(inverses*cns[...,None,2:max_order+1:2],axis =-1)
    bls[...,1:]/=metadata['n_particles']
    bls[...,0]/=metadata['n_particles']**2
    #bls_pseudo = np.swapaxes(cross_correlation_to_deg2_invariant(cc,3,**metadata)[0],0,-1)
    #Ms = qq_matrix[...,2::2,2::2]
    #Cs = cns[...,2:max_order+1:2]
    #bls[...,2::2] = np.array(tuple(solve_triangular(m,v) for m,v in zip(Ms.reshape(-1,*Ms.shape[-2:]),Cs.reshape(-1,*Cs.shape[-1:])) )).reshape(*Cs.shape)
    #bls_t = np.array(tuple(mLib.optimal_tikhonov_regularization(qq_matrix[i],cns[i],allow_offset=True)[0] for i in range(len(bls))))
    #bls_t = Multiprocessing.comm_module.request_mp_evaluation(mLib.optimal_tikhonov_regularization_worker,argArrays = [q_ids,q_ids],const_args=[qq_matrix,cns[...,:max_order+1].astype(complex),True],callWithMultipleArguments=True)
    cns_new = np.zeros_like(cns)
    #cns_new[...,2::2] = np.sum(qq_matrix[...,2::2,2::2]*bls[...,None,2:max_order+1:2],axis =-1)
    #cns/=10
    return locals()

#locals().update(init())

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
