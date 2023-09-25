import numpy as np
import os
import glob
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward,cartesian_to_spherical,spherical_to_cartesian
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward,masked_mean
from xframe.presenters.matplolibPresenter import plot1D,hist2D,heat2D,scatter1D



import logging
#from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import SimpleRegridder2D,AgipdRegridderSimple

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter

log=logging.getLogger('root')

init = False
if init:
    from xframe.startup_routines import dependency_injection
    dependency_injection()
    from xframe.startup_routines import load_recipes
    from xframe.control.Control import Controller
    analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='saxs_65_-bg',exp_name='fxs3046_online',exp_settings = 'default_proc')
    controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
    controller.chooseJobSingleProcess()
    from xframe import database
    from xframe import Multiprocessing
    experiment = Multiprocessing.comm_module.get_experiment()
    aw = controller.controlWorker.analysis_worker
    aw.geometry = aw.load_geometry()
    aw.agipd_regridder = False
    r = controller.controlWorker.analysis_worker.set_and_get_agipd_regridder()

db = default_DB()
#base_path = '/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_65/'
base_path = '/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_140/'
base_path = '/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_{run}/'
#time = '2022-09-03_19:22:42/bg_data_proc.h5'
#time = '2022-09-03_20:15:09/bg_data_proc.h5'
#time = '2022-09-03_21:53:26/bg_data_proc.h5' #run 65 100k
#time = '2022-09-03_23:49:52/bg_data_proc.h5'
#time = '2022-09-04_00:11:48/bg_data_proc.h5' # run 140 100k 0.1,0.3
#time = '2022-09-04_00:11:49/bg_data_proc.h5' # run 140 100k 1.9-2.1
#time = '2022-09-04_12:05:46/pump_diff_run_140_random.h5' #run 140 10k random
#time = '2022-09-04_13:20:07/pump_diff_run_140_random.h5' #run 140 100k random
time = '2022-09-04_10:54:36/bg_data_proc.h5' #run 140 all patterns 
#d = db.load(base_path +time+'/'+ 'bg_data_proc.h5')
time = '2022-09-04_21:04:54/pump_diff_run_140_random.h5' #run 168 all patterns 
d = db.load(base_path.format(run = 168) +time)

d['polar_grid']=r.new_grid
db.save(base_path +time,d)

patterns = d['diod_diff']['saxs_patterns']
#qs = np.arange(512,dtype = float) #d['diod_diff']['polar_grid'][:,0,0]
qs = r.new_grid[:,0,0]

q_mask_max = qs>0.15
q_mask_min = qs<2.5
q_mask1 = q_mask_min & q_mask_max

q_mask_max = qs>0.15
q_mask_min = qs<0.5
q_mask2 = q_mask_min & q_mask_max

q_mask_max = qs > 1.9
q_mask_min = qs < 2.5
q_mask3 = q_mask_max & q_mask_min

q_mask_max = qs > 0.12
q_mask_min = qs < 2
q_mask4 = q_mask_max & q_mask_min


#q_mask_max = qs > 0.12
q_mask_min = qs < 2
q_mask5 =  q_mask_min


patterns_c = []
cells = np.array(tuple( int(c) for c in patterns))
cell_sort = np.argsort(cells)
sorted_cells = cells[cell_sort]
cell_list = []
for key in sorted_cells:
    patterns_c += list(patterns[str(key)]['saxs'])
    #u,s,vh = np.linalg.svd(patterns[str(key)]['saxs'][:,q_mask4],full_matrices=False,compute_uv = True)
    #fig = plot1D.get_fig(value['saxs'],grid = qs)
    #db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/c_{}.matplotlib'.format(key),fig)
    cell_list.append([key]*len(patterns[str(key)]['saxs']))
patterns_c = np.array(patterns_c)
u,s,vh = np.linalg.svd(patterns_c[:,q_mask4],full_matrices=False,compute_uv = True)
cell_ids = np.concatenate(cell_list)

cell_wise_mean_weight_components = []
cell_wise_saxs_no_0 = []
cell_wise_saxs = []
for i in sorted_cells:
    cell_mask = np.nonzero(cell_ids==i)
    mean_weight_components = np.mean(u[cell_mask],axis = 0)
    cell_wise_mean_weight_components.append(mean_weight_components)
    saxs_no_0 = np.mean((u[:,1:]*s[1:]@vh[1:])[cell_mask],axis = 0)
    saxs = np.mean(patterns_c[cell_mask],axis = 0)
    cell_wise_saxs_no_0.append(saxs_no_0)
    cell_wise_saxs.append(saxs)
cws = np.array(cell_wise_saxs)
cws_no0 = np.array(cell_wise_saxs_no_0)
cww =np.array(cell_wise_mean_weight_components)
cww = cww.T

saxs = np.mean(cws,axis = 0)
    
log.info('n_patterns = {}'.format(patterns_c.shape))
chunksize = 4

#new_dataset = np.zeros((patterns_c.shape[0],)+(patterns_c.shape[1]//chunksize,))
#for i in range(0,patterns_c.shape[-1],chunksize):
#    new_dataset[:,i//4] = np.mean(patterns_c[:,i:i+chunksize],axis = -1)
#new_qs = np.zeros_like(new_dataset)
#new_qs[:]= qs[2::chunksize][None,:]

means = np.mean(patterns_c[:,q_mask2],axis=1)
filter_mask1 =  np.abs(means)<1e-1
filter_mask2 =  np.abs(means)>1e-6
filter_mask3 =  np.all(patterns_c[:,q_mask3] <0,axis=1)
filter_mask = filter_mask1 & filter_mask2 #& filter_mask3
print(patterns_c.shape)
patterns_c=patterns_c[filter_mask]
print(patterns_c.shape)

stacked_qs = np.zeros_like(patterns_c)
stacked_qs[:] = qs[None,:]


hist = np.stack((stacked_qs[:,q_mask1],patterns_c[:,q_mask1]),axis =-1).reshape(-1,2)

bins = np.sum(q_mask1)
#fig = hist2D.get_fig(hist,bins=bins,range=[[hist[...,0].min(),hist[...,0].max()],[-5e-1,5e-1]],norm='log',layout={'title':'Difference SAXS/WAXS histogram Run 140 \n normalization region [0.1,0.3] $A^{-1}$ ,randomized pump','x_label': "q [$A^-1$]",'y_label':'I [arb.]'})
#
#fig2 = plot1D.get_fig(np.mean(patterns_c,axis = 0)[qs<3],grid = qs[qs<3],ylim=[-0.005,0.005],layout={'title':'Difference SAXS/WAXS Run 140 \n normalization region [0.1,0.3] $A^-1$ randomized pump','x_label': "q [$A^{-1}$]",'y_label':'I [arb.]'})
#h = np.histogram2d(hist[...,0],hist[...,1],range=[[hist[...,0].min(),hist[...,0].max()],[-0.0005,0.0005]])[0]#,bins=np.sum(q_mask),)[0]

#


#cell_wise_svd = []
#cell_wise_no_0_component=[]
#cell_wise_no_0_component_saxs=[]
#cell_wise_single_weight = []
#for key,value in patterns.items():
#    u,s,vh = np.linalg.svd(value['saxs'][:,q_mask4],full_matrices=False,compute_uv = True)
#    cell_wise_svd.append([u,s,vh])
#    no_zero = u[:,1:]*s[1:]@vh[1:]
#    cell_wise_no_0_component.append(no_zero)
#    cell_wise_no_0_component_saxs.append(np.mean(no_zero,axis=0))
#    #ssvd_component = np.zeros((100,vh.shape[-1]))
#    #for i in range(100):
#    #    ssvd_component[i] = np.mean(u[:,i,None]*s[i]*vh[None,i],axis = 0)
#    weight_mean = np.mean(u,axis = 0)[:100]
#    cell_wise_single_weight.append(weight_mean)
        
    #fig = plot1D.get_fig(value['saxs'],grid = qs)
    #db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/c_{}.matplotlib'.format(key),fig)

#s =db.load('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/svd_s.npy')
#u =db.load('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/svd_u.npy')
#vh =db.load('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/svd_vh.npy')
#
m = np.diag(s)@vh

fig2 = scatter1D.get_fig(s[:500],s=3,layout={'title':'svd of pulse/Difference SAXS/WAXS Run 140 \n normalization region [0.1,0.3] $A^-1$ randomized pump','x_label': "Singular values ",'y_label':'magnitude'},y_scale='log')

fig2 = plot1D.get_fig((m[:10]+np.arange(1,11,dtype = float)[:,None]*50)[::-1],labels=list(np.arange(10).astype(str))[::-1],grid = qs[q_mask4],layout={'title':'s@vh svd of pulse/Difference SAXS/WAXS Run 140 \n normalization region [0.1,0.3] $A^-1$ randomized pump','x_label': "q [$A^{-1}$]",'y_label':'I [arb.]'},y_scale='log')

#fig2 = scatter1D.get_fig(s[:500],s=0.5,layout={'title':'svd of pulse/Difference SAXS/WAXS Run 140 \n normalization region [0.1,0.3] $A^-1$ randomized pump','x_label': "Singular values ",'y_label':'magnitude'},y_scale='log')e('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/svd.matplotlib'.format(key),fig2)

#fig2 = plot1D.get_fig(a,layout={'title':'Singular values of pulse/Difference SAXS/WAXS Run 140 \n normalization region [0.1,0.3] $A^-1$ randomized pump','x_label': "q [$A^{-1}$]",'y_label':'I [arb.]'})
#fig3 = heat2D.get_fig(h,scale='log')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/r140_hist_wide_random.matplotlib'.format(key),fig)

#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/r140_hist_wide_all_4.matplotlib'.format(key),fig)
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/pump_test/mean_mp_test_all_4.matplotlib'.format(key),fig2)


#data = d['diod_diff']['mean_on_polar_grid']
#mask = d['diod_diff']['mask_on_polar_grid'].astype(float)
#grid = GridFactory.construct_grid('uniform',[np.arange(data.shape[0],dtype=float),np.arange(data.shape[1],dtype=float)*2*np.pi/data.shape[1]])[:]
#db.save(base_path +time+'/mean.vts',[data],grid = grid,dset_names=['s','t'],grid_type=['polar'],)

#saxs,counts = masked_mean(data,mask,axis=1)
