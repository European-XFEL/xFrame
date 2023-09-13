import numpy as np
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.startup_routines import dependency_injection
dependency_injection()
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward as ht
from xframe.library.mathLibrary import cartesian_to_spherical,spherical_to_cartesian,SampleShapeFunctions
from xframe.library.mathLibrary import circularHarmonicTransform_real_inverse as iht

from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import SimpleRegridder,create_lookup_array_2D
from xframe.plugins.fxs3046_online.analysisLibrary.cross_correlation import generate_calc_cross_correlation

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import timeit
import time


db = default_DB()



def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)
#
#from xframe.startup_routines import load_recipes
#from xframe.control.Control import Controller
#analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='hist_mean_162',exp_name='fxs3046_online',exp_settings = 'proc_-20_20_03_4')
#controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
#controller.chooseJobSingleProcess()
#from xframe import database
#from xframe import Multiprocessing
#
#comm = Multiprocessing.comm_module
#g = comm.get_geometry()

base_path = '/gpfs/exfel/theory_group/user/berberic/fxs3046/test/ccn/'

#fig=agipd_heat.get_fig(data,grid_cart,mask)
#db.save(base_path + 'data.matplotlib',fig)

#xs = np.arange(-5.5,6,1)
#pixel_centers = GridFactory.construct_grid('uniform',[xs,xs])[:]
#pixel_centers_polar = cartesian_to_spherical(pixel_centers)
#data = np.zeros(pixel_centers.shape[:-1])
#data[5:7]=1
poly_func = SampleShapeFunctions.get_polygon_function(2,6,coordSys = 'polar')
poly_func_cart = SampleShapeFunctions.get_polygon_function(2,6,coordSys = 'cartesian')


centers = spherical_to_cartesian(np.array([(0,0)] + [(1,phi*2*np.pi/5) for phi in range(5) ]))
densities = [25,50,25,50,25,50]

#for n,i in enumerate(data[1:-1]):
#    i[1+n%2:-1:2]=1


#fig = heat2D.get_fig(data,grid = pixel_centers)

#bound = np.min(np.max(spherical_to_cartesian(pixel_centers),axis = (0,1,2))[:2])
bound = 12
n_r_steps = 512
rs = np.arange(n_r_steps//2)*2*bound/n_r_steps
xs = np.arange(-n_r_steps//2,n_r_steps//2)*2*bound/n_r_steps

xs2 = np.arange(-n_r_steps,n_r_steps)*2*bound/n_r_steps

irs = np.pi/(bound)*np.arange(n_r_steps//2)
ixs = np.pi/bound*np.arange(-n_r_steps//2,n_r_steps//2)
ixs2 = np.pi/bound*np.arange(-n_r_steps,n_r_steps)/2


n_phis = 1024
phis = np.arange(n_phis)*2*np.pi/n_phis
cart_grid = GridFactory.construct_grid('uniform',[xs,xs])[:]
polar_grid = GridFactory.construct_grid('uniform',[rs,phis])[:]

icart_grid2 = GridFactory.construct_grid('uniform',[ixs2,ixs2])[:]
icart_grid = GridFactory.construct_grid('uniform',[ixs,ixs])[:]
ipolar_grid = GridFactory.construct_grid('uniform',[irs,phis])[:]

polar_data = poly_func(polar_grid)

#cart_data = poly_func_cart(cart_grid)
cart_data = SampleShapeFunctions.get_disk_function(0.5,lambda points:np.full(points.shape[:-1],densities[0]),center=centers[0],norm='standard',coordSys='cartesian')(cart_grid)
for c,d in zip(centers[1:],densities[1:]):
    cart_data+=SampleShapeFunctions.get_disk_function(0.5,lambda points:np.full(points.shape[:-1],d),center=c,norm='standard',coordSys='cartesian')(cart_grid)
    
pixel_size = ixs[1] - ixs[0]
n_pixels = 2*np.pi*irs//pixel_size
n_orders = (n_pixels//2 + 1 ).astype(int)

cc = generate_calc_cross_correlation(polar_data.shape,n_orders,mask_type='unmasked')
rg = SimpleRegridder(cartesian_to_spherical(icart_grid),ipolar_grid)
rg2 = SimpleRegridder(icart_grid,icart_grid2,coord_sys='cartesian',interpolation = 'linear')
rg3 = SimpleRegridder(icart_grid,icart_grid2,coord_sys='cartesian',interpolation = 'nearest')
rgl = SimpleRegridder(cartesian_to_spherical(icart_grid),ipolar_grid,interpolation='linear')

#polar_data = rg.apply(data)
ftd = np.fft.fft2(cart_data)
I = (ftd*ftd.conj()).real
I = np.roll(I,n_r_steps//2,axis=(0,1))


polar_I=rg.apply(I)
l_polar_I = rgl.apply(I)
#print(timeit.timeit('fu(I)',number=100,globals = globals()))

spline_polar_grid = spherical_to_cartesian(GridFactory.construct_grid('uniform',[np.arange(n_r_steps//2)*2,phis])[:]) + np.array(I.shape)-0.5
spline_polar_grid2 = spherical_to_cartesian(GridFactory.construct_grid('uniform',[np.arange(n_r_steps//2),phis])[:]) + np.array(I.shape)/2-0.5
s_polar_I2 = map_coordinates(rg2.apply(I),(spline_polar_grid[...,0],spline_polar_grid[...,1]))
s_polar_I3 = map_coordinates(rg3.apply(I),(spline_polar_grid[...,0],spline_polar_grid[...,1]))
s_polar_I = map_coordinates(I,(spline_polar_grid2[...,0],spline_polar_grid2[...,1])) 
#print(timeit.timeit('map_coordinates(I,(spline_polar_grid[...,0],spline_polar_grid[...,1]))',number=100,globals = globals()))

cc_data = cc(polar_I)
start = time.time()
Im = np.swapaxes(ht(s_polar_I),0,1)
lIm = np.swapaxes(ht(s_polar_I2),0,1)
nIm = np.swapaxes(ht(s_polar_I3),0,1)
llIm = np.swapaxes(ht(l_polar_I),0,1)

#for i in range(n_r_steps//2):
#    Im[n_orders[i]:,i]=0
#
ccm = np.abs(Im[:,:,None]*Im.conj()[:,None,:]) #np.moveaxis(np.abs(ht(cc_data)),2,0)
lccm = np.abs(lIm[:,:,None]*lIm.conj()[:,None,:]) #np.moveaxis(np.abs(ht(cc_data)),2,0)
nccm = np.abs(nIm[:,:,None]*nIm.conj()[:,None,:]) #np.moveaxis(np.abs(ht(cc_data)),2,0)
llccm = np.abs(llIm[:,:,None]*llIm.conj()[:,None,:]) #np.moveaxis(np.abs(ht(cc_data)),2,0)
print(time.time()-start)
ccm_grid = GridFactory.construct_grid('uniform',[irs,irs])[:]
#
#db.save(base_path + 'polar_data.vts',[polar_data],grid = polar_grid,grid_type='polar')
#db.save(base_path + 'cart_data.vtr',[cart_data],grid = cart_grid,grid_type='cartesian')
#db.save(base_path + 'intensity.vtr',[I],grid = cart_grid,grid_type='cartesian')
db.save(base_path + 'polar_I.vts',[polar_I,l_polar_I,s_polar_I],dset_names=['NN','linear','spline'],grid = ipolar_grid,grid_type='polar')

orders = range(0,30,2)
db.save(base_path + 'ccm.vtr',[ccm[i] for i in orders],dset_names = ['C{}'.format(o) for o in orders],grid = ccm_grid,grid_type='cartesian')
db.save(base_path + 'lccm.vtr',[lccm[i] for i in orders],dset_names = ['lC{}'.format(o) for o in orders],grid = ccm_grid,grid_type='cartesian')
db.save(base_path + 'llccm.vtr',[llccm[i] for i in orders],dset_names = ['llC{}'.format(o) for o in orders],grid = ccm_grid,grid_type='cartesian')
db.save(base_path + 'nccm.vtr',[lccm[i] for i in orders],dset_names = ['nC{}'.format(o) for o in orders],grid = ccm_grid,grid_type='cartesian')

#options = {'plot_range':[1e7,1e16]}
#def _save_first_invariants(bls,radial_points,base_path,options,name='model_0_'):
#    max_value = np.max(tuple(np.abs(bl).max() for bl in bls))
#    vmin,vmax = options.get('plot_range',[max_value*1e-12,max_value])
#    if [vmin,vmax] == [None,None]:
#        vmin,vmax = [max_value*1e-12,max_value]
#        
#    grid = GridFactory.construct_grid('uniform',[radial_points,np.arange(bls.shape[2])])
#    order_ids = [[2,4,16,24,34],[36,38,40,42,44]]
#    #order_ids = [20,40,60,80,127]
#    
#    shape = [2,len(order_ids[0])]
#    layouts = []
#    for i in range(shape[0]):
#        layout_part = []
#        orders = np.array(order_ids[i])
#        #orders = np.arange(10*i,10*i+len(order_ids)*2,2)
#        for o in orders:
#            layout = {'title':'$C_{'+'{}'.format(o)+'}$',
#                      'x_label':'$q_1$',
#                      'y_label':'$q_2$'
#                      }
#            layout_part.append(layout)
#        layouts.append(layout_part)
#                                                
#    fig_bl_masks = heat2D_multi.get_fig([[np.abs(bls[n]) for n in order_ids[0]],[np.abs(bls[n]) for n in order_ids[1]] ],scale = 'log',layout = layouts,grid =grid[:],shape = shape,size = (30,10),vmin= vmin, vmax = vmax,cmap='plasma')
#    bl_path = base_path +name+'ccn.matplotlib'
#    db.save(bl_path,fig_bl_masks,dpi = 300)
#    
##_save_first_invariants(ccm,irs,base_path,options,'pentagon_')
