import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.startup_routines import dependency_injection
dependency_injection()
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward,cartesian_to_spherical,spherical_to_cartesian
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward

from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import NearestNeighbourAgipdRegridder,create_lookup_array_2D,NearestNeighbourRegridder

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter

db = default_DB()

from xframe.startup_routines import load_recipes
from xframe.control.Control import Controller
analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='hist_mean_162',exp_name='fxs3046_online',exp_settings = 'proc_-20_20_03_4')
controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
controller.chooseJobSingleProcess()
from xframe import database
from xframe import Multiprocessing

comm = Multiprocessing.comm_module
g = comm.get_geometry()

base_path = '/gpfs/exfel/theory_group/user/berberic/fxs3046/test/regrid2/'

pixel_centers = g['framed_pixel_centers']
data = np.zeros((16,512,128))
for module in data:
    for n,i in enumerate(module):
        i[n%2::2]=1

grid_cart = spherical_to_cartesian(g['pixel_grid'])
mask = g['mask']
fmask = g['framed_mask']
#fig=agipd_heat.get_fig(data,grid_cart,mask)
#db.save(base_path + 'data.matplotlib',fig)

#xs = np.arange(-5.5,6,1)
#pixel_centers = GridFactory.construct_grid('uniform',[xs,xs])[:]
#pixel_centers_polar = cartesian_to_spherical(pixel_centers)
#data = np.zeros(pixel_centers.shape[:-1])
#for n,i in enumerate(data[1:-1]):
#    i[1+n%2:-1:2]=1


#fig = heat2D.get_fig(data,grid = pixel_centers)

bound = np.min(np.max(spherical_to_cartesian(pixel_centers),axis = (0,1,2))[:2])
rs = np.arange(1024)*bound/1024
phis = np.arange(0,3200)*2*np.pi/3200
polar_grid = GridFactory.construct_grid('uniform',[rs,phis])[:]
cart_grid = spherical_to_cartesian(polar_grid)
#pixel_size = 2
#
rg = NearestNeighbourRegridder(pixel_centers,polar_grid)
#
framed_data = np.zeros(fmask.shape,dtype = float)
framed_data[fmask]=data.flatten()
polar_data = rg.apply(framed_data)

db.save(base_path + 'polar_data.vts',[polar_data],grid = polar_grid,grid_type='polar')


#fig_polar = heatPolar2D.get_fig(polar_data,grid = np.swapaxes(polar_grid,0,1))
#db.save(base_path + 'polar_data.matplotlib',fig_polar,dpi = 1600)
#fig_polar2 = heatPolar2D.get_fig(polar_data,grid = np.swapaxes(polar_grid,0,1))
#db.save(base_path + 'polar_data2.matplotlib',fig_polar2)

#ixs = np.arange(-5.5,6,0.5)
#intermediate_grid = GridFactory.construct_grid('uniform',[ixs,ixs])[:]

#tree = KDTree(pixel_centers.reshape(-1,2))
#tree2 = KDTree(pixel_centers.reshape(-1,2))
#q = tree.query(cart_grid.reshape(-1,2))
#q2 = tree.query(cart_grid.reshape(-1,2))
#ids = np.unravel_index(q[1],pixel_centers[:].shape[:-1])
#id2 = np.unravel_index(q[1],pixel_centers[:].shape[:-1])
#





#cart_pix_centers = spherical_to_cartesian(pixel_centers)[...,:2]
## create lookup for intermediate uniform grid    possible_point_ids[:,0]
#grid_intermediate_ids = (cart_grid+pixel_size/2)//pixel_size
#pix_intermediate_ids = (cart_pix_centers+pixel_size/2)//pixel_size
#
#intermediate_lookup_grid = generate_lookup_dict(grid_intermediate_ids)
#intermediate_lookup_pix = generate_lookup_dict(pix_intermediate_ids)
#
#
#
#lookup_array = create_lookup_array(pixel_centers,polar_grid,pixel_size)
#
#polar_data = data[lookup_array[...,0],lookup_array[...,1]]
#


