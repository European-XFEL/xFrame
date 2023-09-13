if os.getcwd()[-6:]!="source":
    #print(os.getcwd())
    os.chdir('../')

import numpy as np
import traceback

from analysisLibrary.filters import Q_Mean_Sigma_Cart
from control.communicators import mock_comm_module_single_thread


from database.general import load_settings
from analysisLibrary.regrid import AgipdRegridder

from library.mathLibrary import cartesian_to_spherical
from library.mathLibrary import spherical_to_cartesian

from detectors.agipd import AGIPD
from library import physicsLibrary as pLib
from library import units
from database.euxfel import fxs2510_DB
from database.euxfel import fxs2510_EXP_DB

from library.gridLibrary import GridFactory
from library.gridLibrary import get_linspace_log2
from library.physicsLibrary import pixel_grid_to_scattering_grid
from presenters.openGLPresenter import FramePresenter

import log
log=log.setup_custom_logger('root','INFO')

settings_ana = load_settings('fxs2510','latest',target='analysis')
settings_exp = load_settings('fxs2510','proc_latest',target='experiment')
db_exp = fxs2510_EXP_DB(**settings_exp['IO'])
db_ana = fxs2510_DB(**settings_ana['IO'])

agipd = AGIPD(db_exp,load_geometry_file=True)
agipd.origin=np.array([0,0,1500])
d_shape = (16,512,128)
old_data = np.arange(np.prod(d_shape)).reshape(d_shape)
energy = 6e3
wavelength = pLib.energy_to_wavelength(energy)

p_grid = pixel_grid_to_scattering_grid(agipd.pixel_grid,wavelength,out_coord_sys = 'cartesian')
p_grid *= units.standardLength
comm = mock_comm_module_single_thread()

data_grid = p_grid[:,:-1,:-1][agipd.sensitive_pixel_mask].reshape(agipd.data_shape+(3,))

opt = {'pixel_grid' : data_grid}

bragg_filter = Q_Mean_Sigma_Cart(opt,comm)


n = 20
datasets = [np.zeros(agipd.data_shape) for i in range(n)]
bragg_centers = [(np.random.rand(3)*np.array([16,512,128])).astype(int) for i in range(int(n/2))]

for i,center in enumerate(bragg_centers):
    data = datasets[i]
    data[center[0],center[1]-1:center[1]+2,center[2]-1:center[2]+2] = 1.2

datasets = np.array(datasets)
#pixel_mask = np.zeros(agipd.pixel_grid.shape[:-1],dtype = bool)
#pixel_mask[:,:-1,:-1] = agipd.sensitive_pixel_mask
#p = FramePresenter(pixel_vertices = p_grid.reshape(-1,3),pixel_indices = agipd.pixel_corner_index.flatten(), sensitive_pixel_mask = pixel_mask)

#datasets = [mask.astype(np.float32) for mask in bragg_filter.radial_masks]

#p.show(datasets)
result = bragg_filter.calc_masks(np.array(datasets),bragg_filter.radial_masks,3)

if (result == np.array([9]*int(n/2)+[0]*(n-int(n/2)))).all():
    print('test_passed')
else:
    print('Failed')

result2 = bragg_filter._apply({'data':datasets},{})

if (result2[1] == np.array([True]*int(n/2)+[False]*(n-int(n/2)))).all():
    print('test_passed')
else:
    print('Failed')

result3 = bragg_filter.apply({'data':datasets},{})

if result3[0]['data'].shape[0] == n-int(n/2):
    print('test_passed')
else:
    print('Failed')

result4 = bragg_filter.apply(*result3)
