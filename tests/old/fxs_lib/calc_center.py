import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import numpy as np
import scipy
import logging
import importlib
from scipy.stats import unitary_group
from scipy.stats import special_ortho_group
from xframe import log
#from testing.testLibrary import defaultTest_array
from xframe.library import pythonLibrary as pyLib
from xframe.library import mathLibrary as mLib
from xframe.library import physicsLibrary as pLib
from xframe.library.gridLibrary import uniformGrid_func
from xframe.library.gridLibrary import GridFactory
from xframe.library.gridLibrary import ReGrider
from xframe.library.gridLibrary import SampledFunction
from xframe.library.gridLibrary import NestedArray
from xframe.database.general import load_settings
from xframe.externalLibraries.shtns_plugin import sh as shtns
from xframe.plugins.MTIP.analysisLibrary.harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid_new as get_grid
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_zernike_spherical_ft 
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_spherical_zernike_ht 
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import ReciprocalProjection
from xframe.control.communicators import mock_comm_module_single_thread
from xframe.plugins.MTIP.analysisLibrary.misk import generate_calc_center
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import generate_shift_by_operator

from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import calc_spherical_zernike_weights
from xframe.plugins.MTIP.database import Analysis_DB
import vtk
from vtk.util import numpy_support as vtn
from xframe.control.communicators import mock_comm_module_single_thread

log=log.setup_custom_logger('root','INFO')
comm_module = mock_comm_module_single_thread()


def initialize(l_max=99, n_points=200,ft_mode='Zernike_3D',pi_in_q = False):
    opt,_ = load_settings('plugins/MTIP','3d_model_0')    
    db = Analysis_DB(**opt['IO'])
    mLib.shtns = shtns

    harm_opt={'dimensions':3,
              'indices':'lm',
              'max_order':l_max}#19}

    ht = HarmonicTransform('complex',harm_opt)

    if ft_mode == 'Zernike_3D':
        grid_type='Zernike'
        Q=0.05141790928385248
        if pi_in_q:            
            Q*=2*np.pi
            
    elif ft_mode == 'PNAS_3D':
        grid_type = 'PNAS'
        Q=n_points/2
        
    opt = {
        **{
            'dimensions':3,
            'type':grid_type,
            'reciprocal_cut_off':Q,
            'n_radial_points':n_points,
            'pi_in_q':pi_in_q
        },**ht.grid_param
    }

    opt2 = {
        'dimensions':2,
        'type':'Zernike',
        'reciprocal_cut_off':0.05141790928385248,
        'n_radial_points':200,
        'n_angular_points':1600,
        'pi_in_q':pi_in_q
    }
    
    grid_pair =  get_grid(opt)
    grid = grid_pair.realGrid
    grid2 = get_grid(opt2).realGrid
    r_grid = grid_pair.reciprocalGrid
    max_r = grid[:,0,0,0].max()
    n_radial_points = grid.shape[0]
    max_order = harm_opt['max_order']
    n_orders = max_order+1
    name_postfix='N'+str(n_radial_points)+'mO'+str(max_order)+'nO'+str(n_orders)
    #name_postfix='N'+str(n_radial_points)+'mO'+str(29)+'nO'+str(30)

    if ft_mode == 'Zernike_3D':
        pos_orders=np.arange(n_orders)
        expansion_limit = 1000
        opt={'expansion_limit':expansion_limit,'max_radius':max_r,'n_radial_points':n_radial_points,'pi_in_q':pi_in_q}
        weights=calc_spherical_zernike_weights(pos_orders,opt)
        weights_dict={'weights':weights,'maxR':max_r,'posHarmOrders':pos_orders}        
        ft,ift = generate_zernike_spherical_ft(max_r,weights_dict,ht,'complex',pi_in_q)
        zht,izht = generate_spherical_zernike_ht(weights_dict['weights'],weights_dict['posHarmOrders'],max_r,pi_in_q=pi_in_q)
    elif ft_mode == 'PNAS_3D':
        log.info('henerate PNAS versions of transforms')
        weights_dict=db.load('ft_weights',path_modifiers={'postfix':name_postfix,'type':'PNAS_3D'})
        ft,ift = generate_Donatelli_spherical_ft(weights_dict,ht,'complex') 
        zht,izht = generate_ht_spherical_SinCos(weights_dict)
        
    cube_func = mLib.SampleShapeFunctions.get_disk_function(500,norm='inf',random_orientation = True)
        
    return db,ht,ft,ift,grid_pair,grid2,zht,izht

os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe/xframe/')
db,ht,ft,ift,grid_pair,grid2,zht,izht = initialize(n_points = 128,pi_in_q = True)

f_ball = mLib.SampleShapeFunctions.get_disk_function(10,center=(40,np.pi/2,np.pi/2))
f_ball2 = mLib.SampleShapeFunctions.get_disk_function(10,center=(40,np.pi/2,3/2*np.pi))
ball = f_ball(grid_pair.realGrid)
balls = ball + 2*f_ball2(grid_pair.realGrid)
calc_center = generate_calc_center(grid_pair.realGrid)
shift = generate_shift_by_operator(grid_pair.reciprocalGrid,opposite_direction = True)

center = calc_center(balls) 
a_balls = ft(balls)
a_balls = shift(a_balls,center)
balls2 = ift(a_balls)
center2 = calc_center(balls2) 
