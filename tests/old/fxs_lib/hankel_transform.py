import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
import numpy as np
import scipy
import logging
from xframe import log
from xframe.startup_routines import load_recipes
from xframe.library import mathLibrary
from xframe.externalLibraries.flt_plugin import LegendreTransform
mathLibrary.leg_trf = LegendreTransform
from xframe.control.Control import Controller
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict_zernike
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict_trapz
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_ht
from xframe import settings
from xframe import Multiprocessing
import xframe
xframe.settings.general.n_control_workers = 1
xframe.controller.control_worker.restart_working()
#log=log.setup_custom_logger('root','INFO')

analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_model_0')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
n_radial_points = 256
max_order = 99
orders = np.arange(max_order+1)
r_max = 1000


#wd3 = generate_weightDict_zernike(max_order,n_radial_points,n_cpus = 30,dimensions=3)
#wd33 = generate_weightDict_trapz(max_order,n_radial_points,n_cpus = 30,dimensions=3)
#w3 = wd3['weights']
#w33 = wd33['weights']
#
#wd2 = generate_weightDict_zernike(max_order,n_radial_points,n_cpus = 30,dimensions=2)
#wd22 = generate_weightDict_trapz(max_order,n_radial_points,n_cpus = 30,dimensions=2)
#w2 = wd2['weights']
#w22 = wd22['weights']

wd2 = generate_weightDict_trapz(max_order,n_radial_points,n_cpus = 30,dimensions = 2)
ht2,iht2 = generate_ht(wd2['weights'],orders,r_max,pi_in_q=True,dimensions = 2)

wd3 = generate_weightDict_trapz(max_order,n_radial_points,n_cpus = 30,dimensions = 3)
ht3,iht3 = generate_ht(wd3['weights'],orders,r_max,pi_in_q=True,dimensions = 3)
ht3g,iht3g = generate_ht(wd3['weights'],orders,r_max,pi_in_q=True,dimensions = 3,use_gpu=True)
#
d2 = np.ones((n_radial_points,2*len(orders)-1))
i2 = iht2(ht2(d2))
diff2 = i2-d2
rd2 = np.mean(np.sqrt((diff2*diff2.conj()).real)/d2)
#
m_orders=np.concatenate((np.arange(max_order+1,dtype = int),-np.arange(max_order,0,-1,dtype = int)))
d3 = [ np.ones((n_radial_points,max_order+1)) for m in m_orders]
nlm = max_order*(max_order+2)+1
d3g = np.ones((n_radial_points,nlm),dtype = complex)
i3 = iht3(ht3(d3))
i3g = iht3g(ht3g(d3g))


diff3g= d3g-i3g
rd3g = np.mean(np.sqrt((diff3g*diff3g.conj()).real)/d3g)


#wd3 = generate_weightDict_zernike(max_order,n_radial_points,n_cpus = 30)
#w3 = assemble_weights_zernike(wd3['weights'],np.arange(100),1000,True)
#
#w3f = w3['forward']
#w3i = w3['inverse']
#
#d3f = np.sum(w3f*10,axis = 0)
#d3i = np.sum(d3f[1:,None,:]*w3i,axis=0)
#
#weight_dict2 = generate_weightDict_zernike(max_order,n_radial_points,n_cpus = 30,dimension=2)
#w2 = assemble_weights_zernike(weight_dict2['weights'],np.arange(100),1000,True,dimension=2)
##
#w2f = w2['forward']
#w2i = w2['inverse']
##
#d2f = np.sum(w2f*10,axis = 0)
#d2i = np.sum(d2f[1:,None,:]*w2i,axis=0)
#
#ww_pi = calc_zernike_weights_pi_2(np.arange(100),500,128)
#ww = calc_zernike_weights_2(np.arange(100),500,128)
#zht,izht = generate_zernike_ht_old(ww,np.arange(100),1000)

#wd2 = np.moveaxis(calc_zernike_weights_pi_2(np.arange(100),500,128),0,2)
#d2f2 = np.sum(wd2,axis = 0)
#d2i2 = np.sum(d2f2[:,None,:]*wd2,axis = 0)
