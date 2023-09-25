import numpy as np
import os
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_ft
from xframe.plugins.MTIP.analysisLibrary.harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict,calc_spherical_trapz_weights,generate_weightDict_trapz,generate_ht,assemble_weights_trapz
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import density_to_deg2_invariants
#from xframe.library.mathLibrary import SampleShapeFunctions
#from xframe.startup_routines import load_recipes
#from xframe.control.Control import Controller
#analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_1p_00')[0]
#controller = Controller(analysis_sketch)
#from xframe.settings import analysis as opt
#from xframe.database import analysis as db

Q=0.41
n_r=128
max_order=64

ht2d = HarmonicTransform('complex',{'dimensions':2,'max_order':max_order})
grid_param = ht2d.grid_param
g = get_grid({'dimensions':2,'type':'trapz','pi_in_q':True,'max_q':Q,'n_radial_points':n_r,'phis':grid_param['phis']})
R = g.realGrid[:,0,0].max()

w = generate_weightDict(max_order,n_r,dimensions=2,mode='trapz',n_cpus=False,pi_in_q=True)
aw = assemble_weights_trapz(w['weights'],np.arange(max_order+1),R,True)
hk,ihk = generate_ht(w['weights'],np.arange(max_order+1),R,dimensions=2,pi_in_q=True)
ft,ift = generate_ft(R,w,ht2d,2,pi_in_q=True)
d = np.ones_like(g.realGrid.array[...,0])
hd = ht2d.forward(d)
d2=ift(ft(d))

print(np.sum(aw['inverse']*np.sum(aw['forward'],axis=0)[1:,None,:],axis=0)[:,0])



