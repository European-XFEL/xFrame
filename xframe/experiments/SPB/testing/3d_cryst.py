import numpy as np
import time
from xframe.settings import general
general.n_control_worker=0
import xframe
from os import path as pa

db = xframe.database.default

path = '/gpfs/exfel/theory_group/user/berberic/cryst_vols/sphv_agno3-d07_h_final.npy'
_dir = pa.dirname(path)
data = np.load(path,allow_pickle = True)

n_qs = 300
qs = np.arange(n_qs)*9/n_qs
n_phi = 500
phi=np.arange(n_phi)/n_phi *2*np.pi
n_theta = 1000
phi=np.arange(n_thetas)/n_thetas *np.pi
grid = xframe.lib.grid.GridFactory.construct_grid('uniform',[qs,thetas,phis])[:]

db.save(_dir+'volume.vts',[data.reshape(len(qs,n_theta,n_theta))],grid_type='sperical')


xframe.setup_experiment('fxs3046_online','normalized_proc')





