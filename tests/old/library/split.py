import numpy as np
from xframe.settings import general
general.n_control_worker=0
import xframe
xframe.setup_experiment('fxs3046_online','default_proc')
exp = xframe.experiment_worker
exp.set_ids_for_run(169)

frames = np.arange(exp.nframes)
mask = (exp.train_ids>exp.train_ids.min() + 100) & (exp.train_ids < exp.train_ids.max() -100)
mask2 = exp.cell_ids != 0
frames = frames[mask & mask2]
a = xframe.lib.python.split_into_simple_slices(frames,mod = False,return_sliced_args = True)

#frames[-2] = 109324-202
#frames[-1] = 109324
#slices = []
#lengths = []
#jumpsl = []
#mod = 202
#eq_class_ids = frames%mod
#for c_id in range(mod):
#    eq_class = frames[eq_class_ids==c_id]
#    if len(eq_class)<=0:
#        continue
#    eq_min = eq_class[0]
#    temp = (eq_class-eq_min)//mod
#    jumps = np.nonzero(np.diff(temp)!=1)[0]+1
#    jumpsl.append(jumps)
#    connected_components = np.split(eq_class,jumps)
#    for c in connected_components:
#a
#if len(c) >0:
#            slices.append(slice(c[0],c[-1]+mod/2,mod))
#            lengths.append(len(c))
#
