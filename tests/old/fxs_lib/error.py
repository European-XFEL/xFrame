import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.presenters.matplolibPresenter import plot1D

db = default_DB()

base_path = '/gpfs/exfel/theory_group/user/berberic/MTIP/reconstructions/20_8_2022/run_0/'
path = base_path + 'reconstruction_data.h5'
data = db.load(path)
r = data['reconstruction_results']
errors = np.moveaxis(np.array([d['error_dict']['deg2_invariant_l2_diff'] for d in r.values()]),2,0)

order =6
fig = plot1D.get_fig(errors[order,:,2:],y_scale = 'log')
fig_path = base_path + 'order_{}_Bl_errors.matplotlib'.format(order)
db.save(fig_path,fig)
