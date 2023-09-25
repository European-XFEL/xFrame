import numpy as np
from xframe.settings import general
general.n_control_worker=0
import xframe

xframe.setup_experiment('fxs3046_online','normalized_proc')

opt = {
    'run':169,
    'data_mode': 'proc',
    'frame_range': slice(None),
    'selection': {
        'cells': slice(1,177),
        'pulses': slice(None),
        'trains': slice(0,-100,1)
                  },
    'modules':np.arange(16),
    'n_frames':20000,
    'good_cells':np.arange(1,202),
    'in_multiples_of':False,
}
exp =xframe.experiment_worker
gen = exp.get_data(opt)
geom = exp._get_geometry_rois()




