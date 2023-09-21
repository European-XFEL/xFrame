import time
import numpy as np
from scipy.interpolate import griddata
import logging
import traceback
import warnings

log=logging.getLogger('root')

from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe.library.gridLibrary import GridFactory
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library import units


#very bad saxs is average not integral...
def calc_saxs_bad(mean,phis,thetas):
    log.info('nonzero entries in mean for saxs = {}'.format(np.sum(mean!=0)))
    saxs= np.trapz(mean*np.sin(thetas)[:,None],x = phis, axis = 1)
    #saxs= np.trapz(mean,x = phis, axis = 1)
    return saxs

def calc_saxs(mean,mask):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')        
        saxs= np.mean(mean, axis = 1, where = mask)
    nan_mask = np.isnan(saxs)
    if nan_mask.any():
        saxs[nan_mask] = 0
    #saxs= np.trapz(mean,x = phis, axis = 1)
    return saxs

def calc_saxs_slow(mean,mask):
    saxs = np.zeros(len(mean))
    for q_id in range(len(mean)):
        q_mask = mask[q_id]
        _sum = np.sum(mean[q_id],where = q_mask)
        n_points = np.sum(q_mask)
        if n_points != 0:
            saxs[q_id] = _sum/n_points
        else:
            saxs[q_id] = 0

    #saxs= np.trapz(mean,x = phis, axis = 1)
    return saxs

