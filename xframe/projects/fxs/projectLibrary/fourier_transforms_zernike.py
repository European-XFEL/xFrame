import numpy as np
import numpy.ma as mp
import scipy.integrate as spIntegrate
import logging
from itertools import repeat

log=logging.getLogger('root')

import xframe.library.pythonLibrary as pyLib
import xframe.library.mathLibrary as mLib
from .harmonic_transforms import generate_circular_harmonic_transform_pair
from .harmonic_transforms import HarmonicTransform
from .hankel_transforms_zernike import generate_ht
from xframe import database
from xframe.settings import analysis as settings

doublePrecision=pyLib.doublePrecision



#####################
###  Zernike FTs  ###
def select_harmonic_transforms(harm_trf,dimensions,use_gpu):
    if (dimensions == 3) and use_gpu:
        trfs = harm_trf.transforms_by_indices['direct']
    elif (dimensions == 3) and (not use_gpu):
        trfs = harm_trf.transforms_by_indices['ml']
    elif  (dimensions == 2):
        trfs = harm_trf.transforms_by_indices['m']
    ht,iht = trfs['forward'],trfs['inverse']
    return ht,iht

def generate_zernike_ft(r_max,weights,harm_trf,dimensions,use_gpu=False,pi_in_q = False,mode = 'trapz'):
    orders=weights['posHarmOrders']
    weights_matrix = weights['weights']
    hankel,ihankel = generate_ht(weights_matrix,orders,r_max,pi_in_q=pi_in_q,dimensions=dimensions,use_gpu=use_gpu,mode=mode)
    ht,iht = select_harmonic_transforms(harm_trf,dimensions,use_gpu)
    
    def ft(data):
        harmonic_coefficients=ht(data)    
        reciprocal_harmonic_coefficients=hankel(harmonic_coefficients)
        reciprocal_function=iht(reciprocal_harmonic_coefficients)
        reciprocal_function[0] = reciprocal_function[0].mean()
        return reciprocal_function
    
    def ift(data):
        reciprocal_harmonic_coefficients=ht(data)
        harmonic_coefficients=ihankel(reciprocal_harmonic_coefficients)
        real_function=iht(harmonic_coefficients)
        real_function[0] = real_function[0].mean()
        return real_function
    
    return ft,ift

