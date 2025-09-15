import numpy as np
import numpy.ma as mp
import scipy.integrate as spIntegrate
import logging
from itertools import repeat
import sys

log=logging.getLogger('root')

from .hankel_transforms import generate_ht
from .misk import _get_reciprocity_coefficient
from .hankel_transforms import generate_weightDict
from xframe.library.mathLibrary import SphericalIntegrator,PolarIntegrator
from xframe.library.math_transforms import HankelTransformWeights,HankelWeightStruct,SphericalFourierTransformStruct
from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct

from xframe import database
from xframe import settings 

def load_fourier_transform_weights(struct:HankelWeightStruct=HankelWeightStruct(),allow_weight_saving = False):
    db = database.project
    log.info(f'ft name postfix = {struct.weight_name}')
    try:
        weights_dict = db.load('ft_weights',path_modifiers={'name':struct.weight_name})
    except FileNotFoundError as e:
        weights_dict = HankelTransformWeights.get_weights_dict(struct)
        if allow_weight_saving:
            db.save('ft_weights',weights_dict,path_modifiers={'name':struct.weight_name})                    
    return weights_dict

def fourier_transform_from_settings(allow_weight_saving = False):
    opt = settings.project
    struct = SphericalFourierTransformStruct(
        dimension = opt.dimensions,
        n_radial_points = opt.grid.n_radial_points,
        angular_bandwidth = opt.grid.max_order+1,
        hankel_type = opt.fourier_transform.type,
        n_processes_for_weight_generation = opt.multi_process.n_weight_generating_processes,
        max_q = opt.grid.max_q,
        max_nonzero_r = opt.grid.max_nonzero_r*opt.particle_radius,
        use_gpu = opt.GPU.use,
        n_polar_angles = opt.grid.get('n_phi',0),
        n_azimutal_angles = opt.grid.get('n_theta',0)
    )

    weights = load_fourier_transform_weights(struct,allow_weight_saving = allow_weight_saving)
    sft = SphericalFourierTransform(struct,weights = weights['weights'])
    return sft
############################################
###  generate fourier transforms Legacy  ###
def select_harmonic_transforms(harm_trf,dimensions,use_gpu):
    if (dimensions == 3) and use_gpu:
        trfs = harm_trf.transforms_by_indices['direct']
    elif (dimensions == 3) and (not use_gpu):
        trfs = harm_trf.transforms_by_indices['ml']
    elif  (dimensions == 2):
        trfs = harm_trf.transforms_by_indices['m']
    ht,iht = trfs['forward'],trfs['inverse']
    return ht,iht

def generate_ft(r_max,weights,harm_trf,dimensions,pos_orders=False,use_gpu=False,reciprocity_coefficient = np.pi,mode = 'trapz'):
    if not isinstance(pos_orders,bool):
        orders = pos_orders
    else:        
        orders=weights['posHarmOrders']
    weights_matrix = weights['weights']
    hankel,ihankel = generate_ht(weights_matrix,orders,r_max,reciprocity_coefficient=reciprocity_coefficient,dimensions=dimensions,use_gpu=use_gpu,mode=mode)
    ht,iht = select_harmonic_transforms(harm_trf,dimensions,use_gpu)
    def ft(data):
        #log.info('ft input shape = {}'.format(data.shape))
        harmonic_coefficients=ht(data)
        #log.info(f'coeff shape = {harmonic_coefficients.shape}')
        reciprocal_harmonic_coefficients=hankel(harmonic_coefficients)        
        reciprocal_function=iht(reciprocal_harmonic_coefficients)
        #reciprocal_function[0] = reciprocal_function[0].mean()
        #reciprocal_function[0] = 0
        #reciprocal_function[1] = reciprocal_function[1].mean()
        #reciprocal_function[2] = reciprocal_function[2].mean()
        #reciprocal_function[0]*= np.sqrt(2*np.pi)
        #reciprocal_function[0] = real_integrator(data)
        #reciprocal_function[0] = 0
        return reciprocal_function
    
    def ift(data):
        #log.info(f'data shape = {data.shape}')
        reciprocal_harmonic_coefficients=ht(data)
        #log.info(f'coeff shape = {len(reciprocal_harmonic_coefficients)}')
        harmonic_coefficients=ihankel(reciprocal_harmonic_coefficients)
        real_function=iht(harmonic_coefficients)
        #real_function[0] = real_function[0].mean()
        #real_function[0]*= np.sqrt(2*np.pi)
        #real_function[1] = real_function[1].mean()
        #real_function[2] = real_function[2].mean()
        #real_function[0] = reciprocal_integrator(data)
        #real_function[0] = 0
        #log.info('sum ift output = {}\n'.format(np.sum(real_function)))
        return real_function
    return ft,ift


