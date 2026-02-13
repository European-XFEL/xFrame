import scipy.constants as constants
import numpy as np

hJ = constants.physical_constants['Planck constant'][0]
h=constants.physical_constants['Planck constant in eV s'][0]
hbar=constants.physical_constants['Planck constant over 2 pi in eV s'][0]
c=constants.c
standardLength=1e-10

def rad_to_degree(radians):
    return radians*180/np.pi
def degree_to_rad(degrees):
    return degrees*np.pi/180

def wave_vector_unit_to_inverse_length_unit(data):
    return data/(2*np.pi)
def inverse_length_unit_to_wave_vector_unit(data):
    return 2*np.pi*data

def energy_wavelength_conversion(e_or_w,energy_unit='eV'):
    if energy_unit == 'eV':
        w_or_e = (c*h)/e_or_w
    elif energy_unit == 'J':
        w_or_e = (c*hJ)/e_or_w
    else:
        raise Exception(f'Only eV and J are allowed energy units, but {energy_unit} was given.')
    return w_or_e
