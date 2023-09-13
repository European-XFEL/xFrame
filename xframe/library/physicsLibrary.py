import numpy as np
import logging

from xframe.library import units 
from xframe.library import mathLibrary as mLib
log=logging.getLogger('root')

#assumes scattering object is at the coordinate origin (0,0,0).
# beam is along z axis
def pixel_grid_to_scattering_grid(pixel_grid,wavelength,approximation='None',out_coord_sys='spherical'):
    if approximation == 'None':
        scattering_grid = get_spherical_scattering_grid(pixel_grid,wavelength)
    elif approximation == 'small_angle':
        scattering_grid = get_spherical_scattering_grid_small_angle(pixel_grid,wavelength)

    scattering_grid[...,0]*=units.standardLength
    if out_coord_sys == 'cartesian':
        scattering_grid = mLib.spherical_to_cartesian(scattering_grid)
    return scattering_grid
def get_pixel_areas_reziprocal(scattering_grid_cart,wavelength):
    g = scattering_grid_cart
    g[...,2] -= 2*np.pi/wavelength # shift ewald sphere center to origin.

    r = 2*np.pi/wavelength
    arc_angle_x = np.arccos(np.sum(g[:,:-1,:-1]*g[:,1:,:-1],axis = -1)/r**2)
    arc_angle_y = np.arccos(np.sum(g[:,:-1,:-1]*g[:,:-1,1:],axis = -1)/r**2)
    area = r**2*arc_angle_x*arc_angle_y
    return area
    
def get_spherical_scattering_grid_wrong(pixel_grid,incidentWavelength):
    detectorPixelGrid=pixel_grid

    radii=np.linalg.norm(detectorPixelGrid,axis=-1)
    zPositions=detectorPixelGrid[...,2]

    scatteringAngles = np.arctan2(zPositions,radii)

    reciprocalRadii = 2*np.sin(scatteringAngles/2)*1/incidentWavelength
    reciprocalTheta = (np.pi - scatteringAngles)/2
    reciprocalPhi = mLib.phiFrom_R_Theta_X_Y(radii,reciprocalTheta,detectorPixelGrid[...,0],detectorPixelGrid[...,1])                
    
    scattering_grid = np.stack((reciprocalRadii,reciprocalTheta,reciprocalPhi),axis=-1)
    return scattering_grid


def get_spherical_scattering_grid(pixel_grid,incidentWavelength):
    #pixel_grid[...,2]=100
    r = np.linalg.norm(pixel_grid,axis = -1)
    z = pixel_grid[...,2]
    scatteringAngles = np.zeros(z.shape)
    neg_z = z<0
    zr = z/r
    scatteringAngles[~neg_z] = np.arccos(zr[~neg_z])
    scatteringAngles[neg_z] = np.pi - np.arccos(-zr[neg_z])
    #print(scatteringAngles)
    reciprocalRadii = 4*np.pi*np.sin(scatteringAngles/2)/incidentWavelength
    #print(reciprocalRadii)
    reciprocalTheta = (np.pi-scatteringAngles)/2
    #log.info(reciprocalTheta)
    reciprocalPhi = np.arctan2(pixel_grid[...,1],pixel_grid[...,0])
    
    scattering_grid = np.stack((reciprocalRadii,reciprocalTheta,reciprocalPhi),axis=-1)
    return scattering_grid

def scattering_angle_to_reciprocal_radii(scattering_angle,xray_wavelength):
    ''' scattering_angle = 2*theta '''
    reciprocalRadii = 4*np.pi*np.sin(scattering_angle/2)/xray_wavelength
    return reciprocalRadii
def get_spherical_scattering_grid_small_angle(pixel_grid,incidentWavelength):
    r = np.linalg.norm(pixel_grid,axis = -1)
    r_azim = np.linalg.norm(pixel_grid[...,:2],axis = -1)
    z = pixel_grid[...,2]
    neg_z = z<0
    rr = r_azim/r
    scatteringAngles = np.zeros(z.shape)
    scatteringAngles[~neg_z] = rr[~neg_z]
    scatteringAngles[neg_z] = (np.pi-rr)[neg_z]
        
    reciprocalRadii=scatteringAngles*2*np.pi/incidentWavelength
    reciprocalTheta=(np.pi-scatteringAngles)/2
    reciprocalPhi = np.arctan2(pixel_grid[...,1],pixel_grid[...,0])
        
    scattering_grid=np.stack((reciprocalRadii,reciprocalTheta,reciprocalPhi),axis=-1)
    return scattering_grid


def ewald_sphere_theta_func(incidentWavelength):
    def theta(q):
        value=np.arccos(q*incidentWavelength/2)
        return value
    return theta


def ewald_sphere_theta_pi(incidentWavelength,qs):
    return np.arccos(qs*incidentWavelength/(4*np.pi)) 

def ewald_sphere_theta(incidentWavelength,qs):
    return np.arccos(qs*incidentWavelength/2)

def ewald_sphere_q_pi(wavelength,theta):
    return 4*np.pi*np.cos(theta)/wavelength
def ewald_sphere_wavelength_pi(max_q,theta):
    return 4*np.pi*np.cos(theta)/max_q

def energy_to_wavelength(energy):
    wavelength = (units.c*units.h)/energy
    return wavelength



def spherical_formfactor(q, radius = 1000):
    R = radius
    f = np.zeros_like(q)
    zero_mask = (q == 0.)
    V = (4/3)*np.pi*(R**3)
    f[zero_mask]= (4*np.pi*V**2)
    qq=q[~zero_mask]
    f[~zero_mask]= 36*np.pi*V**2*((np.sin(qq*R)-qq*R*np.cos(qq*R))/(qq*R)**3)**2    
    return f

