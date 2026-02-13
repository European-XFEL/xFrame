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
def wavelength_to_energy(wavelength):
    energy = (units.c*units.h)/wavelength
    return energy


def spherical_formfactor(q, radius = 1000):
    R = radius
    f = np.zeros_like(q)
    zero_mask = (q == 0.)
    V = (4/3)*np.pi*(R**3)
    f[zero_mask]= (4*np.pi*V**2)
    qq=q[~zero_mask]
    f[~zero_mask]= 36*np.pi*V**2*((np.sin(qq*R)-qq*R*np.cos(qq*R))/(qq*R)**3)**2    
    return f


def determine_solid_angle_correction_from_qtheta(thetas):
    """
     determine solid_angle_correction factor 
     using the momentum_transfer theta (spherical coordinate) of each detector pixel
     the experimental intensity should be multiplied by solid_angle_correction
    """
    scattering_angle = np.pi-thetas*2
    solid_angle_correction = np.abs(1.0/(np.cos(scattering_angle)**3))
    return solid_angle_correction

def determine_solid_angle_correction_from_corners(pixel_corners:np.ndarray):
    """
     determine solid_angle_correction factor
     using the pixel corner positions in laboratory space, where the z coordinate 
     is the sample-detector distance.
     the experimental intensity should be multiplied by solid_angle_correction


     To compute the solid angle each pixel spans we will do the following.
     1.compute the normals to the planes  connecting the pixel sides with the coodinate origin
                  
     x10--------x11
     |           |
     |           |
     |   Pixel   |
     |           |
     x00--------x01

     2. compute the internal angles (p1,p2,p3,p4) of at each pixel corner projected to the unit sphere,
        by computing the angle between the pairs of normals whose planes intersect in the 
        respective pixel corner. (using <n1,n2> = cos(angle) )
     3. using the spherical excess formula for a quad one can find its area/solid_angle via
        A = p1 + p2 + p3 + p4 - 2 \pi
     4. The correction factor then simply is 1/A normalized by its maximum value.   
    """
    
    x00,x10,x01,x11 = pixel_corners[:,:-1,:-1],pixel_corners[:,1:,:-1],pixel_corners[:,:-1,1:],pixel_corners[:,1:,1:]
    
    normals = np.array((np.cross(x00,x10-x00),np.cross(x10,x11-x10),np.cross(x11,x01-x11),np.cross(x01,x00-x01)))
    normals /= np.linalg.norm(normals,axis=-1)[...,None]
    cos_phi = np.array(tuple(np.sum(a*b,axis=-1) for a,b in zip( normals,np.roll(normals,-1,axis=0) )))

    phi = np.arccos(-cos_phi)
    solid_angle = np.sum(phi,axis = 0)-2*np.pi
    solid_angle_correction = 1/solid_angle
    solid_angle_correction /= solid_corr.max()

    return solid_angle_correction



def determine_xray_polarization_correction(thetas,phis,mode='h'):
    '''
    determine polarization factor
    the experimental intensity should be multiplied by self.Pfactor[i,j]
    '''  
    # here 'h' and 'v' polarisations are implemented in the way to be compatible with array indexing (firt index - verical direction, second, horisontal)
    if mode =='v':
        polarization_factor = 1.0/(np.cos(thetas)**2+(np.sin(thetas)**2)(np.sin(phis)**2))
    elif mode =='h':
        polarization_factor = 1.0/(np.cos(thetas)**2+(np.sin(thetas)**2)(np.cos(phis)**2))
    return polarization_factor
