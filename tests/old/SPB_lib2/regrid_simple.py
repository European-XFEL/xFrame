import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi,heat2D,heatPolar2D,agipd_heat
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward,cartesian_to_spherical,spherical_to_cartesian
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward

import logging
#from xframe.plugins.fxs3046_online.analysisLibrary.regrid2 import SimpleRegridder2D,AgipdRegridderSimple

from scipy.spatial.kdtree import KDTree
from scipy.ndimage import gaussian_filter

log=logging.getLogger('root')
db = default_DB()

init = False
if init:
    from xframe.startup_routines import dependency_injection
    dependency_injection()
    from xframe.startup_routines import load_recipes
    from xframe.control.Control import Controller
    analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='mask_data_high',exp_name='fxs3046_online',exp_settings = 'default_proc')
    controller = Controller(analysis_sketch,experiment_sketch = exp_sketch)
    controller.chooseJobSingleProcess()
    from xframe import database
    from xframe import Multiprocessing



from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe.library.gridLibrary import GridFactory
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library import units
from xframe.library.gridLibrary import get_linspace_log2
from xframe.library.gridLibrary import uniformGrid_func

from scipy.spatial.kdtree import KDTree
from scipy.spatial import Delaunay 
from scipy.ndimage import gaussian_filter
from scipy.ndimage import spline_filter

def create_lookup_array_2D(grid,new_grid,return_counts = False,coord_sys='cartesian'):
    '''assumes grids are in cartesian_coordinates'''
    if coord_sys == 'spherical':        
        grid = spherical_to_cartesian(grid[:])
        new_grid = spherical_to_cartesian(new_grid[:])
    if grid.shape[-1] == 3:
        grid = grid[...,:2]
    shape = grid.shape
    new_shape = new_grid.shape
    tree = KDTree(grid.reshape(-1,2))
    lookup_ids = tree.query(new_grid.reshape(-1,2))[1]
    unraveld_lookup = np.unravel_index(lookup_ids,shape[:-1])
    unraveld_lookup = tuple(part.reshape(new_shape[:-1]) for part in unraveld_lookup)
        
    if return_counts:
        counts = np.zeros(new_shape[0])
        shaped_lookup = lookup_ids.reshape(new_shape[:-1])
        for n,x in enumerate(shaped_lookup):
            counts[n] = len(np.unique(x))
        return unraveld_lookup,counts
    else:
        return unraveld_lookup
    

class SimpleRegridder2D:
    def __init__(self,pixel_centers,new_grid,interpolation = 'nearest',coord_sys='spherical',interpolation_constants=False):
        ''' assumes pixel_centers in spherical coordinates'''
        self.out_shape = new_grid.shape[:-1]
        self.in_shape = pixel_centers.shape[:-1]
        self.pixel_centers = pixel_centers
        self.new_grid = new_grid
        self.interpolation = interpolation
                
        if coord_sys == 'spherical':
            self.pixel_centers = spherical_to_cartesian(pixel_centers)
            self.new_grid = spherical_to_cartesian(new_grid)
        self.pixel_centers_hash = hash(pixel_centers.data.tobytes())
        self.new_grid_hash = hash(new_grid.data.tobytes())
        
        try:
            self.interpolation_constants = getattr(self,'generate_interpolation_constants_'+interpolation)(interpolation_constants)
            self.apply = getattr(self,'generate_apply_'+interpolation)()
        except AttributeError as e:
            log.error('interpolation method {} not known. Available interpolation options are {}'.format(interpolation,'"nearest","linear"')) 

    def check_interpolation_constants(self,interpolation_constants):

        pixel_centers_hash = hash(self.pixel_centers.data.tobytes())
        new_grid_hash = hash(self.new_grid.data.tobytes())
        pixel_centers_match = pixel_centers_hash == interpolation_constants['hash_pixel_centers']
        new_grids_match = new_grid_hash == interpolation_constants['hash_new_grid']
        interpolation_match = self.interpolation == interpolation_constants['interpolation']
        constants_valid = (new_grids_match and pixel_centers_match and interpolation_match)
        #print('constants valid',constants_valid)
        #print('new_grid_match',new_grids_match)
        return constants_valid
    
    def assemble_interpolation_dict(self, data: dict):
        interpolation_constants={
            'hash_pixel_centers':hash(self.pixel_centers.data.tobytes()),
            'hash_new_grid':hash(self.new_grid.data.tobytes()),
            'interpolation':self.interpolation,
            **data
        }
        return interpolation_constants

    def generate_interpolation_constants_nearest(self,interpolation_constants):
        constants_fit_to_grids = False
        if isinstance(interpolation_constants,dict):
            constants_fit_to_grids = self.check_interpolation_constants(interpolation_constants)
        if not constants_fit_to_grids:
            lookup_array,counts = create_lookup_array_2D(self.pixel_centers,self.new_grid,return_counts = True)
            data = {'lookup_array':lookup_array}
            interpolation_constants = self.assemble_interpolation_dict(data)
        return interpolation_constants    
    def generate_interpolation_constants_linear(self,interpolation_constants):
        constants_fit_to_grids = False
        if isinstance(interpolation_constants,dict):
            constants_fit_to_grids = self.check_interpolation_constants(interpolation_constants)
        if not constants_fit_to_grids:
            vertices,weights = self.generate_linear_interp_weights(self.pixel_centers,self.new_grid)            
            data = {'vertices':vertices,'weights':weights}
            interpolation_constants = self.assemble_interpolation_dict(data)            
        return interpolation_constants

    def generate_linear_interp_weights(self,grid,new_grid):
        d = grid.shape[-1]
        g = grid.reshape(-1,d)
        ng = new_grid.reshape(-1,d)
        log.info('grid shape = {} new grid shape = {}'.format(g.shape,ng.shape))
        tri = Delaunay(g)
        simplex = tri.find_simplex(ng)
        outside_hull = (simplex == -1)
        vertices = tri.simplices[simplex]
        temp = tri.transform[simplex]
        delta = ng - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        weights[outside_hull] = 0
        return vertices, weights

    
    def generate_apply_linear(self):
        weights = self.interpolation_constants['weights']
        vertices = self.interpolation_constants['vertices']
        shape = self.out_shape
        ndim_in = len(self.in_shape)
        def apply(data):
            ndim = data.ndim
            N=data.shape[-1]
            if ndim == 1+ndim_in:
                return np.einsum('njk,nj->nk', data.reshape(-1,N)[vertices], weights).reshape(shape+(N,))
            else:                
                return np.einsum('nj,nj->n', data.reshape(-1)[vertices], weights).reshape(shape)
        return apply
    
    def generate_apply_nearest(self):
        lookup = self.interpolation_constants['lookup_array']
        def apply(data):
            if data.ndim == 1+len(lookup):
                return data[lookup]
            else:
                return data[lookup]
        return apply

class AgipdRegridderSimple:
    version = 0.1
    @staticmethod
    def get_interpolation_constants_name(geometry,new_grid_shape,interpolation):
        pixel_hash = hash(geometry['framed_pixel_centers'].data.tobytes())
        new_grid_hash = hash(new_grid_shape + (AgipdRegridderSimple.version,))
        return '{}_{}_{}'.format(interpolation,pixel_hash,new_grid_hash)

    
    def __init__(self,geometry,new_grid_shape, interpolation = 'nearest', interpolation_constants=False):        
        spherical_pixel_centers = geometry['framed_pixel_centers']
        self.default_interpolation_constants = {
            'data':{m_id:False for m_id in range(len(spherical_pixel_centers))},
            'mask':{m_id:False for m_id in range(len(spherical_pixel_centers))}
        }
        self.geometry = geometry
        self.sensitve_pixel_mask = geometry['framed_mask']
        self.interpolation = interpolation
        self.new_grid_shape = new_grid_shape
        self.max_pixel_q = np.max(spherical_pixel_centers[...,0])
        self.new_constants_generated = False
        self.generate_new_constants,self.interpolation_constants = self.process_interpolation_constants(interpolation_constants)
        polar_pixel_centers = spherical_pixel_centers[...,::2]
        cart_pixel_centers = spherical_to_cartesian(polar_pixel_centers)


        module_boundaries = []
        for dim in range(2):
            _min = np.min(cart_pixel_centers[:,...,dim],axis = (1,2))
            _max = np.max(cart_pixel_centers[:,...,dim],axis = (1,2))
            module_boundaries.append([_min,_max])        
        self.module_boundaries = np.moveaxis(np.array(module_boundaries),-1,0)
        
        self.new_grid = self._generate_new_grid(new_grid_shape,self.max_pixel_q)
        
        self.module_masks = self._generate_module_masks(self.new_grid,self.module_boundaries)

        temp = self._generate_regridders(self.new_grid,polar_pixel_centers,self.module_masks)
        self.data_regridders_per_module = temp[0]
        self.mask_regridders_per_module = temp[1]

        if self.generate_new_constants:
            self.interpolation_constants = self._assemble_interpolation_constants()
            self.new_constants_generated = True

        self.regrid = self.generate_regrid()

    def _generate_new_grid(self,new_grid_shape,max_q):
        Nq,Nphi = new_grid_shape
        qs = max_q*np.arange(Nq)/(Nq-1)
        phis = 2*np.pi*np.arange(Nphi)/Nphi
        new_grid = GridFactory.construct_grid('uniform',[qs,phis])[:]
        return new_grid

    def _generate_module_masks(self,new_grid,boundaries):
        cart_grid = spherical_to_cartesian(new_grid)

        min_mask = cart_grid>=boundaries[:,None,None,:,0]
        max_mask = cart_grid<=boundaries[:,None,None,:,1]
        self.min_mask = min_mask
        self.max_mask = max_mask
        module_masks = min_mask[...,0] & min_mask[...,1] & max_mask[...,0] & max_mask[...,1]        
        return module_masks
    
    def _generate_regridders(self,new_grid,polar_pixel_centers,module_masks):
        constants = self.interpolation_constants                
        data_regridders = []
        mask_regridders = []        
        for m_id,mask in enumerate(module_masks):
            module_pixel_centers = polar_pixel_centers[m_id]
            new_points = new_grid[mask]
                
            r = SimpleRegridder2D(module_pixel_centers,new_points,interpolation = self.interpolation,interpolation_constants = constants['data'][m_id])
            if self.interpolation == 'linear':
                rm = SimpleRegridder2D(module_pixel_centers,new_points,interpolation = 'nearest',interpolation_constants = constants['mask'][m_id])
            else:
                rm = r
            data_regridders.append(r)
            mask_regridders.append(rm)
        
        return data_regridders,mask_regridders
    def _assemble_interpolation_constants(self):
        data_constants = {m_id:r.interpolation_constants for m_id,r in enumerate(self.data_regridders_per_module)}
        mask_constants = {m_id:r.interpolation_constants for m_id,r in enumerate(self.mask_regridders_per_module)}
        hash_name = self.get_interpolation_constants_name(self.geometry,self.new_grid_shape,self.interpolation)        
        interpolation_constants={'data' : data_constants,'mask':mask_constants,'hash_name':hash_name}
        return interpolation_constants
    def process_interpolation_constants(self,constants):
        constants_are_valid = False
        if isinstance(constants,dict):            
            hash_name = self.get_interpolation_constants_name(self.geometry,self.new_grid_shape,self.interpolation)        
            constants_name = constants['hash_name']
            constants_are_valid = (hash_name == constants_name)
            
        if constants_are_valid:
            interpolation_constants = constants
        else:
            interpolation_constants = self.default_interpolation_constants
        return (not constants_are_valid),interpolation_constants
    def generate_regrid(self):
        in_mask = self.sensitve_pixel_mask
        in_shape = self.sensitve_pixel_mask.shape
        out_shape = self.new_grid_shape
        module_masks = self.module_masks
        data_regridders = self.data_regridders_per_module
        mask_regridders = self.mask_regridders_per_module
        def regrid(data,mask,modules,**args):
            ndim = data.ndim
            if ndim == len(in_mask):
                N = len(data)
                # have to copy data to framed agipd shape
                data_dtype = data.dtype
                mask_dtype = mask.dtype
                in_d = np.zeros(in_shape+(N,),data_dtype)
                in_m = np.zeros(in_shape+(N,),mask_dtype)
                out_d = np.zeros((N,)+out_shape,data_dtype)
                out_m = np.zeros((N,)+out_shape,mask_dtype)
                #print(in_d[:,in_mask].shape)
                #print(data.shape)            
                in_d[in_mask] = np.moveaxis(data,0,-1).reshape(-1,N)
                in_m[in_mask]= np.moveaxis(mask,0,-1).reshape(-1,N)
                for m in modules:
                    mask = module_masks[m]
                    out_d[:,mask]= np.moveaxis(data_regridders[m].apply(in_d[m]),-1,0)
                    out_m[:,mask]= np.moveaxis(mask_regridders[m].apply(in_m[m]),-1,0)                
            else:
                data_dtype = data.dtype
                mask_dtype = mask.dtype
                in_d = np.zeros(in_shape,data_dtype)
                in_m = np.zeros(in_shape,mask_dtype)
                out_d = np.zeros(out_shape,data_dtype)
                out_m = np.zeros(out_shape,mask_dtype)
                in_d[in_mask] = data.reshape(-1)
                in_m[in_mask]= mask.reshape(-1)
                for m in modules:
                    mask = module_masks[m]
                    out_d[mask]= data_regridders[m].apply(in_d[m])
                    out_m[mask]= mask_regridders[m].apply(in_m[m])                                
            return out_d,out_m
        return regrid
                
                


    
comm = Multiprocessing.comm_module
g = comm.get_geometry()

base_path = '/gpfs/exfel/theory_group/user/berberic/fxs3046/test/regrid2/'

pixel_centers = g['framed_pixel_centers'][:,::16,::16]
sensitive_pixel = g['framed_mask'][:,::16,::16]
data = np.zeros((2,16,32,8))
for module in data:
    for n,i in enumerate(module):
        i[:,n%2::2]=1.0
mask = np.full(data.shape,True,dtype = bool)

from scipy.interpolate import griddata

#cart_grid = GridFactory.construct_grid('uniform',[np.arange(-20,20,dtype = float)/5,np.arange(-20,20,dtype = float)/5])[:]
#polar_grid = GridFactory.construct_grid('uniform',[np.arange(40,dtype = float)/4,np.arange(1000)*2*np.pi/1000])[:]
#rg = SimpleRegridder2D(cart_grid,spherical_to_cartesian(polar_grid),interpolation='linear',coord_sys='cartesian')
#rg2 = SimpleRegridder2D(spherical_to_cartesian(polar_grid),cart_grid,interpolation='linear',coord_sys='cartesian')
#
#data = np.zeros((100,100))
#data[::2,1::2]=1
#
#data_polar = np.zeros(polar_grid.shape[:-1])
#data_polar[::2]=1
#new_data = rg2.apply(data_polar)
#polar_cart = spherical_to_cartesian(polar_grid)
#new_data2 = griddata((polar_cart[...,0].reshape(-1),polar_cart[...,1].reshape(-1)),data_polar.reshape(-1),(cart_grid[...,0].reshape(-1),cart_grid[...,1].reshape(-1)),fill_value = 0.0).reshape(new_data.shape)

#pixel_centers[...,1]+=np.pi/2
#pixel_centers[...,2]+=np.pi
gg = {'framed_pixel_centers':pixel_centers,'framed_mask':sensitive_pixel}
new_grid_shape = (512,1024*2)
ar = AgipdRegridderSimple(gg,new_grid_shape,interpolation='nearest',interpolation_constants=ic)


polar_data,polar_mask = ar.regrid(data[0],mask[0],np.arange(8))

#new_data = np.zeros_like(ar.new_grid[...,0])
#for mask in ar.module_masks:
#    new_data[mask]=1

base_path = '/gpfs/exfel/theory_group/user/berberic/p3046/tests/'
database.analysis.save(base_path+'regrid_simple.vts',[polar_data],grid=ar.new_grid,grid_type='polar')
#database.analysis.save(base_path+'regrid_simple3.vtr',[new_data2],grid=cart_grid,grid_type='cartesian')
#database.analysis.save(base_path+'regrid_simple2.vtr',[new_data],grid=cart_grid,grid_type='cartesian')


