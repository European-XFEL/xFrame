import time
import numpy as np
from scipy.interpolate import griddata
import logging
import traceback

log=logging.getLogger('root')

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
from xframe import Multiprocessing


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
    unraveld_lookup = np.array(tuple(part.reshape(new_shape[:-1]) for part in unraveld_lookup))
        
    if return_counts:
        counts = np.zeros(new_shape[0])
        shaped_lookup = lookup_ids.reshape(new_shape[:-1])
        for n,x in enumerate(shaped_lookup):
            counts[n] = len(np.unique(x))
        return unraveld_lookup,counts
    else:
        return unraveld_lookup
    

def get_min_diffs(array):
    min_diffs = []
    for dim in range(array.ndim-1):
        min_diff = np.abs(np.diff(array[...,dim],axis = dim)).min()
        min_diffs.append(min_diff)
    return np.array(min_diffs)

def get_max_diffs(array):
    max_diffs = []
    for dim in range(array.ndim-1):
        max_diff = np.abs(np.diff(array[...,dim],axis = dim)).max()
        max_diffs.append(max_diff)
    return np.array(max_diffs)



class SimpleRegridder2D:
    def __init__(self,pixel_centers,new_grid,interpolation = 'nearest',coord_sys='spherical',interpolation_constants=False):
        ''' assumes pixel_centers in spherical coordinates'''
        self.out_shape = new_grid.shape[:-1]
        self.in_shape = pixel_centers.shape[:-1]
        self.pixel_centers = pixel_centers
        self.new_grid = new_grid
        self.interpolation = interpolation
        self.interpolation_constants = interpolation_constants
        if coord_sys == 'spherical':
            self.pixel_centers = spherical_to_cartesian(pixel_centers)
            self.new_grid = spherical_to_cartesian(new_grid)
        self.pixel_centers_hash,self.new_grid_hash = self.generate_hashes(self.pixel_centers,self.new_grid)
        
        try:
            self.interpolation_constants = self.generate_interpolation_constants()
        except AttributeError as e:
            traceback.print_exc()
            log.error('interpolation method {} not known. Available interpolation options are {}'.format(interpolation,'"nearest","linear"')) 
        self.apply = getattr(self,'generate_apply_'+interpolation)()
        
    def generate_hashes(self,pixel_centers,new_grid):
        pixel_centers_hash = hash(tuple(pixel_centers.reshape(-1)[:1000]))
        new_grid_hash = hash(tuple(new_grid.reshape(-1)[:1000]))
        return pixel_centers_hash,new_grid_hash
    def check_interpolation_constants(self,interpolation_constants):
        pixel_centers_match = self.pixel_centers_hash == interpolation_constants['hash_pixel_centers']
        new_grids_match = self.new_grid_hash == interpolation_constants['hash_new_grid']

        new_interpolation = interpolation_constants['interpolation']
        if isinstance(new_interpolation,bytes):
            new_interpolation = new_interpolation.decode()
        interpolation_match = self.interpolation == new_interpolation
        constants_valid = (new_grids_match and pixel_centers_match and interpolation_match)
        #log.info('constants valid {}'.format(constants_valid))
        #log.info('new_grid hash {} {}'.format(self.new_grid_hash,interpolation_constants['hash_new_grid']))
        #log.info('pixel_centers hash {} {}'.format(self.pixel_centers_hash,interpolation_constants['hash_pixel_centers']))
        return constants_valid
    
    def assemble_interpolation_dict(self, data: dict):
        interpolation_constants={
            'hash_pixel_centers': self.pixel_centers_hash,
            'hash_new_grid': self.new_grid_hash,
            'interpolation':self.interpolation,
            **data
        }
        return interpolation_constants

    def generate_interpolation_constants(self):
        constants_fit_to_grids = False
        interpolation_constants = self.interpolation_constants
        interpolation = self.interpolation
        if isinstance(interpolation_constants,dict):
            constants_fit_to_grids = self.check_interpolation_constants(interpolation_constants)
        if not constants_fit_to_grids:
            try:
                interpolation_constants = getattr(self,'generate_interpolation_constants_'+interpolation)(interpolation_constants)
            except AttributeError as e:
                traceback.print_exc()
                log.error('interpolation method {} not known. Available interpolation options are {}'.format(interpolation,'"nearest","linear"'))
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
        #log.info('inerpolation constants type is {}'.format(type(interpolation_constants)))
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
        #log.info('grid shape = {} new grid shape = {}'.format(g.shape,ng.shape))
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
            if ndim >= 1+ndim_in:
                invariant_shape = data.shape[:-2]
                N_invar = np.prod(invariant_shape)
                data = np.moveaxis(np.moveaxis(data,-1,0),-1,0)
                return np.einsum('njk,nj->nk', data.reshape(-1,N_invar)[vertices], weights).T.reshape(invariant_shape+shape)
            else:                
                return np.einsum('nj,nj->n', data.reshape(-1)[vertices], weights).reshape(shape)
        return apply
    
    def generate_apply_nearest(self):
        lookup = self.interpolation_constants['lookup_array'].astype(int)
        def apply(data):
            if data.ndim == 1+len(lookup):
                return data[tuple(lookup)]
            else:
                return data[tuple(lookup)]
        return apply

class AgipdRegridderSimple:
    # Regridder for AGIPD data in the 16,512,128 format.
    # Uses a SimpleRegridder2D object for each of the 16 modules separatelly.    
    version = 0.3
    @staticmethod
    def get_interpolation_constants_name(geometry,new_grid_shape,interpolation):

        pixel_hash = hash(tuple(geometry['framed_pixel_centers'].reshape(-1)[:1000]))
        #log.info('new_grid shape = {}'.format(new_grid_shape))
        new_grid_hash = hash(new_grid_shape + (AgipdRegridderSimple.version,))
        return '{}_{}_{}'.format(interpolation,pixel_hash,new_grid_hash)
    
    
    def __init__(self,geometry,new_grid_shape, interpolation = 'nearest', interpolation_constants=False):        
        spherical_pixel_centers = geometry['framed_pixel_centers']
        self.default_interpolation_constants = {
            'data':{str(m_id):False for m_id in range(len(spherical_pixel_centers))},
            'mask':{str(m_id):False for m_id in range(len(spherical_pixel_centers))},
            'module_masks':np.array([False]*len(spherical_pixel_centers)),
            'hash_name': 'not assigned'
        }
        #self.pixel_centers = geometry['framed_pixel_centers']
        self.geometry = geometry
        self.sensitve_pixel_mask = geometry['framed_mask']
        self.interpolation = interpolation
        self.new_grid_shape = new_grid_shape
        self.max_pixel_q = np.max(spherical_pixel_centers[...,0])
        self.polar_pixel_centers = spherical_pixel_centers[...,::2]
        self.cart_pixel_centers = spherical_to_cartesian(self.polar_pixel_centers)
        #self.polar_pixel_centers = cartesian_to_spherical(self.cart_pixel_centers)
        #cart_pixel_centers = spherical_to_cartesian(polar_pixel_centers)
        
        self.new_grid = self._generate_new_grid(new_grid_shape,self.max_pixel_q)
        self.new_qs = self.new_grid[:,0,0]
        self.new_phis = self.new_grid[0,:,1]

        self.generate_new_constants,self.interpolation_constants = self.process_interpolation_constants(interpolation_constants)
        if self.generate_new_constants:
            self.module_masks = self._generate_module_masks()
            self.interpolation_constants['module_masks']=self.module_masks
            temp = self._generate_regridders()
            self.data_regridders_per_module = temp[0]
            self.mask_regridders_per_module = temp[1]
            self.interpolation_constants = self._assemble_interpolation_constants()
        else:            
            temp = self._generate_regridders()
            self.data_regridders_per_module = temp[0]
            self.mask_regridders_per_module = temp[1]
        self.regrid = self.generate_regrid()

    def _generate_new_grid(self,new_grid_shape,max_q):
        Nq,Nphi = new_grid_shape
        qs = max_q*np.arange(Nq)/(Nq-1)
        phis = 2*np.pi*np.arange(Nphi)/Nphi
        new_grid = GridFactory.construct_grid('uniform',[qs,phis])[:]
        return new_grid

    def _generate_module_masks(self):
        def delaunay_mask_finding(module_ids,rough_module_masks,cart_new_grid,in_masks,**kwargs):
            module_masks = [False]*len(module_ids)
            
            #log.info(f'module_ids = {module_ids}')
            #log.info(f'module_masks shape  = {rough_module_masks.shape}')
            #log.info(f'cart_new_grid shape  = {cart_new_grid.shape}')
            for _id,mid in enumerate(module_ids):
                centers = self.cart_pixel_centers[mid]
                #delaunay = Delaunay(centers.reshape(-1,2))
                rough_module_mask = rough_module_masks[mid]
                in_mask = in_masks[mid]
                new_points = cart_new_grid[rough_module_mask]
                regridder = SimpleRegridder2D(centers,new_points,coord_sys='cartesian')
                mask = rough_module_mask.copy()
                mask[mask] = regridder.apply(in_mask).astype(bool)                
                module_masks[_id] = mask                
            return np.asarray(module_masks)
        new_grid = self.new_grid
        in_masks = np.ones(self.sensitve_pixel_mask.shape,dtype = bool)
        in_masks[:,0,:]=0
        in_masks[:,-1,:]=0
        in_masks[:,:,0]=0
        in_masks[:,:,-1]=0
        # calculate rough module boundary box
        module_boundaries = []
        for dim in range(2):
            _min = np.min(self.cart_pixel_centers[:,...,dim],axis = (1,2))
            _max = np.max(self.cart_pixel_centers[:,...,dim],axis = (1,2))
            #log.info(f'dim {dim} : min {_min} max {_max} ')
            module_boundaries.append([_min,_max])        
        module_boundaries = np.moveaxis(np.array(module_boundaries),-1,0)        
        n_modules = module_boundaries.shape[0]

        # improve module masks by delaunay triangulation.
        # This is to avoid overlapping module masks.
        cart_grid = spherical_to_cartesian(new_grid)
        min_mask = cart_grid>=module_boundaries[:,None,None,:,0]
        max_mask = cart_grid<=module_boundaries[:,None,None,:,1]
        temp_module_masks = min_mask[...,0] & min_mask[...,1] & max_mask[...,0] & max_mask[...,1]
        module_masks = Multiprocessing.comm_module.request_mp_evaluation(delaunay_mask_finding,input_arrays=[np.arange(n_modules)],const_inputs = [temp_module_masks,cart_grid,in_masks] ,call_with_multiple_arguments = True,split_mode='modulus',n_processes= False)
        #log.info(f'module_masks .shape = {module_masks.shape} | {module_masks.dtype}')
        return module_masks
    
    def _generate_regridders(self):
        new_grid = self.new_grid
        polar_pixel_centers = self.polar_pixel_centers
        constants = self.interpolation_constants
        module_masks = constants['module_masks']
        #log.info(f'masks .shape = {module_masks.shape}')
        data_regridders = {}
        mask_regridders = {}        
        for m_id,mask in enumerate(module_masks):
            module_pixel_centers = polar_pixel_centers[m_id]
            new_points = new_grid[mask]
                
            r = SimpleRegridder2D(module_pixel_centers,new_points,interpolation = self.interpolation,interpolation_constants = constants['data'][str(m_id)])
            if self.interpolation == 'linear':
                rm = SimpleRegridder2D(module_pixel_centers,new_points,interpolation = 'nearest',interpolation_constants = constants['mask'][str(m_id)])
            else:
                rm = r
            data_regridders[str(m_id)] = r
            mask_regridders[str(m_id)] = rm        
        return data_regridders,mask_regridders
    def _assemble_interpolation_constants(self):
        data_constants = {str(m_id):r.interpolation_constants for m_id,r in self.data_regridders_per_module.items()}
        mask_constants = {str(m_id):r.interpolation_constants for m_id,r in self.mask_regridders_per_module.items()}
        hash_name = self.get_interpolation_constants_name(self.geometry,self.new_grid_shape,self.interpolation)        
        interpolation_constants={'data' : data_constants,'mask':mask_constants,'hash_name':hash_name,'module_masks':self.module_masks}
        return interpolation_constants
    def process_interpolation_constants(self,constants):
        constants_are_valid = False
        if isinstance(constants,dict):            
            hash_name = self.get_interpolation_constants_name(self.geometry,self.new_grid_shape,self.interpolation)                
            constants_name = constants['hash_name']
            if isinstance(constants_name,bytes):
                constants_name = constants_name.decode()
            constants_are_valid = (hash_name == constants_name)
            #log.info('hash_names = {} {}'.format(hash_name,constants_name))
        if constants_are_valid:
            constants['module_masks'] = np.asarray(constants['module_masks'],dtype = bool) 
            interpolation_constants = constants
        else:
            interpolation_constants = self.default_interpolation_constants
        return (not constants_are_valid),interpolation_constants
    def generate_regrid(self):
        in_mask = self.sensitve_pixel_mask
        in_shape = self.sensitve_pixel_mask.shape
        out_shape = self.new_grid_shape
        module_masks = self.interpolation_constants['module_masks']
        data_regridders = self.data_regridders_per_module
        mask_regridders = self.mask_regridders_per_module
        new_masks = [1]*len(module_masks)
        def regrid(data,mask,modules,**args):
            ndim = data.ndim
            #log.info('regrid ndim ={}'.format(ndim))
            #log.info('in mask shape ={}'.format(in_mask.shape))
            if ndim == in_mask.ndim+1:
                N = len(data)
                # have to copy data to framed agipd shape
                data_dtype = data.dtype
                mask_dtype = mask.dtype
                in_d = np.zeros(in_shape+(N,),data_dtype)
                in_m = np.zeros(in_shape+(N,),mask_dtype)
                out_d = np.zeros((N,)+out_shape,data_dtype)
                out_m = np.zeros((N,)+out_shape,mask_dtype)
                                
                in_d[in_mask] = np.moveaxis(data,0,-1).reshape(-1,N)
                in_m[in_mask]= np.moveaxis(mask,0,-1).reshape(-1,N)
                for m in modules:
                    mask = module_masks[m]
                    out_d[:,mask]= np.moveaxis(data_regridders[str(m)].apply(in_d[m]),-1,0)
                    out_m[:,mask]= np.moveaxis(mask_regridders[str(m)].apply(in_m[m]),-1,0)                
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
                    #log.info(f'mask shape = {mask.shape}')
                    #log.info(f'masks shape = {module_masks.shape}')
                    out_d[mask]= data_regridders[str(m)].apply(in_d[m])
                    out_m[mask]= mask_regridders[str(m)].apply(in_m[m])
            return out_d,out_m
        return regrid
                

### old stuff ###
class NearestNeighbourAgipdRegridder:
    def __init__(self,pixel_centers,polar_grid):
        ''' assumes pixel_centers in spherical coordinates'''
        self.cart_pixel_centers = spherical_to_cartesian(pixel_centers)
        self.cart_grid = spherical_to_cartesian(self.polar_grid)
        self.lookup_array,self.counts = create_lookup_array_2D(self.cart_pixel_centers,self.cart_grid,return_counts = True)
        self.apply = self.generate_apply()
    def generate_apply(self):
        lookup = self.lookup_array
        def apply(data):
            return data[lookup]
        return apply
    def generate_regrid_with_gaussian(self,intermediate_sampling=1,sigma=1):
        '''
        input grid -> intermediate uniform grid ---[gaussian_filter]---> intermediate uniform grid -> output grid  
        intermediate_sampling defines the step size of the intermediate uniform grid in terms of multiples of the minimalpoint distance in the input grid.
        sigma defines the standard deviation of the gaussian filter in units of the maximal point distance in the inputgrid.
        Assumes pixel centers to be of shape [modules,x,y,point]
        '''
        assert intermediate_sampling>=1 , 'intermediate_sampling has to be greater than 1'
        min_diff = np.min([get_min_diffs(m) for m in self.cart_pixel_centers])/intermediate_sampling
        max_diff = np.max([get_max_diffs(m) for m in self.cart_pixel_centers],axis = 0)
        max_diff = np.max(max_diff[0]/2,max_diff[1]) # accounting for the dead wide pixels in AGIPD        
        bounds = np.max(cart_pixel_centers,axis = (0,1,2))[:2]
        n_steps = (bounds//min_diff + 1).astype(int)
        xs = np.arange(n_teps[0])*bounds[0]/n_steps[0]
        ys = np.arange(n_teps[1])*bounds[1]/n_steps[1]
        intermediate_grid = GridFactory('uniform',[xs,ys])
        lookup_inter = create_lookup_array_2D(self.cart_pix_centers,intermediate_grid)
        lookup_polar = create_lookup_array_2D(intermediate_grid,self.polar_grid)
        sigma = max_diff*sigma

        def apply_with_gaussian(data):
            intermediate_data = data[lookup_inter]
            intermediate_data = gaussian_filter(intermediate_data,sigma = sigma)
            return intermediate_data[lookup_polar]
        return apply_with_gaussian
                
class AgipdRegridder:
    def __init__(self,db,geometry,wavelength, file_name='regrid_data', save_regrid_data=False, generate_data=False):
        #framed_pixel_grid in spherical_coordinates
        self.pixel_grid_spher = geometry['framed_pixel_grid']
        self.pixel_grid_spher[...,0]*=units.standardLength
        self.lab_pixel_grid_cart = geometry['framed_lab_pixel_grid']
        self.pixel_mask = geometry['framed_mask']
        self.data_shape = geometry['data_shape']
        self.wavelength = wavelength
        self.file_name = file_name
        self.db=db
        if generate_data:
            self.regrid_data=self.generate_regrid_data()
            if save_regrid_data:
                db.save(file_name,self.regrid_data)
        else:
            try:
                self.regrid_data=db.load(file_name)
            except FileNotFoundError as e:
                log.warning('regrid_data file not found. Proceed by generating regrid_data.')
                self.regrid_data=self.generate_regrid_data()
                if save_regrid_data:
                    db.save(file_name,self.regrid_data)

    def approximate_pixel_size(self,grid_cart):
        #absolute minimal x-distance between consecutive agipd pixels along vertical(short direction 128 pixels per module) pixel row
        x_min = np.abs([np.diff(grid_cart[m][:,:,0],axis=1).min() for m in np.arange(len(grid_cart))]).min()
        #same but in y-direction
        y_min = np.abs([np.diff(grid_cart[m][:,:,1],axis=1).min() for m in np.arange(len(grid_cart))]).min()
        #same but in z-direction
        z_min = np.abs([np.diff(grid_cart[m][:,:,2],axis=1).min() for m in np.arange(len(grid_cart))]).min()        
        approximated_pixel_size= np.max([x_min,y_min,z_min])
        log.info('approximate_pixel_size = {}'.format(approximated_pixel_size))
        return approximated_pixel_size


    def generate_regrid_data(self):
        pixel_grid_cart = spherical_to_cartesian(self.pixel_grid_spher)
        data = self.generate_regrid_data_polar(self.pixel_grid_spher,grid_point_multiplicator = [3,1])    
        data2 = self.generate_regrid_data_cart(self.lab_pixel_grid_cart,grid_point_multiplicator = [4,4])
        lab_small = self.generate_regrid_data_cart(self.lab_pixel_grid_cart,grid_point_multiplicator = [2,2])
        q_cart_small = self.generate_regrid_data_cart(pixel_grid_cart,grid_point_multiplicator = [2,2])
        
        cart_data = {'lab':data2,'lab_small':lab_small,'q_small':q_cart_small}
        polar_data = {'reciprocal':data}
        regrid_data = {'cart':cart_data,'polar':polar_data}
        return regrid_data


    def generate_regrid_data_cart(self,pixel_grid,grid_point_multiplicator=[4,4],on_q_space=False):
        regrid_data={}
        d_shape = self.data_shape
        
        pix_size=self.approximate_pixel_size(pixel_grid)        
        
        old_global_x_bounds = (pixel_grid[...,0].min(),pixel_grid[...,0].max())
        old_global_y_bounds = (pixel_grid[...,1].min(),pixel_grid[...,1].max())

        try:
            assert (old_global_x_bounds[0]<=0) and (old_global_x_bounds[1]>0), 'global x bounds: ( {} ) do not contain 0.'.format(old_global_x_bounds)
            assert (old_global_y_bounds[0]<=0) and (old_global_y_bounds[1]>0), 'global y bounds: ( {} ) do not contain 0.'.format(old_global_y_bounds)
        except AssertionError as e:
            log.error(e)
            traceback.print_exc()
            raise e

        target_x_step = pix_size/grid_point_multiplicator[0]
        xs,x_param = get_linspace_log2(old_global_x_bounds,target_x_step,additional_steps = 1,return_parameters = True)
        x_step = x_param['step_size']
        global_x_bounds = x_param['domain']

        target_y_step = pix_size/grid_point_multiplicator[1]
        ys,y_param = get_linspace_log2(old_global_y_bounds,target_y_step,additional_steps = 1, return_parameters = True)
        y_step = y_param['step_size']
        global_y_bounds = y_param['domain']

        out_grid = GridFactory.construct_grid('uniform',[xs,ys]).array
        
        regrid_data['x_step'] = x_step
        regrid_data['xs'] = xs
        regrid_data['y_step'] = y_step
        regrid_data['ys'] = ys
        regrid_data['grid'] = out_grid

        global_x_min = global_x_bounds[0]
        global_y_min = global_y_bounds[0]
        log.info('out grid shape = {}'.format(out_grid.shape))
        for m,g in enumerate(pixel_grid):
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]
        
            x_bounds = [g[...,0].min(),g[...,0].max()]            
            x_min_index = max(0,int((x_bounds[0]-global_x_min)//x_step)+1)
            x_max_index = min(int((x_bounds[1]-global_x_min)//x_step)+1,out_grid.shape[0])        

            y_bounds = [g[...,1].min(),g[...,1].max()]
            #log.info('x_bounds = {}'.format(x_bounds))
            y_min_index = max(int((y_bounds[0]-global_y_min)//y_step)+1,0)
            y_max_index = min(int((y_bounds[1]-global_y_min)//y_step)+1,out_grid.shape[1])        
            m_grid = out_grid[x_min_index:x_max_index,y_min_index:y_max_index]
            log.info("x_range = {} y_range ={}".format([x_max_index,x_min_index],[y_max_index,y_min_index]))
            #log.info("x_range = {} y_range ={}".format([x_max_index-x_min_index],[y_max_index-y_min_index]))
            log.info('mgrid.shape = {}'.format(m_grid.shape))
            old_m_grid =g[...,:2]
            old_index = np.zeros(g.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
        
            index = griddata(old_m_grid.reshape(-1,2),old_index.flatten(),m_grid.reshape(-1,2),method='nearest',rescale = True).reshape(m_grid.shape[:-1])
            #log.info('index.shape = {}'.format(index.shape))
            #log.info('ranges y: {} x:{}'.format(y_max_index-y_min_index,x_max_index - x_min_index))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['x_range']=[x_min_index,x_max_index]
            m_dict['y_range']=[y_min_index,y_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data

    def generate_regrid_data_polar(self,pixel_grid,grid_point_multiplicator=[3,1],on_q_space=False):
        regrid_data={}
        d_shape = self.data_shape

        pixel_grid_c = spherical_to_cartesian(pixel_grid)
        pix_size=self.approximate_pixel_size(pixel_grid_c)
        
        old_global_r_bounds = (pixel_grid[...,0].min(),pixel_grid[...,0].max())
        old_global_phi_bounds = (pixel_grid[...,2].min(),pixel_grid[...,2].max())
        log.info('phi bounds = {}'.format(old_global_phi_bounds))

        target_r_step = pix_size/grid_point_multiplicator[0]
        r_length = old_global_r_bounds[1]-old_global_r_bounds[0]
        n_r_steps = int(r_length//target_r_step)
        r_step = r_length/(n_r_steps-1)
        rs = old_global_r_bounds[0]+ np.arange(n_r_steps)*r_step
        global_r_bounds = old_global_r_bounds
        log.info('r bounds = {} max,min = {}'.format(global_r_bounds,(rs.min(),rs.max())))


        # phi_step such that the arc length at max_x is proportional to the approximated pixel size    
        target_phi_step = pix_size/(global_r_bounds[1]*grid_point_multiplicator[1])
        phis,phi_param = get_linspace_log2(old_global_phi_bounds,target_phi_step,additional_steps = 1, return_parameters = True)
        phi_step = phi_param['step_size']
        global_phi_bounds = phi_param['domain']
        log.info('phi bounds = {} max,min = {}'.format(global_phi_bounds,(phis.min(),phis.max())))
                
        thetas = ewald_sphere_theta(self.wavelength,rs)
        tmp_grid = GridFactory.construct_grid('uniform',[rs,phis]).array
        out_grid = np.zeros(tmp_grid.shape[:-1]+(tmp_grid.shape[-1]+1,))
        out_grid[...,[0,2]] = tmp_grid
        out_grid[...,1] = thetas[:,None]
        
        
        regrid_data['r_step'] = r_step
        regrid_data['rs'] = rs
        regrid_data['phi_step'] = phi_step
        regrid_data['phis'] = phis
        regrid_data['grid'] = out_grid

        global_r_min = global_r_bounds[0]
        global_phi_min = global_phi_bounds[0]
        log.info('out grid shape = {}'.format(out_grid.shape))
        
        out_grid_c = spherical_to_cartesian(out_grid)[...,:2]
        pixel_grid_c = spherical_to_cartesian(pixel_grid)[...,:2]
        for m,g in enumerate(pixel_grid):            
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]

            g_c = pixel_grid_c[m]
        
            r_bounds = [g[...,0].min(),g[...,0].max()]            
            r_min_index = max(int((r_bounds[0]-global_r_min)//r_step)+1,0)
            r_max_index = min(int((r_bounds[1]-global_r_min)//r_step)+1,out_grid.shape[0])        

            phi_bounds = [g[...,2].min(),g[...,2].max()]
            #log.info('global phi_min = {}'.format(global_phi_min))
            #log.info('phi_bounds = {}'.format(phi_bounds))
            phi_min_index = max(int((phi_bounds[0]-global_phi_min)//phi_step)+1,0)
            phi_max_index = min(int((phi_bounds[1]-global_phi_min)//phi_step)+1,out_grid.shape[1])     
            out_g_c = out_grid_c[r_min_index:r_max_index,phi_min_index:phi_max_index]
            #log.info('min - glob =  {}'.format((phi_bounds[0]-global_phi_min)//phi_step+1))
            #log.info('max - glob =  {}'.format((phi_bounds[1]-global_phi_min)//phi_step+1))
            #log.info('out_grid shape = {}'.format(out_grid.shape))
            #log.info("r_range = {} phi_range ={}".format([r_max_index,r_min_index],[phi_max_index,phi_min_index]))
            #log.info("x_range = {} y_range ={}".format([x_max_index-x_min_index],[y_max_index-y_min_index]))
            log.info('mgrid.shape = {}'.format(out_g_c.shape))
            old_index = np.zeros(g.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
        
            index = griddata(g_c.reshape(-1,2),old_index.flatten(),out_g_c.reshape(-1,2),method='nearest',rescale = True).reshape(out_g_c.shape[:-1])
            #log.info('index nonzero: {}'.format((index!=0).any()))
            #log.info('index.shape = {}'.format(index.shape))
            #log.info('ranges y: {} x:{}'.format(y_max_index-y_min_index,x_max_index - x_min_index))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['r_range']=[r_min_index,r_max_index]
            m_dict['phi_range']=[phi_min_index,phi_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data    
    
    def generate_regrid_data_polar_old(self,pixel_grid,grid_point_multiplicator=[2,1]):
        regrid_data={}
        d_shape = self.data_shape

        grid_c = spherical_to_cartesian(pixel_grid)
        pix_size=self.approximate_pixel_size(grid_c)
        
        r_max = pixel_grid[...,0].max()
        # r_step such that it is proportional to the approximated pixel size
        target_r_step = (pix_size/grid_point_multiplicator[0])
        n_new_rs = (r_max//target_r_step)
        r_step = r_max/n_new_rs
        rs= np.arange(n_new_rs)*r_step

        thetas = ewald_sphere_theta(self.wavelength,rs)

        # phi_step such that the arc length at max_x is proportional to the approximated pixel size
        
        target_phi_step = pix_size/(r_max*grid_point_multiplicator[1])
        phis, phi_param = get_linspace_log2([0,2*np.pi -target_phi_step],target_phi_step,additional_steps = 1,return_parameters = True)
        phi_step = phi_param['step_size']

        tmp_grid = GridFactory.construct_grid('uniform',[rs,phis]).array
        out_grid = np.zeros(tmp_grid.shape[:-1]+(tmp_grid.shape[-1]+1,))
        out_grid[...,[0,2]] = tmp_grid
        out_grid[...,1] = thetas[:,None]
        
        regrid_data['r_step'] = r_step
        regrid_data['rs'] = rs
        regrid_data['phi_step'] = phi_step
        regrid_data['phis'] = phis
        regrid_data['grid'] = out_grid
        log.info('out grid shape = {}'.format(out_grid.shape))
        log.info('out grid shape = {}'.format(out_grid.shape))
        grid_c = spherical_to_cartesian(out_grid)[...,:2]
        pixel_grid_c = spherical_to_cartesian(pixel_grid)[...,:2]
        
        for m,g in enumerate(pixel_grid):
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            #log.info('g shape = {}'.format(g.shape))
            g_c=pixel_grid_c[m]
            #log.info('g_c shape = {}'.format(g_c.shape))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]
        
            r_bounds = [g[...,0].min(),g[...,0].max()]
            r_min_index = int(r_bounds[0]//r_step)
            r_max_index = int(r_bounds[1]//r_step)        
            
            phi_bounds = [g[...,2].min(),g[...,2].max()]
            #log.info('phi_bounds = {}'.format(phi_bounds))
            phi_min_index = int((np.pi+phi_bounds[0])//phi_step)
            phi_max_index = int((np.pi+phi_bounds[1])//phi_step)        
            m_grid = grid_c[r_min_index:r_max_index,phi_min_index:phi_max_index]
            log.info("r_range = {} phi_range ={}".format([r_min_index,r_max_index],[phi_min_index,phi_max_index]))
            #log.info('mgrid.shape = {}'.format(m_grid.shape))
            old_m_grid =g_c
            old_index = np.zeros(g_c.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
            log.info('old index nonzero = {}'.format((old_index !=0).any()))
            #log.info('old m grid  = {}'.format(old_m_grid.shape))
            
            index = griddata(old_m_grid.reshape(-1,2),old_index.flatten(),m_grid.reshape(-1,2),method='nearest',rescale = True).reshape(m_grid.shape[:-1])
            log.info('index nonzero = {}'.format((index!=0).any()))
            #log.info('ranges r: {} phi:{}'.format(r_max_index-r_min_index,phi_max_index - phi_min_index))
            #log.info('index.shape = {}'.format(index.shape))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['r_range']=[r_min_index,r_max_index]
            m_dict['phi_range']=[phi_min_index,phi_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data
    
    def generate_and_save_regrid_data(self):
        self.regrid_data=self.generate_regrid_data()
        self.db.save(self.file_name,self.regrid_data)

        

    def put_data_into_grid(self,data,out_grid,r_range,phi_range):
        new_data=np.zeros((data.shape[0],)+out_grid.shape[:-1])
        #log.info('new data shape = {}'.format(new_data.shape))
        #log.info("r_range = {} phi_range ={}".format(r_range,phi_range))
        #log.info("len r = {} len phi ={} data shape = {}".format(np.diff(r_range),np.diff(phi_range),data.shape))
        new_data[:,r_range[0]:r_range[1],phi_range[0]:phi_range[1]] = data
        return new_data
        
    def regrid_polar(self,data, combine_modules=False, modules_on_full_grid=False,grid_type='reciprocal'):
        regrid_polar_dict = self.regrid_data['polar']
        for known_grid_type in regrid_polar_dict:
            if grid_type == known_grid_type:
                regrid_data = regrid_polar_dict[known_grid_type]
                break
            
        out_grid = regrid_data['grid']
        out_grid_max_index = regrid_data['grid']
        
        out_data=[]
        for m in np.arange(len(data)):
            #log.info("m = {} of {}".format(m,len(data)))
            valid_data_mask=regrid_data[str(m)]['valid_data_mask'].astype(bool)
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            valid_index=regrid_data[str(m)]['valid_index']
            #log.info('valid_index = {}'.format(valid_index.shape))

            if len(data.shape) == len(self.data_shape):
                data = data[None,...]
            regridded_m_data=np.zeros((data.shape[0],)+valid_data_mask.shape)
            regridded_m_data[:,valid_data_mask]=data[:,m].reshape(data.shape[0],-1)[:,valid_index]
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            if (combine_modules or modules_on_full_grid):
                r_range = regrid_data[str(m)]['r_range']
                phi_range = regrid_data[str(m)]['phi_range']
                regridded_m_data = self.put_data_into_grid(regridded_m_data,out_grid,r_range,phi_range)[:,:-1,:-1]
            out_data.append(regridded_m_data)            
            
        if combine_modules:
            out_data = np.sum(out_data,axis=0)
        if modules_on_full_grid:
            out_data = np.array(out_data)
        if isinstance(out_data,np.ndarray):
            out_data = np.squeeze(out_data)
            
        return out_data
    
    def regrid_cart(self,data, combine_modules=False, modules_on_full_grid=False,grid_type='lab'):
        regrid_cart_dict = self.regrid_data['cart']
        #log.info(regrid_cart_dict.keys())
        for known_grid_type in regrid_cart_dict:
            if grid_type == known_grid_type:
                regrid_data = regrid_cart_dict[known_grid_type]
                break
        
        out_grid = regrid_data['grid']
        out_data=[]
        for m in np.arange(len(data)):
            #log.info("m = {} of {}".format(m,len(data)))
            valid_data_mask=regrid_data[str(m)]['valid_data_mask'].astype(bool)
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            valid_index=regrid_data[str(m)]['valid_index']
            #log.info('valid_index = {}'.format(valid_index.shape))
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            #log.info('regridded_m_data shape = {}'.format(regridded_m_data[~valid_data_mask].shape))
            if len(data.shape) == len(self.data_shape):
                data = data[None,...]
            regridded_m_data=np.zeros((data.shape[0],)+valid_data_mask.shape)
            regridded_m_data[:,valid_data_mask]=data[:,m].reshape(data.shape[0],-1)[:,valid_index]
            #log.info("regridded shape = {}".format(regridded_m_data.shape))
            
            if (combine_modules or modules_on_full_grid):
                r_range = regrid_data[str(m)]['x_range']
                phi_range = regrid_data[str(m)]['y_range']
                #log.info("x_range: {} y_range: {}".format(r_range,phi_range))
                regridded_m_data = self.put_data_into_grid(regridded_m_data,out_grid,r_range,phi_range)[:,:-1,:-1]                
                
            out_data.append(regridded_m_data)            
            
        if combine_modules:
            out_data = np.sum(out_data,axis=0)
        if modules_on_full_grid:
            out_data = np.array(out_data)
        if isinstance(out_data,np.ndarray):
            out_data = np.squeeze(out_data)
            
        return out_data
