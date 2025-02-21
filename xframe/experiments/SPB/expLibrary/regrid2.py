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


def create_lookup_array_2D(grid,new_grid,in_mask=True,data_shape=None,return_counts = False,coord_sys='cartesian'):
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
    out_mask=True
    #print(f'in_mask type = {type(in_mask)}')
    if isinstance(in_mask,np.ndarray):
        assert np.sum(in_mask.astype(int))==np.prod(data_shape),f'True values in in_mask do not match with given data_shape, { np.sum(in_mask.astype(int))}!={np.prod(data_shape)}'
        template=np.zeros(grid.shape[:-1],dtype=int)
        template[in_mask] = np.arange(np.prod(data_shape))
        out_mask = in_mask[unraveld_lookup[0],unraveld_lookup[1]]
        unraveld_lookup = np.array(np.unravel_index(template[unraveld_lookup[0],unraveld_lookup[1]],data_shape))
        #print(f'max lookup ids = unraveld_lookup.max(axis=1)')
        
    if return_counts:
        counts = np.zeros(new_shape[0])
        shaped_lookup = lookup_ids.reshape(new_shape[:-1])
        for n,x in enumerate(shaped_lookup):
            counts[n] = len(np.unique(x))
        return unraveld_lookup,out_mask,counts
    else:
        return unraveld_lookup,out_mask
    

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
    def __init__(self,pixel_centers,new_grid,data_shape,in_mask = True, interpolation = 'linear',coord_sys='spherical',interpolation_constants=False):
        '''
        pixel_centerns: ndarray of grid points. These are the grid points in which data is given
        new_grid: ndarray of target (new) grid points. These are the grid points ontowhich the data should be interpolated
        in_mask: ndarray of same shape as pixel_centers. False values specify pixels which are always zero.
        data_shape: input data array shape. number of elements has to coincide with the number of True values in in_mask (only used in nearest approximation)
        interpolation: string currently only 'nearest' and 'linear' are supported.
        coord_sys: string. 'cartesian' or 'spherical'. specifies the coordinate type which is used to interprete the grid points in pixel_centers and new_grid
        interpolation_constants: dict|other. posibility to provide precomputed interpolation_constants to skip their computation in instance creation.
        '''
        if interpolation!='linear':
            print('Warning: Switching to linear interpolation, nearest is to buggy currently.')
            interpolation='linear'
        self.out_shape = new_grid.shape[:-1]
        self.in_shape = pixel_centers.shape[:-1]
        self.in_mask = in_mask
        self.data_shape = data_shape
        self.pixel_centers = pixel_centers
        self.new_grid = new_grid
        self.interpolation = interpolation
        self.interpolation_constants = interpolation_constants
        if coord_sys == 'spherical':
            self.pixel_centers = spherical_to_cartesian(pixel_centers)
            self.new_grid = spherical_to_cartesian(new_grid)
        self.pixel_centers_hash,self.new_grid_hash,self.in_mask_hash = self.generate_hashes(self.pixel_centers,self.new_grid,self.in_mask)
        
        try:
            self.interpolation_constants = self.generate_interpolation_constants()
        except AttributeError as e:
            traceback.print_exc()
            log.error('interpolation method {} not known. Available interpolation options are {}'.format(interpolation,'"nearest","linear"')) 
        self.apply = getattr(self,'generate_apply_'+interpolation)()
        
    def generate_hashes(self,pixel_centers,new_grid,in_mask):
        pixel_centers_hash = hash(tuple(pixel_centers.reshape(-1)[:1000]))
        new_grid_hash = hash(tuple(new_grid.reshape(-1)[:1000]))
        if isinstance(in_mask,np.ndarray):
            in_mask_hash = hash(tuple(in_mask.flatten()[:1000]))
        else:
            in_mask_hash = 0
        return pixel_centers_hash,new_grid_hash, in_mask_hash
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
            lookup_array,out_mask,counts = create_lookup_array_2D(self.pixel_centers,self.new_grid,in_mask=self.in_mask,data_shape=self.data_shape,return_counts = True,coord_sys='spherical')
            data = {'lookup_array':lookup_array,'out_mask':out_mask}
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
        if isinstance(self.in_mask,np.ndarray):
            vert_mask = self.in_mask.flatten()[vertices]
            template = np.zeros(self.in_shape,dtype=int)
            template[self.in_mask]=np.arange(self.in_mask.astype(int).sum())
            vertices=template.flatten()[vertices]
            weights[~vert_mask]=0
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
                N = data.shape[0]
                return np.einsum('knj,nj->kn', data.reshape(N,-1)[:,vertices], weights)
            elif ndim == ndim_in:                
                return np.einsum('nj,nj->n', data.flatten()[vertices],weights)
            else:
                raise AssertionError('input data array has to have 3 or 2 dimensions.')
        return apply
    
    def generate_apply_nearest(self):
        lookup = self.interpolation_constants['lookup_array'].astype(int)
        out_zero_mask = ~self.interpolation_constants['out_mask']
        in_mask = self.in_mask
        data_shape = self.data_shape
        if isinstance(in_mask,np.ndarray):
            def apply(data):
                #print(f'data shape = {data.shape} lookup shape = {lookup.shape}')
                if data.ndim == 1+len(lookup):
                    out = data[:,lookup[0],lookup[1]]
                    out[:,out_zero_mask]=0
                    return out
                else:
                    out = data[tuple(lookup)]
                    out[out_zero_mask]=0
                    return out
        else:
            def apply(data):
                if data.ndim == 1+len(lookup):
                    return data[:,lookup[0],lookup[1]]
                else:
                    return data[tuple(lookup)]
        return apply

class AgipdRegridderSimple:
    # Regridder for AGIPD data in the 16,512,128 format.
    # Uses a SimpleRegridder2D object for each of the 16 modules separatelly.    
    version = 0.4 
    @staticmethod
    def get_interpolation_constants_name(geometry,new_grid_shape,interpolation):

        pixel_hash = hash(tuple(geometry['q_framed_pixel_centers'].reshape(-1)[:2000]))
        #log.info('new_grid shape = {}'.format(new_grid_shape))
        new_grid_hash = hash(new_grid_shape + (AgipdRegridderSimple.version,))
        return '{}_{}_{}'.format(interpolation,pixel_hash,new_grid_hash)
    
    
    def __init__(self,geometry,new_grid_shape, interpolation = 'nearest', interpolation_constants=False,mask_threshold = 1-1e-10):        
        spherical_pixel_centers = geometry['q_framed_pixel_centers']
        self.default_interpolation_constants = {
            'data':{str(m_id):False for m_id in range(len(spherical_pixel_centers))},
            'mask':{str(m_id):False for m_id in range(len(spherical_pixel_centers))},
            'module_masks':np.array([False]*len(spherical_pixel_centers)),
            'hash_name': 'not assigned'
        }
        #self.pixel_centers = geometry['framed_pixel_centers']
        self._mask_threshold = [mask_threshold]
        self.geometry = geometry
        self.in_data_shape = geometry['data_shape']
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
    @property
    def mask_threshold(self):
        return self._mask_threshold[0]
    @mask_threshold.setter
    def mask_threshold(self,value):
        self._mask_threshold[0]=value
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
                #print(f'{rough_module_mask.shape}')
                in_mask = in_masks[mid]
                new_points = cart_new_grid[rough_module_mask]
                #print(f'P{kwargs['local_name']} in_mask shape = {in_mask.shape} nonzeros = {in_mask.astype(int).sum()}')
                regridder = SimpleRegridder2D(centers,new_points,self.in_data_shape[1:],coord_sys='cartesian',in_mask = in_mask, interpolation = 'linear')
                mask = rough_module_mask.copy()                
                mask[mask] = (regridder.apply(np.ones(self.in_data_shape[1:],dtype=float)).flatten())>1e-10
                module_masks[_id] = mask                
            return np.asarray(module_masks)
        new_grid = self.new_grid
        in_masks = self.sensitve_pixel_mask
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
        #temp_module_masks = np.ones((16,)+cart_grid.shape[:-1],dtype=bool)
        #print(f'temp mod mask shape = {temp_module_masks.shape}')
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
                
            r = SimpleRegridder2D(module_pixel_centers,new_points,self.in_data_shape[1:],in_mask=self.sensitve_pixel_mask[m_id],interpolation = self.interpolation,interpolation_constants = constants['data'][str(m_id)])
            if self.interpolation == 'linear':
                rm = SimpleRegridder2D(module_pixel_centers,new_points,self.in_data_shape[1:],in_mask = self.sensitve_pixel_mask[m_id],interpolation = 'linear',interpolation_constants = constants['mask'][str(m_id)])
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
        mask_threshold = self._mask_threshold
        def regrid(data,mask,modules,**args):
            ndim = data.ndim
            #log.info('regrid ndim ={}'.format(ndim))
            #log.info('in mask shape ={}'.format(in_mask.shape))
            if ndim == 1+in_mask.ndim:
                N = len(data)
                data_dtype = data.dtype
                mask_dtype = mask.dtype
                #print(f'agipd regridder out_m dtype = {mask_dtype}')
                #in_d = np.zeros(in_shape+(N,),data_dtype)
                #in_m = np.zeros(in_shape+(N,),mask_dtype)
                out_d = np.zeros((N,)+out_shape,data_dtype)
                out_m = np.zeros((N,)+out_shape,mask_dtype)
                #print(f' data shape = {data.shape} mask shape = {mask.shape}')
                #print(f' out data shape = {out_d.shape} out mask shape = {out_m.shape}')
                for n in np.arange(N):
                    out_d_part = out_d[n]
                    out_m_part = out_m[n]
                    data_part = data[n]
                    mask_part = mask[n]
                    #print(f' data part shape = {data_part.shape} mask part shape = {mask_part.shape}')
                    for m in modules:
                        module_mask = module_masks[m]
                        out_d_part[module_mask]= data_regridders[str(m)].apply(data_part[m])
                        out_m_part[module_mask]= mask_regridders[str(m)].apply(mask_part[m].astype(float))>mask_threshold[0]             
            elif ndim == in_mask.ndim:
                data_dtype = data.dtype
                mask_dtype = mask.dtype
                out_d = np.zeros(out_shape,data_dtype)
                out_m = np.zeros(out_shape,mask_dtype)
                #print(f'agipd regridder out_m dtype = {mask_dtype}')
                #print(f' data shape = {data.shape} mask shape = {mask.shape}')
                for m in modules:                    
                    module_mask = module_masks[m]
                    #log.info(f'mask shape = {mask.shape}')
                    #log.info(f'masks shape = {module_masks.shape}')
                    #print(f' data part shape = {data[m].shape} mask part shape = {mask[m].shape}')
                    out_d[module_mask] = data_regridders[str(m)].apply(data[m])
                    out_m[module_mask] = mask_regridders[str(m)].apply(mask[m].astype(float))>mask_threshold[0]
            else:
                raise AssertionError(f'Too many input dimensions. Input has to have {ndim} or {ndim+1} dimensions.')
            return out_d,out_m
        return regrid
                
