import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import logging
from typing import Protocol,Any,Literal
from dataclasses import dataclass
import inspect
import abc

from xframe.library.pythonLibrary import DictNamespace,xprint
from xframe.library.physicsLibrary import spherical_formfactor
from xframe.library.gridLibrary import NestedArray,GridFactory,ReGrider
from xframe.library.mathLibrary import (PolarIntegrator,
                                        SphericalIntegrator,
                                        distance_from_line_2d,midpoint_rule,
                                        gaussian_fourier_transformed_spherical,
                                        spherical_to_cartesian,
                                        cartesian_to_spherical) 

from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_2d
from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_3d
from . import fxs_invariant_tools as i_tools

from xframe import settings

log=logging.getLogger('root')

REAL_PROJECTION_REGISTRY:dict[str:Any] = {}
MetricMode = Literal["last", "history"]

class Projection(Protocol):
    def __call__(self,x:NDArray,context:dict|None = None ):
        pass
    
def register_real_projection(name:str):
    def decorator(cls):
        if name in REAL_PROJECTION_REGISTRY:
            raise ValueError(f"Projection '{name}' is already registered.")
        REAL_PROJECTION_REGISTRY[name]=cls
        if isinstance(cls,type):
            cls.name = name
        return cls
    return decorator


@dataclass
class ProjectionContext:
    iteration:int = 0
    update_support:bool = False

class MetricProperty(property):
    """A property that is also registered as a metric."""
    pass
def metric(func):
    return MetricProperty(func)

class ProjectionBase(abc.ABC):
    name = 'not assigned'
    def __init__(self,metrics_to_save:dict[str,MetricMode]|None = None):
        self._saved_data: dict[str, Any] = {}
        self._metrics_to_save: dict[str, MetricMode] = dict(metrics_to_save or {})

        unknown = set(self._metrics_to_save) - self._available_metrics
        if unknown:
            raise ValueError(
                f"Unknown metrics for projection '{self.name}': {sorted(unknown)}. "
                f"Available metrics are: {sorted(self._available_metrics)}"
            )
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        metrics = set()
        for base in cls.__mro__[1:]:
            metrics.update(getattr(base, "_available_metrics", set()))

        for name, obj in cls.__dict__.items():
            if isinstance(obj, MetricProperty):
                metrics.add(name)

        cls._available_metrics = metrics

    @abc.abstractmethod
    def __call__(self,density:NDArray,context = None)->NDArray:
        pass    
    def save_configured_metrics(self):
        for metric,mode in self._metrics_to_save.items():
            val = getattr(self,metric)
            self._save_metric(metric,val,mode=mode)
            
    def _save_metric(self, name: str, value: Any, mode:MetricMode = 'last') -> None:
        """
        Save a metric locally inside the projection.
        
        history=False:
            store only the latest value
        
        history=True:
            append to a list
        """
        
        if isinstance(value, np.ndarray):
            value = value.copy()
        
        if mode == "last":
            self._saved_data[name] = value
        elif mode == "history":
            self._saved_data.setdefault(name, []).append(
                value
            )
        else:
            raise ValueError(
                f"Unknown metric mode {mode!r}. "
                "Expected 'last' or 'history'."
            )        
    def export_data(self, history_to_numpy: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self._saved_data.items():
            if history_to_numpy and isinstance(value, list):            
                out[name] = np.asarray(value)
            else:
                out[name] = value
        return out    
    def reset_data(self) -> None:
        self._saved_data.clear()
    
class CompositeRealProjection:
    def __init__(self,projections:dict):
        self.projections = projections
        
    def __call__(self,density:NDArray,context:ProjectionContext|None = None)->NDArray:
        for proj in self.projections.values():
            density=proj(density,context=context)
        return density
    
    def export_data(self, history_to_numpy: bool = True) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        
        for name, proj in self.projections.items():
            if isinstance(proj, ProjectionBase):
                data = proj.export_data(history_to_numpy=history_to_numpy)
                if data:
                    out[name] = data                    
        return out
    
    def reset_data(self) -> None:
        for proj in self.projections.values():
            if isinstance(proj, ProjectionBase):
                proj.reset_data()

@register_real_projection("realness")
def realness_projection(density:NDArray,context=None):
    density[...] = density.real
    return density

@register_real_projection("positivity")
def positivity_projection(density:NDArray,context=None):
    density[density.real<0]=0
    return density

@register_real_projection("centering")
class CenterDensity(ProjectionBase):
    def __init__(self,
                 fourier_transform,
                 activation_distance=0,
                 offset:NDArray|None=None,
                 metrics_to_save:dict[str,MetricMode] = None):
        
        self.shift = fourier_transform.shift

        # offset can help with avoiding twinning maybe
        # since point inversion would also change the support
        # in case the particle is symmetric.
        # But that may be bullshit ... in practice
        self.ft = fourier_transform
        real_grid = self.ft.real_grid
        
        if self.ft.dimensions==2:
            self.integrator = PolarIntegrator(real_grid)
        elif self.ft.dimensions==3:
            self.integrator = SphericalIntegrator(real_grid)
        else:
            raise ValueError("Only 2d and 3d fourier transforms are supported.")
        
        self.cart_real_grid = spherical_to_cartesian(real_grid)
        self.cart_reciprocal_grid = spherical_to_cartesian(self.ft.reciprocal_grid)

        self.activation_distance = activation_distance        
        if offset is None:
            self.offset = np.zeros(self.ft.dimensions,float)
        else:
            self.offset = np.asarray(offset)
        self._center = None
        super().__init__(metrics_to_save)
        
    @metric
    def center(self):
        return self._center
    
    def compute_center(self,density:NDArray)->NDArray:
        abs_density = np.abs(density)
        tot_density = self.integrator(abs_density)
        #print(tot_density)
        if tot_density == 0:
            center =- np.array([0]*self.ft.dimensions)
        else:
            center = np.array(tuple(
                self.integrator(abs_density*self.cart_real_grid[...,i]) for i in range(self.ft.dimensions)
            ))/tot_density
        return center

    def __call__(self,density:NDArray,context:ProjectionContext = None)->NDArray:
        center = self.compute_center(density)
        self._center = center
        if np.linalg.norm(center) > self.activation_distance:
            phases = np.exp(1.j*np.dot(self.cart_reciprocal_grid,(center + self.offset)))
            density = self.ft.inverse_cmplx(self.ft.forward_cmplx(density)*phases)
            center_moved = True
        else:
            center_moved = False
        if (context is not None) and center_moved:
            # tell other projections that the support needs to be updated
            context.update_support = True            
        self.save_configured_metrics()
        return density

@register_real_projection("support_by_volume")
class VolumetricSupportProjection(ProjectionBase):
    def __init__(self,
                 fourier_transform,
                 real_grid:NDArray,
                 max_radius:float = np.inf,
                 volume_limits=(0,np.inf),
                 force_connected = False,
                 gaussian_sigma = None,
                 metrics_to_save = None):
        self.ft = fourier_transform
        self.real_grid = real_grid
        self.cart_grid = spherical_to_cartesian(real_grid)
        self.dim = self.real_grid.shape[-1]
        if self.dim == 2:
            self.integrator = PolarIntegrator(self.real_grid)
        elif self.dim ==3:
            self.integrator = SphericalIntegrator(self.real_grid)
        else:
            raise ValueError(f'Only 2 and 3 Dimensional grids are  supported, given grid dim is {self.real_grid.shape[-1]}.')
        self.vol_elements = np.zeros(self.real_grid.shape[:-1],dtype=float)
        self.vol_elements[...] = self.integrator.get_volume_elements()[...,None]
        self.total_volume = self.integrator.get_total_volume()
        self._volume = 0
        self._contrast = 0
        self._contrast_function = 0
        self._connected_counts = 0
        self._support = np.zeros(self.vol_elements.shape,float)
        self._distance_mask = np.zeros(self.vol_elements.shape,bool)
        self.force_connected = force_connected
        self.max_radius = float(max_radius)
        if volume_limits[0]>volume_limits[1]:
            raise ValueError(f'volume_limits[0] must be <= volume_limits[1], but volume_limits= {volume_limits} was given.')
        self.max_allowed_volume = min(self.total_volume,float(volume_limits[1]))
        self.min_allowed_volume = max(float(volume_limits[0]),np.min(self.vol_elements))
        self.gaussian_sigma = gaussian_sigma
        if self.gaussian_sigma is not None:
            self.gaussian_values = gaussian_fourier_transformed_spherical(self.ft.reciprocal_grid,self.gaussian_sigma)
        super().__init__(metrics_to_save)
    @metric
    def support(self):
        return self._support
    @metric
    def distance_mask(self):
        return self._distance_mask
    @metric
    def contrast(self):
        return self._contrast
    @metric
    def contrast_function(self):
        return self._contrast_function
    @metric
    def volume(self):
        return self._volume
    @metric
    def connected_counts(self):
        return self._connected_counts
    
    def compute_distance_from_barycenter(self,density):
        abs_density = np.abs(density)
        tot_density = self.integrator(abs_density)
        if tot_density == 0:
            center =- np.array([0]*self.dim)
        else:
            center = np.array(tuple(
                self.integrator(abs_density*self.cart_grid[...,i]) for i in range(self.dim)
            ))/tot_density        
        distances = np.linalg.norm(self.cart_grid-center,axis = -1)
        return distances
    
    def compute_contrast_metric(self,data,order):
        max_data = np.max(data)
        if max_data == 0:
            raise ValueError("Maximum absolute density is 0. Stop computation.")
        
        meds = np.zeros(len(order),float)
        maxs = np.zeros(len(order),float)
        
        # fill maxima
        maxs[0] = data[order[-1]]
        for i,o in enumerate(order[::-1][1:]):
            val = data[o]
            maxs[i+1] = max(maxs[i],val)
            
        # fill medians
        meds[0::2] = data[order[:len(meds[0::2])]]
        meds[1::2] = (data[order[:len(meds[1::2])]]+data[order[1:len(meds[1::2])+1]])/2
        
        # compute and return contrast metric
        return (meds-maxs[::-1])/max_data
                    
    def __call__(self,density,context=None):
        smooth_density = density
        if self.gaussian_sigma is not None:
            ft_d = self.ft.forward_cmplx(density)
            ft_d *= self.gaussian_values
            smooth_density = self.ft.inverse_cmplx(ft_d)
            
        abs_density = np.abs(smooth_density)
        #abs_density = gaussian_filter1d(abs_density,self.gaussian_sigma,mode='constant',axis=0)
        flat_d = abs_density.ravel()
        order = np.argsort(flat_d)[::-1]
        
        distances = self.compute_distance_from_barycenter(abs_density)
        self._distance_mask[:] = (distances <=self.max_radius)
        distance_mask = self._distance_mask
        # restrict to voxels within max_radius
        order = order[distance_mask.ravel()[order]]

        if order.size==0:
            raise ValueError("No valid pixels. Support collapsed.")
        
        # find Volume that gives maximum contrast
        self._contrast_function = self.compute_contrast_metric(flat_d,order)
        best_vol_id = np.argmax(self._contrast_function)
        
        ordered_vol = self.vol_elements.ravel()[order]
        c_volume = np.cumsum(ordered_vol)
        new_vol = max(min(c_volume[best_vol_id],self.max_allowed_volume),self.min_allowed_volume)
        volume_id = min(np.searchsorted(c_volume,new_vol,side='left'),len(c_volume)-1)
        self._volume = new_vol
        self._contrast = self.contrast_function[volume_id]

        # define greedy support by picking voxels until volume is reached
        self._support[:]=False
        self._support.ravel()[order[:max(volume_id+1,1)]]=True
        
        if self.force_connected:
            connected_components,_ = ndimage.label(self.support)
            component_names,counts = np.unique(connected_components[connected_components>0],return_counts=True)
            self._connected_counts = np.sort(counts)
            for n in component_names[counts<100]:
                self._support[connected_components==n]=False
            #largest_component_id = np.argmax(counts)
            #self._support[:] = (connected_components == component_names[largest_component_id])
            
        # Do density projection
        density *= self._support
        self.save_configured_metrics()
        return density

@register_real_projection("shrink_wrap_support")
class Support(ProjectionBase):
    def __init__(self,fourier_transform,
                 initial_support_radius = None,
                 sw_sigma=None,
                 sw_threshold=0.3,
                 max_radius = np.inf,
                 metrics_to_save:dict[str,MetricMode]|None=None):
        self.ft = fourier_transform
        if initial_support_radius is None:
            self.initial_support_radius = 0.3*self.ft.rs.max()
        else:
            self.initial_support_radius = initial_support_radius
        self._initial_support = self.ft.real_grid[...,0]<self.initial_support_radius
        self._support = self._initial_support.copy()
        self._distance_mask = self._initial_support.copy()

        self.dim = self.ft.dimensions
        if self.dim == 2:
            self.integrator = PolarIntegrator(self.ft.real_grid)
        elif self.dim ==3:
            self.integrator = SphericalIntegrator(self.ft.real_grid)
        else:
            raise ValueError(f'Only 2 and 3 Dimensional grids are  supported, given grid dim is {self.real_grid.shape[-1]}.')

        self.max_radius = max_radius
        if sw_sigma is None:
            sw_sigma = 2*np.pi/self.ft.qs.max() # resolution limit
        self._sw_sigma = max(sw_sigma,0.)
        self._sw_threshold = min(max(sw_threshold,0.),1.)
        self.gaussian_values = gaussian_fourier_transformed_spherical(self.ft.reciprocal_grid,self._sw_sigma)
        self.cart_grid = spherical_to_cartesian(self.ft.real_grid)
        super().__init__(metrics_to_save)
        
    @property
    def sw_sigma(self):
        return self._sw_sigma
    @sw_sigma.setter
    def sw_sigma(self,value):
        if value>=0:
            self._sw_sigma = value
        else:
            self._sw_sigma=0
            log.warning(f'Gaussian sigma has to be grater than 0 but given value is {value}. Projecting threshold to {self._sw_sigma}.')
        self.gaussian_values[:] = gaussian_fourier_transformed_spherical(self.ft.reciprocal_grid,self._sw_sigma)
        
    @property
    def sw_threshold(self):
        return self._sw_threshold
    @sw_threshold.setter
    def sw_threshold(self,value):
        if value<=0:
            self._sw_threshold=0
            log.warning(f'Shrink-wrap threshold has to lie in [0,1] but given value is {value}. Setting threshold to {self._sw_threshold}.')
        elif value>=1:
            self._sw_threshold=1
            log.warning(f'Shrink-wrap threshold has to lie in [0,1] but given value is {value}. Projecting threshold to {self._sw_threshold}.')
        else:
            self._sw_threshold=value

    @metric
    def support(self):
        return self._support
    @metric
    def distance_mask(self):
        return self._distance_mask
    @metric
    def initial_support(self):
        return self._initial_support

    def compute_distance_from_barycenter(self,density):
        abs_density = np.abs(density)
        tot_density = self.integrator(abs_density)
        if tot_density == 0:
            center =- np.array([0]*self.dim)
        else:
            center = np.array(tuple(
                self.integrator(abs_density*self.cart_grid[...,i]) for i in range(self.dim)
            ))/tot_density        
        distances = np.linalg.norm(self.cart_grid-center,axis = -1)
        return distances
    
    def shrink_wrap(self,density):
        # Apply gaussian bluring
        ft_d = self.ft.forward_cmplx(density)
        ft_d *= self.gaussian_values
        convolved_d = np.abs(self.ft.inverse_cmplx(ft_d))
        self._distance_mask = self.compute_distance_from_barycenter(convolved_d)<=self.max_radius
        
        # Define new support
        max_value= convolved_d.max()
        min_value= convolved_d.min()
        diff = max_value-min_value
        self._support = (convolved_d >= min_value + self._sw_threshold*diff) & self._distance_mask
    
    def __call__(self,density:NDArray,context = None)->NDArray:
        if context is not None:
            if context.update_support:
                self.shrink_wrap(density)
        density*=self._support
        self.save_configured_metrics()
        return density

def real_projection_factory(
    name: str,
    data: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
):
    obj = REAL_PROJECTION_REGISTRY[name]

    # plain function projection -> return as is
    if not inspect.isclass(obj):
        return obj

    data = data or {}
    options = options or {}

    sig = inspect.signature(obj.__init__)
    params = list(sig.parameters.values())[1:]   # skip self

    pos_args = []
    kw_args = {}

    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"{obj.__name__}.__init__ uses *args, not supported")

        elif p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if p.name in data:
                pos_args.append(data[p.name])
            elif p.name in options:
                kw_args[p.name] = options[p.name]
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing positional argument '{p.name}'")

        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            if p.name in options:
                kw_args[p.name] = options[p.name]

        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            kw_args.update({k: v for k, v in options.items() if k not in kw_args})
        
    return obj(*pos_args, **kw_args)

def real_projection_factory(
    name: str,
    data: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
):

    try:
        obj = REAL_PROJECTION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown real projection {name!r}") from exc
    
    # plain function projection -> return as is
    if not inspect.isclass(obj):
        return obj

    kwargs = {**(options or {}), **(data or {})}
    sig = inspect.signature(obj)
    bound = sig.bind(**{
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    })
    return obj(*bound.args,**bound.kwargs)

    
### FXS Projections
### 
class ReciprocalProjection:
    def load_data(self,data):
        opt = settings.project
        self.dimensions = data['dimensions']
        self.xray_wavelength = data['xray_wavelength']
        self.average_intensity = data['average_intensity']
        #log.info('aint type ={}'.format(type(self.average_intensity)))
        self.data_radial_points = data['data_radial_points'][:]
        self.data_angular_points = data['data_angular_points'][:]
        self.data_max_q = np.max(self.data_radial_points)
        self.data_min_q = np.min(self.data_radial_points)
        #self.pi_in_q = data.get('pi_in_q',False)
        #self.pi_in_q = data['pi_in_q']
        #log.info(f'data q_min = {self.data_min_q} max_q ={self.data_max_q}')
        
        self.data_max_order=data['max_order']
        self.data_projection_matrices = data['data_projection_matrices']
        self.data_low_resolution_intensity_coefficients = data.get('data_low_resolution_intensity_coefficients',self.data_projection_matrices)
        #log.info(f' pr matrix type {type(self.data_projection_matrices)}')
        self.data_projection_matrices_q_id_limits = data.get('data_projection_matrices_q_id_limits',False)
                
    def __init__(self,grid,data,max_order):
        #xprint(f"max_order = {max_order}")
        self.load_data(data)
        opt = settings.project.projections.reciprocal
        self.opt = opt
        if self.dimensions==2:
            self.integrated_intensity = midpoint_rule(self.average_intensity.data * self.data_radial_points , self.data_radial_points,axis = 0)*2*np.sqrt(np.pi)
        else:
            
            self.integrated_intensity = midpoint_rule(self.average_intensity.data * self.data_radial_points**2 , self.data_radial_points,axis = 0)*2*np.sqrt(np.pi)
            if opt.use_real_spherical_harmonics:
                temp = np.empty(len(self.data_projection_matrices),object)
                for tid,p in enumerate(self.data_projection_matrices):
                    temp[tid]=p.real
                #self.data_projection_matrices=np.array(tuple(d.real for d in self.data_projection_matrices),dtype=object)
                self.data_projection_matrices = temp
                #xprint(self.data_projection_matrices[0].dtype)
            else:
                for p in self.data_projection_matrices:
                    p.imag =0
                #self.data_projection_matrices=np.array(tuple(d.real for d in self.data_projection_matrices),dtype=object)
            #xprint(f'data prolection_matrices = {[d.dtype for d in self.data_projection_matrices]}')
        self.grid=grid        
        self.radial_points=grid.__getitem__((slice(None),)+(0,)*self.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        self.max_q = np.max(self.radial_points)
        #self.input_is_complex=opt.input_is_halfcomplex
        
        self.positive_orders = np.arange(max_order+1)        
        
        self.projection_orders = self.read_projection_orders(opt.get('used_orders',opt.used_order_ids),max_order)
        #log.info('projection_orders = {}'.format(self.projection_orders))
        self.projection_order_ids_local = np.arange(len(self.projection_orders))
        self.projection_order_ids_global = np.isin(self.positive_orders,self.projection_orders).nonzero()[0]
                    
        self.used_order_ids = opt.used_order_ids
        self.used_orders = {order:id for (order,id) in zip(self.positive_orders,self.used_order_ids)}
        #xprint(f"orders = {self.used_orders}")
        self.use_SO_freedom=opt.SO_freedom.use

        pm,low_res = self._regrid_data()        
        #log.info(f'len regridded proj mat = {len(pm)}')
        
        self.projection_matrices = pm
        nq = self.grid.shape[0]
        if self.dimensions ==3:            
            self.full_projection_matrices = [np.zeros((nq,min(nq,2*o+1)),dtype=float) for o in range(max_order+1)]
            self.use_real_spherical_harmonics = opt.use_real_spherical_harmonics
        elif self.dimensions == 2:
            self.full_projection_matrices = np.zeros((max_order+1,nq),dtype=complex)
        for oid,pm in zip(self.used_order_ids,self.projection_matrices):
            self.full_projection_matrices[oid]=pm
            
        self.low_resolution_intensity_coefficients = low_res
        #log.info('projection_matrices after regrid = {}'.format(tuple(d.shape for d in self.projection_matrices)))
        self.projection_matrices=self.modify_projection_matrices(opt)
        #log.info('projection_matrices after modify = {}'.format(tuple(d.shape for d in self.projection_matrices)))
        self.assert_projection_matrices_have_right_shape()        
        self.radial_mask=self.generate_radial_mask(opt.q_mask)        
        self._fixed_intensity = [np.ones(grid.shape[:-1],dtype=complex)]        
        mtip_projection_base=self.generate_coeff_projection_base()        
        self.mtip_projection_base  = mtip_projection_base        
        self.mtip_projection = self.generate_coeff_projection(mtip_projection_base)            
        self.approximate_unknowns=self.generate_approximate_unknowns()
            
        
        self.project_to_modified_intensity = self.generate_project_to_modified_intensity()
        self.project_to_fixed_intensity = self.generate_project_to_fixed_intensity()
        self.deg2_invariants = self.calc_deg2_invariants()

        self.remaining_SO_projection=False
        if self.use_SO_freedom:
            radial_high_pass=opt.SO_freedom.get('radial_high_pass',0.2)
            self.remaining_SO_projection = self.generate_remaining_SO_projection(radial_high_pass=radial_high_pass)
    

    @property
    def fixed_intensity(self):
        return self._fixed_intinsity[0]
    @fixed_intensity.setter
    def fixed_intensity(self,value):
        self._fixed_intensity[0]=value.real
    def read_projection_orders(self,projection_orders,max_harmonic_order):
        try:
            assert projection_orders.max()<=self.data_max_order, 'Max available harmonic order from dataset is {} but the maximal projection order is {}. Restricting ordes to the ones present in the dataset.'.format(self.data_max_order,projection_orders.max())
        except AssertionError as e:
            log.warning(e)
            projection_orders = projection_orders[projection_orders<=self.data_max_order]
        try:
            assert projection_orders.max()<=max_harmonic_order, 'The maximal harmonic oder given by the input harmonic analysis {} is smaller than the specified maximal projection order {}.Limiting projection orders and continue.'.format(max_harmonic_order,projection_orders.max())
        except AssertionError as e :
            log.warning(e)
            projection_orders = projection_orders[projection_orders<=max_harmonic_order]
        return projection_orders
    def assert_projection_matrices_have_right_shape(self):
        try:
            n_radial_points_grid = len(self.radial_points)
            n_radial_points_projection_matrix = self.projection_matrices[0].shape[0]
            assert n_radial_points_grid == n_radial_points_projection_matrix,'Mismatch between th number of radial sampling points in reconstruction grid {}  and projection data {}. Abort reconstruction !'.format(n_radial_points_grid,n_radial_points_projection_matrix)
        except AssertionError as e:
            log.error(e)
            raise e
    def extract_used_options(self):
        fourier_opt = settings.project.fourier_transform
        rp_opt = settings.project.projections.reciprocal
        opt = DictNamespace(
            pos_orders = fourier_opt.pos_orders,
            **rp_opt
        )
        return opt
    def generate_radial_mask(self,mask_opt):
        '''
        Constructs the radial mask (mask of momentum transfair values). Masked values are False and will not be used 
        in the reciprocal coefficient projection defined in self.generate_coeff_projection.
        '''
        radial_points = self.radial_points
        mask = np.full((len(self.positive_orders),len(radial_points)),False)        
        if ~isinstance(self.data_min_q,bool):
            data_mask =  mask | (radial_points>=self.data_min_q) & (radial_points<=self.data_max_q)
        
        if isinstance(mask_opt,(dict,DictNamespace)):
            mtype=mask_opt['type']
            #log.info('mask type = {}'.format(mtype))
            if mtype=='none':
                mask = True
            elif mtype == 'from_projection_matrices':
                data_limit_ids = self.data_projection_matrices_q_id_limits['I1I1']
                for mask_part,lim in zip(mask,data_limit_ids):
                    min_q = self.data_radial_points[lim[0]]
                    max_q = self.data_radial_points[lim[1]-1]
                    mask_part[:] = (radial_points>min_q) & (radial_points<max_q)
            elif mtype == 'manual':
                manual_opt = mask_opt['manual']
                manual_type = manual_opt['type']
                #log.info('manual type = {}'.format(manual_type))
                if manual_type == 'region':
                    region=manual_opt['region']
                    radial_points=self.radial_points
                    #log.info(f'region = {region}')
                    if (region[0] == False) and (region[1] != False):
                        #log.info(f'radial_points = {radial_points}')
                        mask[:]=(radial_points<region[1])[None,:]
                    elif (region[0] != False) and (region[1] == False):                    
                        mask[:]=(radial_points>=region[0])[None,:]
                        log.info('radial mask non zero count {} of {}'.format(np.sum(mask),np.prod(mask.shape)))
                    elif (region[0] != False) and (region[1] != False):
                        mask[:]=((radial_points>=region[0])  & (radial_points<region[1]))[None,:]
                    else:
                        mask[:]=True
                        log.info('nothing masked  {}'.format(mask.all()))
                elif manual_type == 'order_dependent_line':
                    points = manual_opt['order_dependent_line']
                    log.info('points = {}'.format(points))
                    orders = self.positive_orders
                    data_grid = GridFactory.construct_grid('uniform',[orders,radial_points])                 
                    mask = (-1*distance_from_line_2d(np.array(points),data_grid[:]))>=0
        else:
            log.warning('Could not parse projections.reciprocal.q_mask option. Proceeding without custom q_mask')
            mask = True
                
        mask = mask & data_mask
        return mask
    def calc_deg2_invariants(self):
        if self.dimensions == 2:
            invariants = harmonic_coeff_to_deg2_invariants_2d(self.projection_matrices.T)
        elif self.dimensions == 3:
            invariants = harmonic_coeff_to_deg2_invariants_3d(self.projection_matrices)
        #log.info('rp deg 2 invariant shape = {}'.format(invariants.shape))
        return invariants
    def _regrid_data(self):
        dim = self.dimensions
        order_ids=list(self.used_orders.values())
        have_same_shapes = self.radial_points.shape == self.data_radial_points.shape
        projection_matrices = self.data_projection_matrices[order_ids]
        needs_regridding = True
        interpolation_type = self.opt.regrid.interpolation
        if have_same_shapes:
            if (self.data_radial_points == self.radial_points).all():
                needs_regridding = False
        #log.info('needs regridding = {}'.format(needs_regridding))
        #log.info('initial projection matrix shape = {}'.format(projection_matrices[-1].shape))   
        if needs_regridding:
            r_pt=NestedArray(self.radial_points[:,None],1)
            #log.info('r_pt = {}'.format(r_pt[:10]))
            data_r_pt=NestedArray(self.data_radial_points[:,None],1)
            #log.info('n new points={} n old points ={}'.format(len(r_pt[:]),len(data_r_pt[:])))
            low_res = False
            if dim == 2:                
                self.average_intensity.regrid(r_pt,options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                #log.info('proj data shape = {}'.format(self.data_projection_matrices.shape))
                #log.info(f'order ids = {order_ids}')
                projection_matrices = ReGrider.regrid(np.array(self.data_projection_matrices)[order_ids,...],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                #log.info("data_r_min = {} grid r min = {}".format(data_r_pt.array.min(),r_pt.array.min()))
                #log.info('regrided projection matrices complete shape = {}'.format(projection_matrices.shape))
                #log.info('regrided projection matrices shape = {}'.format(projection_matrices.shape))
                #self.average_intensity.regrid(r_pt)
                #projection_matrices = ReGrider.regrid(self.data_projection_matrices[...,order_ids],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type})
            elif dim == 3:
                self.average_intensity.regrid(r_pt,options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                data_projection_matrices=self.data_projection_matrices
                #xprint(f'data proj nans = {[np.isnan(p).any() for p in data_projection_matrices]}')
                projection_matrices = tuple(ReGrider.regrid(data_projection_matrices[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type}) for o_id in order_ids)
                #xprint(f'proj nans = {[np.isnan(p).any() for p in projection_matrices]}')
                if isinstance(self.data_low_resolution_intensity_coefficients,np.ndarray):
                    data_low_res = self.data_low_resolution_intensity_coefficients
                    low_res = tuple(ReGrider.regrid(data_low_res[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type}) for o_id in np.arange(len(data_low_res)))
                #xprint(f'low res contains nans = {[np.isnan(p).any() for p in low_res]}')
                #xprint(f'data proj dtype = {[d.dtype for d in projection_matrices]}')
                    
        #log.info('regrided projection matrices shape = {}'.format(projection_matrices[-1].shape))
        return projection_matrices,low_res
    

    def modify_projection_matrices(self,opt):
        dim = self.dimensions
        use_averaged_intensity=opt.get('use_averaged_intensity',False)
        use_odd_orders_to_0=opt.get('odd_orders_to_0',False)
        
        average_intensity=self.average_intensity.data.astype(complex)
        used_orders=self.used_orders
        odd_order_mask=np.array(tuple(used_orders))%2==1
        if dim == 2:
            proj_matrices=self.projection_matrices.copy()
        elif dim == 3:
            proj_matrices=[matrix.copy() for matrix in self.projection_matrices]
        #log.info(f'use average intensity = {use_averaged_intensity}')
        
        if use_odd_orders_to_0:
            if dim == 2:        
                proj_matrices[odd_order_mask,:]=0
            elif dim == 3:
                for odd_order in np.array(tuple(used_orders))[odd_order_mask]:
                    proj_matrices[used_orders[odd_order]][:]=0

        if use_averaged_intensity:
            zero_id = used_orders[0]            
            if dim == 2:
                #log.info('proj dtype = {} average dtype = {}'.format(proj_matrices.dtype,average_intensity.dtype))
                proj_matrices[zero_id] = average_intensity
                #proj_matrices[:,used_orders[0]]=average_intensity
            elif dim == 3:
                # assumes schmidt seminormalized definition of spherical harmonics -> factor of 2*np.sqrt(pi) 
                proj_matrices[zero_id]= average_intensity[:,None].real*2*np.sqrt(np.pi)
                
        # internally orthonarmalized spherical harmonics are used but data is ussually supplied for schmidt seminormalized spherical harmonics
        if dim == 3:
            for pm in proj_matrices:
                pm[:]*=2
        #xprint(f'proj nans after modification = {[np.isnan(p).any() for p in proj_matrices]}')
        return proj_matrices

    def generate_approximate_unknowns(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        radial_mask = self.radial_mask
        if dim == 2:
            proj_matrices = self.projection_matrices.T
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            unknowns = np.zeros(len(order_ids),dtype = complex)
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info(proj_matrices.shape)
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(proj_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                non_zero_mask = scalar_prod_Im_vm !=0
                unknowns[non_zero_mask] = scalar_prod_Im_vm[non_zero_mask]/abs(scalar_prod_Im_vm[non_zero_mask])
                #        log.info(scalar_prod_Im_vm)
                unknowns[~non_zero_mask]=1
                #tmp = np.concatenate((unknowns,unknowns[:0:-1].conj()))
                #unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns#np.zeros_like(unknowns)
            
            if self.use_SO_freedom:
                SO_order_id =self.get_SO_application_order()
                def function(intensity_harmonic_coefficients):
                    unknowns = approximate_unknowns(intensity_harmonic_coefficients)
                    unknowns[SO_order_id]=1
                    return unknowns
            else:
                function = approximate_unknowns
            
        if dim == 3:
            from scipy.ndimage import gaussian_filter1d
            D=np.diag(radial_points) #diagonal matrix of radial points
            #L = proj_matrices[0]
            #L[L==0]=1
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            determinant = np.linalg.det
            n_orders = len(PDs)
            if self.use_real_spherical_harmonics:
                unknowns = tuple(np.eye(2*o+1,dtype = float) for PD,o in zip(PDs,used_orders))
            else:
                unknowns = tuple(np.eye(2*o+1,dtype = complex) for PD,o in zip(PDs,used_orders))
            unknowns = tuple(u[:len(PD),:] for PD,u in zip(PDs,unknowns))
            PDs=tuple(pd[:,mask] for pd,mask in zip(PDs,radial_mask))
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                for unknown,PD,oid,l,qmask in zip(unknowns,PDs,order_ids,used_orders,radial_mask):
                    I = intensity_harmonic_coefficients.lm[l]
                    #xprint(f'PD contains Nans : {np.isnan(PD).any()}')
                    #xprint(f'unknown dtype = {unknown.dtype}, PD dtype = {PD.dtype}, coeff dtype = {I[qmask].dtype}')
                    matmul(*svd(PD @ (I[qmask]),full_matrices=False)[::2],out = unknown)  # PD @ Intensity is  B^\dagger A in a Procrustres Problem min|A-BR|
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                return unknowns
            if self.use_SO_freedom:
                radial_high_pass = self.opt.SO_freedom.radial_high_pass
                ranked_ids,ranked_orders,_ = i_tools.rank_projection_matrices(self.dimensions,proj_matrices,self.positive_orders,radial_points,radial_high_pass = radial_high_pass)
                SO_order_id = ranked_ids[0]
                SO_order = self.positive_orders[SO_order_id]
                max_order = self.positive_orders.max()
                ms_per_order = []
                for order in np.arange(max_order+1):
                    ms_per_order.append(np.concatenate((np.arange(order+1),np.arange(-order,0))))
                
                def function(intensity_harmonic_coefficients):
                    for ms,unknown,PD,l,qmask in zip(unknowns,PDs,used_orders,radial_mask):
                        I = intensity_harmonic_coefficients.lm[l]
                        u,s,vh = svd(PD @ I[qmask],full_matrices=False)
                        matmul(u,vh,out = unknown)  # PD @ Intensity is  B^\dagger A in a Procrustres Problem min|A-BR|
                    
                    u_SO = unknowns[SO_order_id]
                    u_SO[4,2] = u_SO[4,2].real                    #u_SO[4,2] = u_SO[4,2].real
                    #log.info(f'unknown SO_order imag part = {unknowns[SO_order_id][4,2].imag}')
                    return unknowns
            else:                
                function = approximate_unknowns
            
        return function
    def generate_coeff_projection_base(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id  = used_orders.get(0,False)
        zero_pos = np.argmax(np.array(tuple(used_orders.keys())) == 0)
        
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        #average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        #mask=self.mask_2d
        radial_mask = self.radial_mask
        #projection_matrices[zero_id]= np.abs(projection_matrices[zero_id])
        if dim == 2:
            where=np.where
            order_array=np.array(list(used_orders.values()))
            #order_array=np.concatenate((order_array,order_array[:0:-1]))
            new_intensity_coefficients=np.zeros((self.grid.shape[0],(self.grid.shape[1]+1)//2),dtype=complex)
            mult=np.multiply
            mask = np.zeros((self.grid.shape[0],(self.grid.shape[1]+1)//2),dtype=bool)
            for o_id in order_array:
                mask[:,o_id]=radial_mask[o_id]
            radial_mask2 = radial_mask[order_array].T
            #log.info('2d mask .shape = {}, 2d_radial_mask shape = {}, grid shape = {} len orders = {} \n\n\n'.format(mask.shape,radial_mask2.shape,self.grid.shape,order_array.shape))
            #projection_matrices=np.concatenate((projection_matrices,projection_matrices[:0:-1].conj()),axis=0)
            projection_matrices=projection_matrices.T
            
            qs = self.grid[:,0,0]
            sphere = spherical_formfactor(qs,radius = 10)
            if not isinstance(zero_id,bool):
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    new_intensity_coefficients[:]=intensity_harmonic_coefficients
                    new_intensity_coefficients[mask] = (projection_matrices*unknowns[None,:])[radial_mask2]
                    new_intensity_coefficients[radial_mask[zero_id],zero_id] = projection_matrices[radial_mask[zero_id],zero_pos]
                    return new_intensity_coefficients                
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    new_intensity_coefficients[:]=intensity_harmonic_coefficients
                    new_intensity_coefficients[mask] = (projection_matrices*unknowns[None,:])[radial_mask2]
                    return new_intensity_coefficients
        elif dim == 3:
            copy=np.array #array is faster than copy
            if not isinstance(zero_id,bool):
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    #projected_intensity_coefficients=[copy(coeff) for coeff in intensity_harmonic_coefficients]
                    projected_intensity_coefficients=intensity_harmonic_coefficients.copy()
                    for l,o_id in self.used_orders.items():
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients.lm[l][radial_mask[o_id]] = tmp_coeff[radial_mask[o_id]]
                    projected_intensity_coefficients.lm[0][radial_mask[zero_id]] = projection_matrices[zero_id][radial_mask[zero_id]]
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[copy(coeff) for coeff in intensity_harmonic_coefficients]
                    for o_id in self.used_orders.values():
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        #projected_intensity_coefficients[o_id][radial_mask[o_id],...] = tmp_coeff[radial_mask[o_id],...]
                        projected_intensity_coefficients.lm[l][radial_mask[o_id]] = tmp_coeff[radial_mask[o_id]]
                    return projected_intensity_coefficients
        return mtip_projection
    def generate_coeff_projection(self,coeff_projection):
        dim = self.dimensions
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        radial_mask = self.radial_mask
        number_of_particles = [1]
        if dim == 2:
            def fixed_projection(intensity_harmonic_coefficients,unknowns):
                projected_intensity_coefficients = coeff_projection(intensity_harmonic_coefficients,unknowns)
                #projected_intensity_coefficients[zero_id][radial_mask[zero_id]]/=np.sqrt(number_of_particles[0])
                #log.info(f'number of particles scaling factor = {1/np.sqrt(number_of_particles[0])}')
                #log.info(f'out_array = {projected_intensity_coefficients[:,zero_id].shape} in array = {self.projection_matrices[zero_id].shape}')
                projected_intensity_coefficients[:,zero_id]/=np.sqrt(number_of_particles[0])
                return projected_intensity_coefficients
        elif dim == 3:
            def fixed_projection(intensity_harmonic_coefficients,unknowns):
                projected_intensity_coefficients = coeff_projection(intensity_harmonic_coefficients,unknowns)
                #projected_intensity_coefficients[zero_id][radial_mask[zero_id]]/=np.sqrt(number_of_particles[0])
                #log.info(f'number of particles scaling factor = {1/np.sqrt(number_of_particles[0])}')
                projected_intensity_coefficients.lm[0][:]/=np.sqrt(number_of_particles[0])
                
                #for l,o_id in self.used_orders.items():
                #    if not radial_mask[o_id].all():
                #        qmask = radial_mask[o_id]
                #        first_unmasked = np.argmin(~qmask)+1
                #        #xprint(first_unmasked)
                #        Il = projected_intensity_coefficients.lm[l]
                #        signs_ref = np.sign((Il@Il[first_unmasked].conj()).real)
                #        signs = np.sign((Il@Il[~qmask].T.conj()).real)
                #        metrics = np.sum(np.abs(signs-signs_ref[:,None]),axis =0)>np.sum(np.abs(-signs-signs_ref[:,None]),axis =0)
                #        tmp = Il[~qmask]
                #        #xprint([metrics.shape,tmp.shape])
                #        ##assert metrics.shape == tmp.shape[0], 'fuu'
                #        tmp[metrics]*=-1
                #        Il[~qmask]=tmp
                #        #Il[l][(~qmask)&metrics]*=-1
                #        #xprint(metrics.any())
                return projected_intensity_coefficients
        return fixed_projection
    def generate_project_to_modified_intensity(self):
        if settings.general.cache_aware:
            L2_cache = settings.general.L2_cache
            func = self._generate_project_to_modified_intensity_cache_aware(L2_cache)
        else:
            func = self._generate_project_to_modified_intensity_default()
        return func
    
    def generate_project_to_fixed_intensity(self):
        if settings.general.cache_aware:
            L2_cache = settings.general.L2_cache
            func = self._generate_project_to_modified_intensity_cache_aware(L2_cache,use_fixed_intensity=True)
        else:
            func = self._generate_project_to_modified_intensity_default(use_fixed_intensity=True)
        return func
    def _generate_project_to_modified_intensity_default(self,use_fixed_intensity=False):
        zeros = np.zeros
        density_shape = self.grid[:].shape[:-1]
        sqrt = np.sqrt
        mult = np.multiply
        nabs = np.abs
        new_reciprocal_density = np.zeros(density_shape,dtype =complex)
        intensity_multipliers = np.zeros(new_reciprocal_density.shape , dtype= complex)
        temp = np.zeros(new_reciprocal_density.shape , dtype= float)
        if not use_fixed_intensity:
            def project_to_modified_intensity(reciprocal_density,square,square_from_harmonics,new_intensity):
                # update by difference to take into account part of square that is not represented by its harmonic coeff (due to harmonic cutoff) 
                new_intensity = square.real + (new_intensity.real - square_from_harmonics.real)
                new_neg_mask = new_intensity<0                
                new_intensity[new_neg_mask] = 0 #square[new_neg_mask]    
                non_zero_mask = (square!=0) #& (np.abs(new_intensity)>0)
                temp[non_zero_mask] = new_intensity[non_zero_mask]/square[non_zero_mask].real
                np.sqrt(temp,out = intensity_multipliers,dtype = complex)
                mult(reciprocal_density ,intensity_multipliers,out = new_reciprocal_density)
                new_reciprocal_density[~non_zero_mask] = np.sqrt(new_intensity[~non_zero_mask],dtype = complex)
                
                #log.info('nans = {} infs = {}'.format(np.isnan(new_reciprocal_density),np.isinf(new_reciprocal_density)))
                return new_reciprocal_density
        else:
            new_intensity2 = self._fixed_intensity[0]
            new_intensity2 = new_intensity2.real
            new_neg_mask = new_intensity2<0
            new_intensity2[new_neg_mask]=0
            def project_to_modified_intensity(reciprocal_density,square):
                #log.info(new_intensity2[0][:10,0,0])
                non_zero_mask = (square>0) 
                #log.info('square dtype = {}'.format(square.dtype))
                temp.real[non_zero_mask] = new_intensity2[non_zero_mask]/square[non_zero_mask].real
                np.sqrt(temp,out = intensity_multipliers,dtype=complex)
                mult(reciprocal_density ,intensity_multipliers,out = new_reciprocal_density)
                new_reciprocal_density[~non_zero_mask] = np.sqrt(new_intensity2[~non_zero_mask])
                #new_reciprocal_density[new_neg_mask] = reciprocal_density[new_neg_mask]
                #log.info("old intesity sum = {} new intensity sum = {}".format(np.sum(square),np.sum(new_intensity)))
                #log.info('nans = {} infs = {}'.format(np.isnan(new_reciprocal_density),np.isinf(new_reciprocal_density)))
                #new_reciprocal_density = reciprocal_density * sqrt(new_intensity.real/(reciprocal_density*reciprocal_density.conj()).real)            
                return new_reciprocal_density
        
        return project_to_modified_intensity
    def _generate_project_to_modified_intensity_cache_aware(self,L2_cache,use_fixed_intensity=False):
        data_shape = self.grid[:].shape[:-1]
        #split_id,step = get_L2_cache_split_parameters(data_shape,np.dtype(complex),L2_cache)
        return self._generate_project_to_modified_intensity_default(use_fixed_intensity = use_fixed_intensity)

    #scaled version to donatellies n particle determination

    def rank_projection_matrix_orders_2d(self):
        radial_high_pass=self.opt.SO_freedom.get('radial_high_pass',0.2)
        radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
        radial_high_pass=0
        if not isinstance(radial_high_pass_index,bool):
            radial_high_pass=radial_high_pass_index
        radial_points=self.radial_points[radial_high_pass:]


        orders = self.projection_orders
        #log.info(orders)
        order_ids= self.projection_order_ids_local
        even_order_mask= orders%2 == 0
        non_zero_mask= orders != 0
        order_mask=even_order_mask*non_zero_mask
        relevant_orders=orders[order_mask]
        #relevant_order_ids = order_ids[order_mask]

        projection_vector=self.projection_matrices[order_mask,radial_high_pass_index:].T        
        

        weighted_vect=projection_vector*radial_points[:,None]#/(np.concatenate(([100.0,100.0,6,8],harmonic_orders[4:])))[None,:]
        #log.info('weighted_vect shape =\n {}'.format(weighted_vect.shape))
        metric = np.mean(np.abs(weighted_vect),axis=0)#/orders[order_mask]
        #log.info(metric)
        sorted_indices = np.argsort(metric)[::-1]
        #log.info(sorted_indice)
        SO_order_indices=order_mask.nonzero()[0][sorted_indices]
        SO_orders=orders[SO_order_indices]
        return SO_order_indices,SO_orders,sorted_indices
    def get_SO_application_order(self):
        dim=self.dimensions
        if dim ==2 :
            SO_order_indices,SO_orders,sorted_indices=self.rank_projection_matrix_orders_2d()            
            #log.info('Selected SO_order_ids = {}'.format(SO_order_indices))
        elif dim == 3:
            raise NotImplementedError
        return SO_order_indices[0]
    def generate_apply_SO_freedom_2D(self,opt):
        dim=self.dimensions
        if dim ==2 :
            ordes_dict=self.used_orders
            orders=np.array(tuple(orders_dict))
            order_ids=tuple(orders_dict.values())
            
            radial_high_pass=opt['SO_freedom'].get('radial_high_pass',0.2)
            radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
            radial_points=self.radial_points

            even_order_mask= orders%2 == 0
            if use_averaged_intensity:
                non_zero_mask= orders != 0
            else:
                non_zero_mask=True
            order_mask=even_order_mask*non_zero_mask
            relevant_orders=orders[order_mask]
            max_order=np.max(relevant_orders)
        
            projection_vector=self.projection_matrices[radial_high_pass_index:,order_mask]
            
            #pres_cart.present(np.swapaxes(np.abs(projection_vector),0,1),scale='log')
            #pres_1d.present(np.sum(np.abs(projection_vector),axis=0))
            sorted_projection_indices=self.rank_orders(projection_vector,radial_points[radial_high_pass_index:],relevant_orders)
            #log.info('sorted_projection_indices={}'.format(sorted_projection_indices))
            SO_order_index=order_mask.nonzero()[0][sorted_projection_indices[0]]
            #log.info('nonzero = {}'.format(order_mask.nonzero()[0]))
            log.info('SO order Index={}'.format(SO_order_index))
            #   log.info('first_order_index={}'.format(first_order_index))
            #   log.info('first_order={}'.format(first_order))
            
            def apply_SO_freedom(unknowns):
                #log.info('unknowns shape={}'.format(unknowns.shape))
                unknowns[SO_order_index]=1
                return unknowns
        elif dim == 3:
            raise NotImplementedError
        return apply_SO_freedom
    def generate_remaining_SO_projection(self,radial_high_pass=0.2):
        dim=self.dimensions
        if dim == 2:
            remaining_SO_projection=self.generate_remaining_SO_projection_2D(radial_high_pass=radial_high_pass)
        elif dim == 3:
            def remaining_SO_projection(harmonic_coefficients,fxs_unknowns):            
                raise NotImplementedError()        
        return remaining_SO_projection
    def generate_remaining_SO_projection_2D(self,radial_high_pass=0.2):
        radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
        projection_vectors = self.projection_matrices
        n_angular_points=self.grid.shape[1]
        projection_orders=np.concatenate((np.arange(int(n_angular_points/2)+1),-1*np.arange(int(n_angular_points/2)+n_angular_points%2)[:0:-1]))
        
        pos_orders = self.positive_orders
        positive_harmonic_orders=np.array(tuple(self.used_orders.keys()))
        order_ids= np.array(tuple(self.used_orders.values()))
        even_order_mask=positive_harmonic_orders%2==0
        non_zero_mask=positive_harmonic_orders!=0
        order_mask=even_order_mask*non_zero_mask
        harmonic_orders=positive_harmonic_orders[order_mask]
        harmonic_orders_ids=order_ids[order_mask]
        max_order=np.max(harmonic_orders)
            
        SO_order_indices,SO_orders,sorted_order_indices=self.rank_projection_matrix_orders_2d() 
        #sorted_order_indices=self.rank_projection_matrix_orders_2d(projection_vector,self.radial_points[radial_high_pass_index:])
        
        #log.info('sorted_order_indices={}'.format(SO_order_indices))
        first_order_index=SO_order_indices[0]
        first_order=SO_orders[0]#harmonic_orders[first_order_index]
        #log.info('ranked_orders={}'.format(harmonic_orders[sorted_order_indices]))
    
        remaining_rotations=first_order
        current_order=first_order    
        free_orders_mask=True
        angle_coeffs=()
        angles=()
        order_indices=()
        gcds=()
        while remaining_rotations>2:
            #find non invariant orders
            order_multiples=np.arange(current_order,max_order+1,current_order)
            multiple_indices=np.where(np.isin(harmonic_orders,order_multiples))
            free_orders_mask*=~np.isin(sorted_order_indices,multiple_indices)
            #log.info('remaining_orders={}'.format(harmonic_orders[sorted_order_indices[free_orders_mask]]))
            if free_orders_mask.any()==False:
                break
            else:
                #select next order
                current_order_index=sorted_order_indices[free_orders_mask][0]
                current_order=harmonic_orders[current_order_index]
                #log.info('current order ={}'.format(current_order))
                #calculate remaining rotations
                gcd=np.gcd(remaining_rotations,current_order)
                #log.info('gcd={}'.format(gcd))
                #use rotational freedom on current_order
                n_independent_rotations=remaining_rotations/gcd
                #log.info('n independent rotations={}'.format(n_independent_rotations))
                smallest_angle=2*np.pi/n_independent_rotations
                smallest_angle_coeff=np.argmin((np.arange(1,n_independent_rotations)*current_order/gcd)%n_independent_rotations)+1
    
                order_indices+=(current_order_index,)
                angle_coeffs+=(smallest_angle_coeff,)
                angles+=(smallest_angle,)
                gcds+=(gcd,)
                remaining_rotations=gcd
        
        def apply_SO_freedom(harmonic_coefficients,fxs_unknowns):
            #log.info('fxs unknowns shape ={}'.format(fxs_unknowns.shape))
            phases=(-1.j*np.log(fxs_unknowns[order_mask])).real
            #log.info('complete phases ={}'.format(-1.j*np.log(fxs_unknowns[::2])))
            #log.info('phases ={}'.format(phases))
            rotation_phase=0
            for order_index,angle,angle_coeff,gcd in zip(order_indices,angles,angle_coeffs,gcds):
                rotation_phase-=(phases[order_index]//angle)*angle_coeff*angle/gcd
                #log.info('order={} phase={}'.format(harmonic_orders[order_index],phases[order_index]))
                #log.info('min angle*gcd={}'.format(angle))
                #log.info('rotated phase={}'.format((phases[order_index]+harmonic_orders[order_index]*rotation_phase)%(2*np.pi)))
            
            harmonic_coefficients*=np.exp(1.j*projection_orders*rotation_phase)
            return harmonic_coefficients
        return apply_SO_freedom


    # Number of particle estimation experimental not working jet
    def change_n_particles(self,N):
        '''Setter for n_particles attribute'''
        self.n_particles = N
    def get_number_of_particles(self):
        return 1

    
#positions and point inversion projections
def generate_fix_point_inversion(radial_low_pass = 0.1):
    
    def fix_point_inversion(scattering_amplitude):
        'Routine that corrects point inversion if given scattering amplitudes which only differ by point inversion (i.e. complex conjugation) from each other.'
        phases = scattering_amplitude/np.abs(scattering_amplitude)
        shape = scattering_amplitude.shape
        inversion_indicator = np.sum(phases.imag[:int(shape[0]*radial_low_pass),:shape[1]//2])
        log.info('inversion indicator ={} radial_low_pass = {}'.format(inversion_indicator,radial_low_pass))
        if inversion_indicator<0:
            scattering_amplitude=scattering_amplitude.conjugate()
        return scattering_amplitude
    return fix_point_inversion
def generate_negative_shift_operator(reciprocal_grid,fourier_type):
    dim=reciprocal_grid.n_shape[0]
    pre_exponent=2*np.pi
    pi=np.pi
    if (fourier_type=='Zernike') or (fourier_type=='trapz'):
        pre_exponent=1

    cart_grid = spherical_to_cartesian(reciprocal_grid)
    def negative_shift(reciprocal_density,vector):
        cart_vect = spherical_to_cartesian(vector)
        log.info('cart_vector = {}'.format(cart_vect))
        
        phases = np.exp(-1.j*pre_exponent*(cart_grid*cart_vect).sum(axis=-1))
        log.info('phases shape = {}'.format(phases.shape))
        reciprocal_density*=phases
        return reciprocal_density        
    return negative_shift

def generate_shift_by_operator(grid,opposite_direction=False):
    dim=grid[:].shape[-1]
    pi = np.pi
    if opposite_direction:
        prefactor = -1
    else:
        prefactor = 1
    if dim == 2:
        cart_grid = spherical_to_cartesian(grid)
        def shift_by(reciprocal_density,vector):
            #log.info('vector = {} /n'.format(vector))
            cart_vect = spherical_to_cartesian(vector)
            #log.info('cart shift vect = {}'.format(-cart_vect*prefactor))
            phases = np.exp(-1.j*prefactor*(cart_grid*cart_vect).sum(axis=-1))
            reciprocal_density*=phases
            return reciprocal_density   
    elif dim == 3:
        cart_grid = spherical_to_cartesian(grid)
        def shift_by(reciprocal_density,vector):
            #log.info('vector = {}'.format(vector))
            cart_vect = spherical_to_cartesian(vector)
            #log.info('cart shift vect = {}'.format(-cart_vect*prefactor))
            phases = np.exp(-1.j*prefactor*(cart_grid*cart_vect).sum(axis=-1))
            print(f'reciprocal dens shape = {reciprocal_density.shape} phases shape = {phases.shape}')
            reciprocal_density*=phases
            return reciprocal_density        
    return shift_by


