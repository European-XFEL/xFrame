from xframe.library.math_transforms import (SphericalFourierTransform,
                                            SphericalFourierTransformStruct,
                                            HankelTransformWeights)
import numpy as np
import pysofft
from pysofft import Soft,utils
from numpy.typing import NDArray
from typing import Literal,List,Callable
from xframe.library.mathLibrary import (PolarIntegrator,
                                        SphericalIntegrator,
                                        spherical_to_cartesian,
                                        CumulativeVariance)
from xframe import Multiprocessing
from dataclasses import dataclass,field
import abc

MapDomain = Literal['real','fourier']

class Map(np.ndarray):
    r"""
    Numpy ndarray subclass that simply adds a domain field that identifies
    whether the data is to be interpreted as scalar field in real or momentum space.
    
    Attributes
    ----------
    domain: 
        Bandwidth of the distribution.
    """
    def __new__(cls,map_array:NDArray,domain:MapDomain = 'real',is_support:bool=False):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(map_array).view(cls)
        
        # add the new attribute to the created instance
        obj.domain = domain
        obj.is_support = is_support#
        return obj
        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        
        self.domain = getattr(obj, 'domain', 'real')
        self.is_support = getattr(obj, 'is_support', False)
        
    def normalization_constant(self,method=np.max):
        if self.is_support:
            return 1.0
        else:
            return method(self)
class Aligner:
    def __init__(self,
                 fourier_transform:SphericalFourierTransform,
                 normalization_method=np.max,
                 reference: Map|NDArray |None = None):
        self.ft = fourier_transform
        self.ft_shift = self.ft.shift 
        self.dim = fourier_transform.dimensions
        self.normalization_method = normalization_method
        self.soft = Soft(self.ft.ht.bandwidth,
                                 init_ffts=True,
                                 fftw_flags=pysofft.fftw.flags.fftw_measure,
                                 use_fftw_wisdom=True)
        if self.dim==2:
            self._real_integrator = PolarIntegrator(self.ft.real_grid)
            self._fourier_integrator = PolarIntegrator(self.ft.reciprocal_grid)
        elif self.dim == 3:
            self._real_integrator = SphericalIntegrator(self.ft.real_grid)
            self._fourier_integrator = SphericalIntegrator(self.ft.reciprocal_grid)
        self.real_cart_grid = spherical_to_cartesian(self.ft.real_grid)
        self.fourier_cart_grid = spherical_to_cartesian(self.ft.reciprocal_grid)

        self._reference = None
        self._reference_lm = None
        if reference is not None:
            self.reference = reference
    @property
    def reference(self):
        return self._reference
    @reference.setter
    def reference(self,value):
        self._reference = value
        self._reference_lm = self.ft.harm.forward(value)
    @property
    def reference_lm(self):
        return self._reference_lm
    @reference_lm.setter
    def reference_lm(self,value):
        self._reference_lm = value
        self._reference = self.ft.harm.inverse(value)
    
    def _convert_to_list_of_map(self,data:Map|NDArray|List[Map|NDArray]) -> List[Map]:
        if not isinstance(data,(list,tuple)):
            data = [data]

        out = []
        for d in data:
            if isinstance(d,Map):
                out.append(d)
            else:
                out.append(Map(d,domain='real',is_support=False))
        return out

    def find_center(self,data:Map):
        if data.domain == 'real':
            integrator = self._real_integrator
            grid = self.real_cart_grid
        else:
            integrator = self._fourier_integrator
            grid = self.fourier_cart_grid
            
        abs_data = np.abs(data).real
        tot_int = integrator(abs_data)         
        if tot_int==0:
            tot_int = 1
        cart_center = integrator(np.moveaxis(grid[:],-1,0)*abs_data[None,...])/tot_int
        return cart_center  

    def shift(self,data:Map,vector:NDArray) -> Map:
        if data.domain == "real":
            ft_data = self.ft.forward_cmplx(data.astype(complex))
            new_ft_data = self.ft_shift(ft_data,vector,opposite_direction=True,center_coord_sys='cartesian')
            centered_data = self.ft.inverse_cmplx(new_ft_data)
            if not np.iscomplexobj(data):
                centered_data = centered_data.real
        else:
            centered_data = self.ft_shift(data,vector,opposite_direction=True,center_coord_sys='cartesian')
        return Map(centered_data,domain=data.domain,is_support=data.is_support)
         
    def normalize_and_center(self,
                             data: Map | NDArray | List[Map|NDArray],
                             anchor_id: int = 0) -> List[Map]:
        data = self._convert_to_list_of_map(data)
        center = self.find_center(data[anchor_id])
        data  = tuple( self.shift(d,center) for d in data )
        normalization_constant = data[anchor_id].normalization_constant(method=self.normalization_method)
        data = tuple( d if d.is_support else d/normalization_constant for d in data)
        return  data

    def find_optimal_rotation_lm(self,
                                 data_lm,
                                 reference_lm = None,
                                 domain='real',
                                 radial_limits = None,
                                 return_correlation = False):
        
        if reference_lm is None:
            if self.reference_lm is not None:
                ref_lm = self.reference_lm
            else:
                raise ValueError("No reference provided + no global reference (self.reference) set. Can not perform rotational alignment without a reference.")
        else:
            ref_lm = reference_lm
            
        if ref_lm.complex_data != data_lm.complex_data:
            raise ValueError('Trying to align complex valued data to real valued reference or vice versa is not possible.')

        if domain == 'real':
            radial_points = self.ft.rs
        else:
            radial_points = self.ft.qs

        if data_lm.complex_data:
            so3_correlation = self.soft.cross_correlation_ylm_cmplx_3d(data_lm,
                                                                       ref_lm,
                                                                       radial_sampling_points = radial_points,
                                                                       radial_limits = radial_limits )
        else:
            so3_correlation = self.soft.cross_correlation_ylm_real_3d(data_lm,
                                                                      ref_lm,
                                                                      radial_sampling_points = radial_points,
                                                                      radial_limits = radial_limits )            
        euler_angles,max_corr = self.soft.cross_correlation_to_aligning_rotation(so3_correlation,
                                                                                return_max_corr=True)
        
        if return_correlation:
            return euler_angles,max_corr,so3_correlation
        else:
            return euler_angles,max_corr
            

    def find_optimal_rotation(self,
                              data:Map,
                              reference:Map|None = None,
                              radial_limits = None,
                              return_correlation=False):
        harm = self.ft.harm
        
        # get reference harmonic coefficients
        if reference is None:
            if self.reference is not None:
                ref = self.reference
                ref_lm = self.reference_lm
            else: 
                raise ValueError("No reference probided + no global reference (self.reference) set. Can not perform rotational alignment without a reference.")
        else:
            ref = reference
            ref_lm = harm.forward(ref)

        if data.domain != ref.domain:
            raise ValueError(f'Trying to aligning data from different domains is not possible: data domain = {data.domain} reference domain = {ref.domain}')
        
        if data.domain == 'real':
            radial_points = self.ft.rs
        else:
            radial_points = self.ft.qs
        
            
        # get data harmonic coefficients
        data_lm = harm.forward(data)
        
        if data_lm.complex_data != ref_lm.complex_data:
            raise ValueError('Trying to allign complex valued data to real valued reference or vice versa.')

        if data_lm.complex_data:
            so3_correlation = self.soft.cross_correlation_ylm_cmplx_3d(data_lm,
                                                                       ref_lm,
                                                                       radial_sampling_points = radial_points,
                                                                       radial_limits = radial_limits )
        else:
            so3_correlation = self.soft.cross_correlation_ylm_real_3d(data_lm,
                                                                      ref_lm,
                                                                      radial_sampling_points = radial_points,
                                                                      radial_limits = radial_limits )
            
        euler_angles,max_corr = self.soft.cross_correlation_to_aligning_rotation(so3_correlation,
                                                                                 return_max_corr=True)

        if return_correlation:
            return euler_angles,max_corr,so3_correlation
        else:
            return euler_angles,max_corr

    def _rotate_single_lm(self,dlm,euler_angles):
        if dlm.complex_data:
            coeff = self.ft.harm.get_empty_coeff(pre_shape = (euler_angles.shape[0],)+dlm.shape[:-1],
                                             complex_data = True)
            coeff = np.squeeze(coeff)
            coeff[...] = self.soft.rotate_ylm_cmplx(dlm,euler_angles)
        else:
            coeff = self.ft.harm.get_empty_coeff(pre_shape = (euler_angles.shape[0],)+dlm.shape[:-1],
                                                 complex_data = False, real_harmonics=False)
            coeff = np.squeeze(coeff)
            coeff[...] = self.soft.rotate_ylm_real(dlm,euler_angles)
        return coeff
    def rotate_lm(self,data_lm,euler_angles):
        data_lms = data_lm if isinstance(data_lm, (list, tuple)) else [data_lm]
        euler_angles = euler_angles if euler_angles.ndim==2 else euler_angles[None,:]
        out = []
        for dlm in data_lms:
            out.append(self._rotate_single_lm(dlm,euler_angles))
        return out                
    def rotate(self,data,euler_angles):
        data_list = data if isinstance(data, (list, tuple)) else [data]
        euler_angles = euler_angles if euler_angles.ndim==2 else euler_angles[None,:]
        out = []
        for d in data_list:
            dlm = self.ft.harm.forward(d)
            dlm_rot = self._rotate_single_lm(dlm,euler_angles)
            drot = self.ft.harm.inverse(dlm_rot)
            out.append(Map(drot,domain=d.domain,is_support = d.is_support))
        return out
    def point_invert(self,data:List[Map]|Map):
        harm = self.ft.harm
        data_list = data if isinstance(data, (list, tuple)) else [data]
        out = []
        for d in data_list:
            dlm_p = harm.forward(d).point_inverse()
            d_p = harm.inverse(dlm_p)
            out.append(Map(d_p,domain=d.domain,is_support = d.is_support))
        return out 
            
    def rot_align(self,
                  data:List[Map],
                  reference:Map|None = None,
                  anchor_map_id = 0,
                  radial_limits = None,
                  return_correlation=False):
        
        harm = self.ft.harm
        # get reference harmonic coefficients
        if reference is None:
            if self.reference is not None:
                ref = self.reference
            else: 
                raise ValueError("No reference probided + no global reference (self.reference) set. Can not perform rotational alignment without a reference.")
        else:
            ref = reference

        if return_correlation:
            rot,metric,corr = self.find_optimal_rotation(data[anchor_map_id],ref,radial_limits=radial_limits,return_correlation=True)
        else:
            rot,metric = self.find_optimal_rotation(data[anchor_map_id],ref,radial_limits=radial_limits,return_correlation=False)

        rot_data = self.rotate(data,rot)
        if return_correlation:
            return rot_data,metric,corr
        else:
            return rot_data,metric
        
@dataclass
class AlignedAveragerStruct:
    fourier_struct: SphericalFourierTransformStruct = field(
        default_factory=SphericalFourierTransformStruct
    )
    normalization_method: Callable = np.max
    so3_radial_limits: tuple | None = None
    consider_point_inverse: bool = True
    averaging_mode: Literal["pairwise", "single_reference"] = "pairwise"
    n_processes: int | None = None
class DataSourceInterface(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self):
        pass
    @abc.abstractmethod
    def __len__(self):
        pass
    @abc.abstractmethod
    def shape(self):
        pass
class AlignedAverager:
    def __init__(self,struct:AlignedAveragerStruct|None=None,hankel_weights:NDArray|None=None):
        if struct is None:
            struct = AlignedAveragerStruct()
        self.struct = struct
        if hankel_weights is not None:
            self.hankel_weights =  hankel_weights
        else:
            self.hankel_weights = HankelTransformWeights.get_weights(struct.fourier_struct)
        self.fourier = SphericalFourierTransform(struct.fourier_struct,weights=self.hankel_weights)
        self.aligner = Aligner(self.fourier,
                               normalization_method=struct.normalization_method)
        self.so3_radial_limits = struct.so3_radial_limits
        self._const_reference = None

    @staticmethod
    def _create_balanced_binary_tree(n_elements,seed=12345,return_fibers=False):
        """
        Return the reduction indices for n initial leaves of a binary tree.
        such that one tries to minimize the support variance in each layer.
        Where the support of a now 
        
        In each returned array:
          - row i represents node i on the next layer
          - [a, b] means combine nodes a and b
          - [a, -1] means carry node a unchanged
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Randomized input IDs for the first reduction level
            initial_layer = rng.permutation(np.arange(n_elements))
        else:
            initial_layer = np.arange(n_elements)
        
        layers = []
        support = np.ones(n_elements, dtype=int)
        
        while len(support) > 1:
            m = len(support)
            order = np.argsort(support, kind="stable")
            
            k = m // 2
            
            # Pair smallest with largest.
            layer = np.zeros((k+m%2,2),dtype = int)
            layer[:k,0] = order[:k]
            layer[:k,1] = order[k:2*k][::-1]
            
            is_odd = m%2
            # Carry the largest-support node if m is odd.
            if is_odd:
                layer[-1, :] = order[-1]
            layers.append(layer)
            
            # Compute supports of the next layer.
            support = support[order]
            support[:k] += support[k:2*k][::-1]
            support[k+1:k+2] = support[-1]
            support = support[:k+is_odd]
            
        # apply initial random ordering
        for i,pair in enumerate(layers[0]):
            layers[0][i,0] = initial_layer[pair[0]]
            layers[0][i,1] = initial_layer[pair[1]]
            
        tree_fibers = [tuple(np.array([a]) for a in range(n_elements)),
                       tuple(np.unique(a) for a in layers[0])]
        for tree_layer in layers[1:]:
            fiber = tuple(np.unique(
                np.concatenate((tree_fibers[-1][a],tree_fibers[-1][b]))
            ) for a,b in tree_layer)
            tree_fibers.append(fiber)
            
        if return_fibers:
            return layers, tree_fibers
        else:
            return layers

    def align_inversion_and_rotation(self,
                                     data_map:Map,
                                     reference_map:Map = None,
                                     aligner = None,
                                     consider_point_inverse = True):
        a = aligner if isinstance(aligner,Aligner) else self.aligner
        harm = a.ft.harm
        domain = data_map.domain
        data_lm = harm.forward(data_map)
        rot,metric = a.find_optimal_rotation_lm(data_lm,
                                                domain = domain,
                                                radial_limits=self.struct.so3_radial_limits)
        is_point_inverted = None
        if consider_point_inverse:
            data_lm_p = data_lm.point_inverse()
            rot_p,metric_p = a.find_optimal_rotation_lm(data_lm_p,
                                                        domain=domain,
                                                        radial_limits=self.struct.so3_radial_limits)
            is_point_inverted = metric_p>metric
            if is_point_inverted:
                rot = rot_p
                metric = metric_p
                
        return rot,metric,is_point_inverted

    def _single_reference_worker(self,ids
                                 ,data_source:DataSourceInterface|NDArray,
                                 anchor_id,
                                 consider_point_inverse,
                                 **kwargs):
        if len(ids)==0:
            return tuple()
        # instantiate Fourier transform and Aligner since the fourier transforms are not fork safe.
        ft = SphericalFourierTransform(self.struct.fourier_struct,weights = self.hankel_weights)
        a = Aligner(ft,normalization_method=self.struct.normalization_method,reference=self._const_reference[anchor_id])
        outs = kwargs['outputs']
        out_ids = kwargs['output_ids']
        
        variances = None
        for i in ids:
            # get data
            dataset = data_source[i]
            if variances is None:
                # initialize variances since only now I know how long a dataset is.
                variances = tuple(CumulativeVariance() for i in range(len(dataset)))
                
            nc_dataset = a.normalize_and_center(dataset,anchor_id=anchor_id)
            anchor = nc_dataset[anchor_id]
            rot,metric,is_point_inverted = self.align_inversion_and_rotation(anchor,
                                                                             aligner = a,
                                                                             consider_point_inverse=consider_point_inverse)
            
            if consider_point_inverse and is_point_inverted:
                nc_dataset = a.point_invert(nc_dataset)
            aligned_dataset = a.rotate(nc_dataset,rot)
            for d,var in zip(aligned_dataset,variances):
                var.update(d)
        for d_id,var in enumerate(variances):
            outs[3*d_id][out_ids] = var.mean
            outs[3*d_id+1][out_ids] = np.array([var.count])
            outs[3*d_id+2][out_ids] = var.m2
        
    def average_single_reference(self,
                                 data_source:DataSourceInterface|NDArray,
                                 reference_id=0,
                                 anchor_map_id=0,
                                 consider_point_inverse = True,
                                 n_processes = 'auto'):
        n_datasets = len(data_source)
        if reference_id<0 or reference_id>=n_datasets:
            raise ValueError(f"invalid reference_id {reference_id}. Has to be >=0 and  < {n_datasets} (the number of provided datasets.")
        ids = np.delete(np.arange(n_datasets),reference_id)
        
        # Reference creation
        self._const_reference = self.aligner.normalize_and_center(data_source[reference_id])
        shape_types = tuple((d.shape,d.dtype) for d in self._const_reference)
        out_shapes = []
        out_dtypes = []
        for s,t in shape_types:
            out_shapes += [s,(1,),s]
            out_dtypes += [t,int,float]
        #self.aligner.reference = self._const_reference[anchor_map_id]
        
        results = Multiprocessing.process_mp_request(self._single_reference_worker,
                                                     mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes,reduce_arguments=True),
                                                     input_arrays=[ids],
                                                     const_inputs=[data_source,anchor_map_id,consider_point_inverse],
                                                     call_with_multiple_arguments=True,
                                                     split_mode='sequential',
                                                     n_processes = n_processes)
        
        variances = tuple(CumulativeVariance().update(self._const_reference[i]) for i in range(len(self._const_reference)))
        progression = []
        for part_id in range(len(results[0])):
            for d_id,var in enumerate(variances):
                var.merge_from_data(results[3*d_id][part_id],
                                    np.squeeze(results[3*d_id+1][part_id]),
                                    results[3*d_id+2][part_id])
            progression.append(tuple(v.copy() for v in variances))
                
        return variances,progression
    
    def _pairwise_aligning_worker(self,
                                  layer_part,
                                  data_source,
                                  anchor_id,
                                  consider_point_inverse,
                                  do_centering,
                                  **kwargs):
        ft = SphericalFourierTransform(self.struct.fourier_struct,weights = self.hankel_weights)
        a = Aligner(ft,normalization_method=self.struct.normalization_method)
        outs = kwargs['outputs']
        out_ids = kwargs['output_ids'][0]
        for (d_id,ref_id),out_id in zip(layer_part,out_ids):
            ref = data_source[ref_id]
            dat = data_source[d_id]
            if do_centering:
                if isinstance(ref,CumulativeVariance) or isinstance(ref,CumulativeVariance):
                    raise ValueError("can not normalize & center cumulative variances.")
                ref = a.normalize_and_center(ref,anchor_id = anchor_id)
                dat = a.normalize_and_center(dat,anchor_id = anchor_id)
                
            if not isinstance(ref[0],CumulativeVariance):
                ref = [CumulativeVariance().update(r) for r in ref]
            if not isinstance(dat[0],CumulativeVariance):
                dat = [CumulativeVariance().update(d) for d in dat]
                
            rot,metric,is_point_inverted = self.align_inversion_and_rotation(dat[anchor_id].mean,
                                                                             reference = ref[anchor_id].mean,
                                                                             aligner = a,
                                                                             consider_point_inverse=consider_point_inverse)
            dat_mean = [d.mean for d in dat]
            if is_point_inverted:
                dat_mean = a.point_invert(dat_mean)
            new_means = a.rotate(dat_mean,rot)
            
            for  d,new_mean in zip(dat,new_means):
                d.mean = new_mean #note this completely destroys the variance bit only means are correct

            combined_means = [r.merge(d) for r,d in zip(ref,dat)]
            for d_id,cm in enumerate(combined_means):
                mean_out = outs[3*d_id]
                mean_out[out_id] = cm.mean
                count_out = outs[3*d_id+1]
                count_out[out_id] = cm.count
                m2_out = outs[3*d_id+2]
                m2_out[out_id]=cm.m2
            rot_out = outs[-2]
            rot_out[out_id]=utils.euler_to_matrix(rot)
            inv_out = outs[-1]
            inv_out[out_id]=is_point_inverted
                    
    def _pairwise_averaging_worker(self,
                                   ids,
                                   data_source,
                                   rotations,
                                   inversions,
                                   anchor_id,
                                   **kwargs):
        variances = None
        nc  = []
        for i,rot,inv in zip(ids,rotations,inversions):
            # get data
            dataset = data_source[i]
            if variances is None:
                # initialize variances since only now I know how long a dataset is.
                variances = tuple(CumulativeVariance() for i in range(len(dataset)))                
            nc_dataset = a.normalize_and_center(dataset,anchor_id=anchor_id)
            if inv:
                nc_dataset = a.point_invert(nc_dataset)
            aligned_dataset = a.rotate(nc_dataset,rot)
            
            for d,var in zip(aligned_dataset,variances):
                var.update(d)
        out = tuple(var.data for var in variances)
        return out
    
    def average_pairwise(self,
                         data_source:DataSourceInterface|NDArray,
                         anchor_map_id=0,
                         order_seed=12345,
                         n_processes = 'auto'):
        tree,fibers = self._create_binary_tree_ids(len(data_source),seed=order_seed,return_fibers=True)
        shape_types = tuple((d.shape,d.dtype) for d in data_source[0])
        dataset_length = len(shape_types)
        n_datasets = len(data_source)
        rotations = np.array([np.eye(3) for i in range(n_datasets)])
        inversions = np.zeros(n_datasets,dtype=bool)
        
        step_data = data_source
        do_centering = True
        for level_id,tree_level in enumerate(tree):
            level_size = len(tree_level)
            fiber = fibers[level_id]
            out_shapes = []
            out_dtypes = []
            for s,t in shape_types:
                s = (level_size,)+s
                out_shapes+=[s,(level_size,),s]
                out_dtypes+=[t,np.int,np.dtype('float')]
            out_shapes += [(level_size,3,3),(level_size,)]
            out_dtypes *= [np.dtype('float'),np.dtype('bool')]
            results = Multiprocessing.process_mp_request(self._pairwise_aligning_worker,
                                                         mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes),
                                                         input_arrays=[tree_level],
                                                         const_inputs=[step_data,anchor_map_id,do_centering],
                                                         call_with_multiple_arguments=True,
                                                         split_mode='sequential',
                                                         n_processes = n_processes)
            do_centering = False # only do centering + normalization in the first loop iteration
            applied_rotations = results[-2]
            applied_inversions = results[-1]
            results_flat = tuple(
                tuple(CumulativeVariance(mean = results[3*i][j],
                                         count = results[3*i+1][j],
                                         m2 = results[3*i+2][j]) for i in range(dataset_length)
                      ) for j in range(level_size)
            )
            step_data = results_flat.__getitem__
            
            # update rotations and
            for (idx,_),rotation,inversion in zip(tree_level,applied_rotations,applied_inversions):
                # idx is the data id of the current leaf i.e. the structure that was rotated and invertied
                # during alignment.
                changed_ids = fiber[idx]
                inversions[changed_ids] != inversion
                rotations[changed_ids] = rotation[None,...]@rotations[changed_ids]
            
        # now that I found all rotations, lets compute variances
        results = Multiprocessing.process_mp_request(self._pairwise_averaging_worker,
                                                     input_arrays=[np.arange(n_datasets)],
                                                     const_inputs=[data_source,rotations,inversions,anchor_map_id],
                                                     call_with_multiple_arguments=True,
                                                     split_mode='sequential',
                                                     n_processes = n_processes)
        
        variances = tuple(CumulativeVariance(*d) for d in results[0])
        progression = []
        for worker_part in results[1:]:
            for new_data,var in zip(worker_part,variances):
                var.merge_from_data(*new_data)
                progression.append(tuple(v.copy() for v in variances))
        return variances,progression

    
