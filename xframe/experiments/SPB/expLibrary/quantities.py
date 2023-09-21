import logging
log = logging.getLogger('root')
import numpy as np
#from numba import njit
import traceback
import abc
from xframe.library.pythonLibrary import convert_to_slice_if_possible
from xframe.library.mathLibrary import masked_mean
from xframe.library.mathLibrary import combine_means_2D
from xframe.library.mathLibrary import masked_variance
from xframe.library.mathLibrary import combine_variances_ND
from xframe import Multiprocessing

import warnings


def get_slize_by_bounds(array,bounds):
    start,stop = bounds
    array = np.asarray(array)
    amin = bounds[0]
    amax = bounds[1]
    log.info('array = ')
    if isinstance(amin,(int,float)):
        start = np.sum(array<amin)
    if isinstance(amax,(int,float)):
        stop = np.sum(array<=amax)
    return slice(start,stop,1)
        

def update_dict(default,_dict):
    #log.info('default_keys = {}'.format(default.keys()))
    for key,val in default.items():
        if not (key in _dict):
            _dict[key]=val
    return _dict

class Quantity(abc.ABC):
    _Quantity_default_param = {'data_shape':(16,512,128),'roi_mask':True}
    _Quantity_default_options = {'use_multiprocessing':False}
    def __init__(self,name,parameters = {},options = {}):
        parameters = update_dict(self._Quantity_default_param,parameters)
        options = update_dict(self._Quantity_default_options,options)
        self.name = name
        self.param = parameters
        self.opt = options
        self.data_shape = self.param['data_shape']
        # internal roi_mask_inverted is tue on masked points.
        self.roi_mask_inverted = ~np.asarray(self.param['roi_mask'])
        self.use_multiprocessing = self.opt['use_multiprocessing']
        self.n_processed_chunks=0
        self.data = {}
    @abc.abstractmethod
    def _compute(self,data):
        pass
        
    def compute(self,chunk_id,data):
        log.info('inside name ={} c_id = {} q.n_processed_chunks = {}'.format(self.name,chunk_id,self.n_processed_chunks))
        if chunk_id==self.n_processed_chunks:
            log.info('c_id = {} q.n_processed_chunks = {}'.format(chunk_id,self.n_processed_chunks))
            self._compute(data)
            self.n_processed_chunks+=1
            
    def reset(self):
        self.__init__(self.name,self.param,self.opt)

class DependendQuantity(Quantity):
    quantity_classes = []
    def __init__(self,quantities,name,parameters = {},options = {}):
        super().__init__(name,parameters = parameters,options = options)
        self._check_input_quantities(quantities)
        self.parent_quantities = quantities        
    def _check_input_quantities(self,quantities):
        for q_class,q in zip(self.quantity_classes,quantities.values()):
            if not isinstance(q,q_class):
                raise AssertionError('quantity {} is not of type {}'.format(q,q_class))
        
    def compute(self,chunk_id,data):
        n_processed_chunks = self.n_processed_chunks
        for pq in self.parent_quantities.values():
            pq.compute(chunk_id,data)
            
        if chunk_id==self.n_processed_chunks:
            self._compute(data)
            self.n_processed_chunks+=1
    def reset(self):        
        self.__init__(self.parent_quantities,self.name,self.param,self.opt)

        
class Mean2D(Quantity):
    default_options = {}
    default_param = {}
    def __init__(self,name,options = {}, parameters = {}):
        parameters = update_dict(self.default_param,parameters)
        options = update_dict(self.default_options,options)
        super().__init__(name, options = options,parameters=parameters)
        data_shape = self.data_shape
        log.info('data shape = {}'.format(data_shape))
        
        out_shapes=[data_shape,data_shape]
        out_dtypes = [float,int]
        self._mp_mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes,reduce_arguments = True)
        
        mean=np.zeros(data_shape,dtype=float)
        mask=np.zeros(data_shape,dtype=bool)
        counts=np.zeros(data_shape,dtype=int)
        #roi_mask is true where data is allowed and self.roi_mask is true where data is masked
        self.data['mean']=mean
        self.data['mask']=mask
        self.data['counts']=counts
    def _compute(self,data_chunk):
        log.info('data shape = {}'.format(self.data['mean'].shape))
        roi_mask = self.roi_mask_inverted
        datasets,masks = data_chunk['data'],data_chunk['mask']
        if self.use_multiprocessing:
            log.error('Multiprocessing not implemented yet. Switch to normal mode.')
            #mean,mask,counts = self.calc_mean_mp(datasets,masks)
            mean,mask,counts = self.calc_mean_mp(datasets,masks)
        else:
            mean,mask,counts = self.calc_mean_mp(datasets,masks)
        #mean[roi_mask]=0
        #counts[roi_mask]=0
        log.info('mask shape ={} roi_mask shape = {}'.format(mask.shape,roi_mask.shape))
        #mask[roi_mask]=False
        self.data['mean'] = mean
        self.data['counts'] = counts
        self.data['mask'] = mask
    def calc_mean_mp(self,datasets,masks):
        old_mean,old_counts,old_mask = self.data['mean'],self.data['counts'],self.data['mask']
        def _mean(frame_ids,datasets,masks,**kwargs):
            out_mean,out_counts = kwargs['outputs']
            out_ids = kwargs['output_ids']
            tmp_masks = masks[frame_ids].copy()
            tmp_masks[:,self.roi_mask_inverted]=False
            mean,counts  = masked_mean(datasets[frame_ids],tmp_masks,axis=0)
            out_mean[out_ids] = mean
            out_counts[out_ids] = counts
        n_frames = len(datasets)
        data_means,data_counts = Multiprocessing.comm_module.request_mp_evaluation(_mean,self._mp_mode,input_arrays=[np.arange(n_frames)],const_inputs=[datasets,masks],call_with_multiple_arguments=True,split_mode='modulus')
        log.info(f'mean {data_means.dtype} counts = {data_counts.dtype}')
        data_mean,data_count = combine_means_2D(data_means,data_counts)
        log.info(f'mean {data_mean.dtype} counts = {data_count.dtype}')
        mean,counts = combine_means_2D((data_mean,old_mean),(data_count,old_counts))
        log.info(f'mean {mean.dtype} counts = {counts.dtype}')
        mask = (counts != 0)
        return mean,mask,counts
   
class Std2DSimple(DependendQuantity):
    quantity_classes = [Mean2D]
    def __init__(self,quantities,name,parameters = {},options = {}):
        super().__init__(quantities,name,parameters = parameters,options = options)
        data_shape = self.data_shape
        log.info('data shape = {}'.format(data_shape))
        
        out_shapes=[data_shape,data_shape]
        out_dtypes = [float,int]
        self._mp_mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes,reduce_arguments = True)
        
        mean=np.zeros(data_shape,dtype=float)
        mask=np.zeros(data_shape,dtype=bool)
        counts=np.zeros(data_shape,dtype=int)
        #roi_mask is true where data is allowed and self.roi_mask is true where data is masked
        self.data['mean_square']=mean
        self.data['mask']=mask
        self.data['counts']=counts

        self.mean_obj = tuple(self.parent_quantities.values())[0]
        
    def _compute(self,data_chunk):
        mean_square,counts = self.calc_mean_square_mp(data_chunk['data'],data_chunk['mask'])
        self.data['mean_square']=mean_square
        self.data['mask']=self.mean_obj.data['mask'].copy()
        self.data['counts']=counts
        var = mean_square-self.mean_obj.data['mean']**2
        var[var<0]=0
        self.data['std']=np.sqrt(var)
        
    def calc_mean_square_mp(self,datasets,masks):
        old_mean,old_counts = self.data['mean_square'],self.data['counts']
        def _mean(frame_ids,datasets,masks,**kwargs):
            out_mean,out_counts = kwargs['outputs']
            out_ids = kwargs['output_ids']
            tmp_masks = masks[frame_ids].copy()
            tmp_masks[:,self.roi_mask_inverted]=False
            out_mean[out_ids],out_counts[out_ids] = masked_mean(datasets[frame_ids]**2,tmp_masks,axis=0)
        n_frames = len(datasets)
        data_means,data_counts = Multiprocessing.comm_module.request_mp_evaluation(_mean,self._mp_mode,input_arrays=[np.arange(n_frames)],const_inputs=[datasets,masks],call_with_multiple_arguments=True,split_mode='modulus')
        data_mean,data_count = combine_means_2D(data_means,data_counts)
        mean,counts = combine_means_2D((data_mean,old_mean),(data_count,old_counts))
        return mean,counts
    
class Std2D(Quantity):
    default_options = {}
    default_param = {}
    def __init__(self,name,options = {}, parameters = {}):
        parameters = update_dict(self.default_param,parameters)
        options = update_dict(self.default_options,options)
        super().__init__(name, options = options,parameters=parameters)
        data_shape = self.data_shape
        log.info('data shape = {}'.format(data_shape))
        std=np.zeros(data_shape,dtype=float)
        mask=np.zeros(data_shape,dtype=bool)
        counts=np.zeros(data_shape,dtype=int)
        #roi_mask is true where data is allowed and self.roi_mask is true where data is masked
        self.data['std']=std
        self.data['var']=std.copy()
        self.data['mean']=std.copy()
        self.data['mask']=mask
        self.data['counts']=counts
        
    def _compute(self,data_chunk):
        log.info('data shape = {}'.format(self.data['mean'].shape))
        roi_mask = self.roi_mask_inverted
        datasets,masks = data_chunk['data'],data_chunk['mask']
        if self.use_multiprocessing:
            log.error('Multiprocessing not implemented yet. Switch to normal mode.')
            #mean,mask,counts = self.calc_mean_mp(datasets,masks)
            var,mean,mask,counts = self.calc_mean(datasets,masks)
        else:
            var,mean,mask,counts = self.calc_mean(datasets,masks)
        #mean[roi_mask]=0
        #counts[roi_mask]=0
        log.info('mask shape ={} roi_mask shape = {}'.format(mask.shape,roi_mask.shape))
        mask[roi_mask]=False
        self.data['var']=var
        self.data['std']=np.sqrt(var)
        self.data['mean'] = mean
        self.data['counts'] = counts
        self.data['mask'] = mask
    def calc_mean_mp(self,datasets,masks):
        pass
    def calc_mean(self,datasets,masks,**kwargs):
        old_var,old_mean,old_counts,old_mask = self.data['var'],self.data['mean'],self.data['counts'],self.data['mask']
        #log.info('datasets shape = {} masks shape = {}'.format(datasets.shape,masks.shape))
        new_var,new_counts = masked_variance(datasets,masks,axis=0)
        new_mean,_ = masked_mean(datasets,masks,axis=0)
        var,mean,counts = combine_variances_ND((new_var,old_var),(new_mean,old_mean),(new_counts,old_counts))
        mask = np.sum(masks,axis=0).astype(bool) | old_mask
        return [var,mean,mask,counts]
    
class Maximum2D(Quantity):
    default_param = {}
    default_options = {}
    def __init__(self,name,parameters = {}, options = {}):
        parameters = update_dict(self.default_param,parameters)
        options = update_dict(self.default_options,options)
        super().__init__(name,options = options,parameters = parameters)
        data_shape = self.data_shape

        out_shapes=[data_shape,data_shape]
        out_dtypes = [float,bool]
        self._mp_mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes,reduce_arguments = True)
        
        maximum = np.zeros(data_shape,dtype=float)
        mask = np.zeros(data_shape,dtype=bool)
        self.data['maximum']=maximum
        self.data['mask']=mask
    
    def _compute(self,data_chunk):
        roi_mask = self.roi_mask_inverted
        maximum,mask = self.calc_max_mp(data_chunk['data'],data_chunk['mask'])
        log.info('maximum shape ={} mask shape = {}'.format(maximum.shape,mask.shape))
        self.data['maximum']=maximum
        self.data['mask']=mask
    
    def calc_max_mp(self,data,masks):
        log.info('call multiprocess')
        n_frames = len(data)
        max_per_process,mask_per_process = Multiprocessing.comm_module.request_mp_evaluation(self._max_worker,self._mp_mode,input_arrays=[np.arange(n_frames)],const_inputs=[data,masks,self.roi_mask_inverted], call_with_multiple_arguments = True,split_mode = 'modulus')
        log.info('per process shapes = {} | {}'.format(max_per_process.shape,mask_per_process.shape))
        #if len(data_per_process[0])==2:
        #    max_per_process = np.array([d[0] for d in data_per_process.values()])
        #    mask_per_process = np.array([d[1] for d in data_per_process.values()])
        #else:
        #    max_per_process = np.array([d[0][0] for d in data_per_process.values()])
        #    mask_per_process = np.array([d[0][1] for d in data_per_process.values()])
        #data_per_process = np.asarray(data_per_process)
        #log.info('max data shape = {} '.format(mask_per_process.shape))
        data_max = np.max(max_per_process,axis = 0,where=mask_per_process,initial=0)
        data_mask = np.sum(mask_per_process,axis=0).astype(bool)
        
        new_maximum = np.maximum(data_max,self.data['maximum'])
        new_mask = data_mask | self.data['mask']
        return new_maximum,new_mask

    @staticmethod
    def _max_worker(frame_ids,data,masks,roi_mask_inverted,**kwargs):
        out_max,out_masks = kwargs['outputs']
        out_ids = kwargs['output_ids']
        tmp_masks = np.array(masks[frame_ids])
        tmp_masks[:,roi_mask_inverted]=False
        #log.info('shapes = {} | {} | out_max {} | out ids {}'.format(data.shape,masks.shape,out_max.shape,out_ids))
        out_max[out_ids] = np.max(data[frame_ids],axis = 0,where=tmp_masks,initial=0)
        out_masks[out_ids] = np.sum(tmp_masks,axis = 0).astype(bool)
    
class radial_profiles(Quantity):
    def __init__(self,quantities,name,parameters = {},options = {}):
        super().__init__(name,options = options,parameters = parameters)
        data_shape = self.data_shape

        out_shapes=[data_shape,data_shape]
        out_dtypes = [float,bool]
        self._mp_mode = Multiprocessing.MPMode_SharedArray(out_shapes,out_dtypes,reduce_arguments = True)

        self.regridder = self.param['regridder']
        
        self.qs = new_grid[:,0]
        self.phis = new_grid[0,:]
        quantity_shape = self.qs.shape
        
        super().__init__(quantities,name,parameters = parameters,options = options)

        self.roi_modules=roi_modules
        self.data_modules = data_modules
        #roi_mask is true where data is allowed and self.roi_mask is true where data is masked        
        self.roi_mask =  ~np.asarray(roi_mask)
        
        saxs = np.zeros(data_shape,dtype = float)
        mask = np.zeros(data_shape,dtype = bool)
        self.data = {'saxs':saxs,'mask':mask}

        self.use_multiprocessing=use_multiprocessing
        self.background_scale = False
class Saxs(DependendQuantity):
    quantity_classes = [Mean2D]

    def __init__(self,quantities,name,parameters = {},options = {}):
    #def __init__(self,mean_2d,regridder,data_shape = (16,512,128),roi_mask = False,roi_modules = True,data_modules= True,use_multiprocessing=False,options = {}):
        self.regridder = regridder
        self.qs = new_grid[:,0]
        self.phis = new_grid[0,:]
        quantity_shape = self.qs.shape
        
        super().__init__(quantities,name,parameters = parameters,options = options)

        self.roi_modules=roi_modules
        self.data_modules = data_modules
        #roi_mask is true where data is allowed and self.roi_mask is true where data is masked        
        self.roi_mask =  ~np.asarray(roi_mask)
        
        saxs = np.zeros(data_shape,dtype = float)
        mask = np.zeros(data_shape,dtype = bool)
        self.data = {'saxs':saxs,'mask':mask}

        self.use_multiprocessing=use_multiprocessing
        self.background_scale = False
    def _compute(self,data_chunk):
        datasets,masks = data_chunk['data'],data_chunk['mask']
        if self.use_multiprocessing:
            log.error('Multiprocessing not implemented yet. Switch to normal mode.')
            #mean,mask,counts = self.calc_saxs_mp(datasets,masks)
            saxs,mask,counts = self.calc_saxs()
        else:
            saxs,mask,counts = self.calc_saxs()
        self.data['mean'] = mean
        self.data['counts'] = counts
        self.data['mask'] = mask

    def calc_saxs_mp(self):
        pass
        
    def calc_saxs(self):
        opt = self.opt
        regridder = self.regridder
        mean2d = self.parent_quantities[0]
        roi_modules = self.roi_modules
        data_modules = self.data_modules
        roi_mask = self.roi_mask

        
        mean_data=np.zeros(data_shape,dtype = float)
        mask_data=np.zeros(data_shape,dtype = bool)
        tmp_mean = mean_data[roi_modules]
        tmp_mask = mask_data[roi_modules]
        tmp_mean[roi_mask] = mean2d.data['mean'][data_modules][roi_mask].flatten()
        tmp_mask[roi_mask] = mean2d.data['mean'][data_modules][roi_mask].flatten()
        mean_data[roi_modules] = tmp_mean
        mask_data[roi_modules] = tmp_mask
        
        mean_on_polar_grid,mask_on_polar_grid = regridder.regrid(mean_data,mask_data.astype(bool),np.arange(16))
        
        saxs,counts = masked_mean(mean_on_polar_grid,mask_on_polar_grid,axis = 1)
        mask = np.sum(mask_on_polar_grid,axis = 1).astype(bool)

        if opt.get('background',{'subtract':False})['subtract']:
            saxs = self.subtract_background(saxs)
        return saxs,counts,mask


    def subtract_background(self,saxs):
        opt = self.opt
        bg_saxs = opt['background']['saxs']
        bg_saxs_mask = opt['background']['mask']
        scaled_bg_saxs = self.scale_background_to_data(saxs,bg_saxs)
        return saxs-scaled_bg_saxs
    def scale_background_to_data(self,data_saxs,bg_saxs):
        scale_opt = self.opt['background']['scale_opt']
        diff_metric = scale_opt['diff_metric']
        q_range = scale_opt['q_range']
        
        #calculate saxs slice corresponding to scaling q_range
        qs = self.qs
        #log.info('use q_range for scaling = {}'.format(q_range))
        q_slice = get_slize_by_bounds(qs,q_range)
        
        #determine bounds for the scaling factor in the 1d minimization.
        #That is: 1 sigma interval around mean value of data_saxs/background_saxs
        non_zero_mask = (bg_saxs!=0)
        #log.info('data type = {} backgorund type = {}'.format(data_saxs.shape,background_saxs.shape))
        scales_per_q=data_saxs[non_zero_mask]/background_saxs[non_zero_mask]
        diff_mean = np.mean(scales_per_q)
        diff_std = np.std(scales_per_q)
        #prefiousely std of background - data
        bounds = (diff_mean-diff_std,diff_mean+diff_std)
        
        def diff(scale):
            return diff_metric(np.abs(data_saxs[q_slice] - scale*background_saxs[q_slice]))
        scale = minimize_scalar(diff,method='bounded',bounds = bounds,options={'maxiter':scale_opt['max_iterations'],'xatol':scale_opt['abs_tolerance']}).x
        self.background_scale = scale
        return bg_saxs*scale


class Sum1D(Quantity):
    def __init__(self,name,parameters = {},options = {}):
        super().__init__(name,options = options,parameters = parameters)
        self.x_axis_type = self.opt['x_axis']
        self.data['pixels_per_frame']=np.atleast_1d(np.prod(self.data_shape)-np.sum(self.roi_mask_inverted))
        self.data['values'] = np.array([],dtype = float)
        self.data['points'] = np.array([],dtype = float)
        self.data['frame_ids'] = np.array([],dtype = float)
        self.data['mask'] = np.array([],dtype = bool)
        self.data['point_type']=self.x_axis_type
        

    def _compute(self,data_chunk):
        datasets,masks,points = data_chunk['data'],data_chunk['mask'],data_chunk[self.x_axis_type]
        self.data['frame_ids'] = np.concatenate((self.data['frame_ids'],data_chunk['frame_ids']))
        #roi_mask = self.roi_mask_inverted
        if self.use_multiprocessing:
            log.error('Multiprocessing not implemented yet. Switch to normal mode.')
            #mean,mask,counts = self.calc_mean_mp(datasets,masks)
            values,masks = self.reduce_1d_mp(datasets,masks,points)
        else:
            values,masks = self.reduce_1d_mp(datasets,masks,points)
        self.data['values'] = np.concatenate((self.data['values'],values))
        self.data['points'] = np.concatenate((self.data['points'],points))
        self.data['mask'] = np.concatenate((self.data['mask'],masks))
        
    def reduce_1d_mp(self,data,masks,new_points):
        roi_mask = self.param['roi_mask']
        def _sum(frame_ids,data,masks,**kwargs):
            n_frames = len(frame_ids)
            out_sums,out_masks = kwargs['outputs']
            out_ids = kwargs['output_ids']
            flat_data = data[frame_ids].reshape(n_frames,-1)
            flat_masks = (masks[frame_ids]*roi_mask[None,:]).reshape(n_frames,-1)
            
            outs = np.sum(flat_data,where=flat_masks,axis = 1)
            outm = np.sum(flat_masks,axis = 1).astype(bool)
            out_sums[out_ids] = outs
            out_masks[out_ids] = outm
        n_patterns = len(data)
        mp_mode = Multiprocessing.MPMode_SharedArray([n_patterns,n_patterns],[float,bool])
        sums,masks = Multiprocessing.comm_module.request_mp_evaluation(_sum,mp_mode,input_arrays=[np.arange(n_patterns)],const_inputs = [data,masks], call_with_multiple_arguments = True)
        return sums,masks
