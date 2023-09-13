import logging
import numpy as np
#from numba import njit
import traceback
import abc
from xframe.analysis.interfaces import CommunicationInterface
from xframe.library.pythonLibrary import grow_mask
from xframe.library.pythonLibrary import DictNamespace
from xframe.library.mathLibrary import masked_mean
from xframe.library.pythonLibrary import convert_to_slice_if_possible
from xframe.library.mathLibrary import combine_means
from xframe.library.mathLibrary import masked_std
from xframe import Multiprocessing

import warnings


#############################################
# Note in the present analysis and experimen workers filters are applied inside child processes
# so don't use multiprocessing in their apply routines exept if you know exactly what you are doing.
#
# That is alsy why you need to modify the 'data' and 'mask' arrays of a data_chunk in place,
# since they are views in a shared array.
# That means you need to do e.g. data_chunk['data'][:]=0 instead of data_chunk['data'] = np.zeros_like(data_chunk['data'])

log = logging.getLogger('root')

class FilterTools():
    ''' Class that contins static methods that might be important for a single filter as well as a sequence of fliters.'''
    @staticmethod
    def init_masks(filtered_mask=None,modified_mask=None):
        return {'total_filtered':filtered_mask,'total_modified': modified_mask, 'filtered':[filtered_mask] , 'modified':[modified_mask]}    
    
    @staticmethod
    def combine_masks(masks, filtered_mask, modified_mask):
        if len(masks)!=4:
            masks = FilterTools.init_masks(filtered_mask,modified_mask)
        else:                       
            masks['filtered'].append(filtered_mask)
            masks['modified'].append(modified_mask)
            #masks['total_modified'][~masks['total_filtered']] = modified_mask.flatten()
            masks['total_filtered'] |= filtered_mask.flatten()
        return masks
    
    @staticmethod
    def remove_filtered_elements(data_chunk,filtered_mask):
        '''
        Assumes elements of data_chuk are filtered along their first dimension. That means it assumes all masks are 1 dimensional.
        filtered_mask is 1 for data values to exclude and 0 for data values to keep. 
        '''
        # If masks dont seem to be present do nothing
        if np.atleast_1d(filtered_mask).any():
            len_data = len(filtered_mask)
            for key,item in data_chunk.items():
                if isinstance(item, (np.ndarray,tuple,list)):
                    if len(item) == len_data:
                        log.info('replacing: {}'.format(key))
                        data_chunk[key] = item[~filtered_mask]
            log.info('alive')
            n_good_frames = len_data - np.sum(filtered_mask)
            log.info('{} % or {} of frames remain after filtering'.format(n_good_frames/len_data*100,n_good_frames))
        return data_chunk
    
class FilterSequence(FilterTools):
    '''
    Allows to combine list of filters into a single object that can be applied to a data_chunk.
    '''
    def __init__(self,filter_list):        
        self.filter_list = filter_list
    def apply(self,data_chunk,masks={}):
        filter_result = [data_chunk,masks]
        for _filter in self.filter_list:
            #log.info('apply {}'.format(_filter))
            filter_result = _filter.apply(*filter_result)
        return filter_result
    
    def append_filter(self,_filter):
        self.filter_list.append(_filter)        
    def get_data(self):
        f_data ={f.name:f.data for f in self.filter_list}
        return f_data
    def reset_data(self):
        for f in self.filter_list:
            f.reset_data()
class Filter(abc.ABC,FilterTools):
    '''
    Abstract filter class from which all custimiced filters need to inherit.
    Logic is as follows each Filter needs to have a function method _apply(data_chunk:dict,masks:dict), that does the following:
    It does something to data_chunk and records in two masks which elements where modified by the filter and which where filtered (i.e. excluded).
    It then needs to return: new_data_chunk, filtered_mask, modified_mask.

    The filter itself handles the cimbination of the filtered and modified masks.
    '''
    # filtered_mask is true when the data is supposed to be excluded (i.e. was filtered out)
    # modified_mask is true when the data was modified  
    def __init__(self, opt : dict):
        self.opt = opt
        self.name = opt['name']
        self.data = {}
        self.roi_mask = opt['roi_mask']
        #log.info('roi_mask shape = {} any = {}'.format(self.roi_mask.shape,self.roi_mask.any()))
        #log.info('roi_modules = {}'.format(self.roi_modules))
        
    @abc.abstractmethod
    def _apply(self, data_chunk : dict, masks : dict):
        pass

    def reset_data(self):
        self.data={}
                
    #returns new calib_chunk and masks
    def apply(self, calib_chunk : dict , masks : dict = {}):
        filtered_chunk, filtered_mask,modified_mask = self._apply(calib_chunk, masks)
        new_masks = self.combine_masks(masks,filtered_mask,modified_mask)
        return filtered_chunk,new_masks

    
class BraggFilter(Filter):
    def __init__(self, *args):
        super().__init__(*args)
        self.mode = self.opt.get('mode','discard') # 'mask'
        self.max_sigma = self.opt.get('max_sigma',3)
        self.radial_bin_size_in_pixels = self.opt.get('radial_bin_size_in_pixels',3)
        self.pixel_grid = self.opt['data_grid']
        self.q_limits = self.opt.get('q_limits',[None,None])        
        self.fill_value = self.opt.get('fill_value',0)
        self.use_log_scale = self.opt.get('log_scale',False)
        self.radial_overlap_in_pixels = self.opt.get('radial_overlap_in_pixels',0)
        self.max_std_growth = self.opt.get('max_std_growth',0)
        self.min_frames_per_mean = self.opt.get('min_part_len',70)
        
        if self.radial_overlap_in_pixels >= self.radial_bin_size_in_pixels:
            self.radial_overlap_in_pixels = 1
            log.error('Radial overlap bigger or equal to radial bin size setting overlap to 1 pixel')
        
        # if true will calculate the number of 
        self.save_bragg_pixel_count = self.opt.get('save_bragg_pixel_count',False)
        self.pixel_width = abs(self.pixel_grid[0,-1,0,0] - self.pixel_grid[0,-2,0,0])
        #log.info('pixel width = {}'.format(self.pixel_width))
        self.radial_masks, self.qs = self.generate_growing_radial_masks()
    

    def generate_growing_radial_masks(self):
        # important that the masks are ordered from lowest to highest q-bin
        grid = self.pixel_grid
        qs = np.linalg.norm(grid,axis = -1)        
        step = self.radial_bin_size_in_pixels*self.pixel_width
        overlap = self.radial_overlap_in_pixels*self.pixel_width
        q_limits = [qs.min(),qs.max()]
        for i,limit in enumerate(self.q_limits):
            if limit != None:
                q_limits[i] = limit
        #log.info('q_limits = {}'.format(q_limits))
        #q_splits = np.arange(q_limits[0],q_limits[1]+step,step)
        q_splits = np.arange(q_limits[0],q_limits[1]+step,step)
        #radial_masks = tuple((qs>q_splits[i]) & (qs<=q_splits[i+1]+overlap) for i in range(len(q_splits)-1))
        radial_masks = tuple((qs>=q_splits[i]) & (qs<q_splits[i+1]) for i in range(len(q_splits)-1))
        return radial_masks , qs
        
    
    def _apply(self,calib_chunk, masks):
        data = calib_chunk['data']
        mask = calib_chunk['mask']
        if self.mode == 'discard':
            bragg_counts = self.comm.request_mp_evaluation(self.count_bragg_pixels, mp_type = 'shared_array', out_shape = (len(data),), out_dtype = np.dtype(int), argArrays = [data], const_args = [self.radial_masks,self.max_sigma,self.use_log_scale], callWithMultipleArguments = True)
            filtered_mask = bragg_counts.astype(bool)
            #log.info('bragg_counts = {} '.format(bragg_counts))
            modified_mask = np.zeros(filtered_mask.shape,dtype = bool)
        elif self.mode == 'select':
            bragg_counts = self.comm.request_mp_evaluation(self.count_bragg_pixels, mp_type = 'shared_array', out_shape = (len(data),), out_dtype = np.dtype(int), argArrays = [data], const_args = [self.radial_masks,self.max_sigma,self.use_log_scale], callWithMultipleArguments = True)
            filtered_mask = ~bragg_counts.astype(bool)
            #log.info('bragg_counts = {} '.format(bragg_counts))
            modified_mask = np.zeros(filtered_mask.shape,dtype = bool)
        elif self.mode == 'mask':
            # need to detrministically split data based on chunk ids
            data, bragg_mask = self.comm.request_mp_evaluation(self.mask_bragg_pixels, mp_type = 'shared_array_multi', out_shape = data.shape, out_dtypes = [data.dtype,np.dtype(bool)], argArrays = [data, mask], const_args = [], callWithMultipleArguments = True,split_together = True,min_part_len = self.min_frames_per_mean)
            log.info('bad pixels = {} %'.format(np.sum(bragg_mask)/np.sum(mask)*100))
            flat_mask = bragg_mask.reshape(data.shape[0],-1)
            bragg_counts = np.sum(flat_mask, axis = 1)
            calib_chunk['data'] = data
            calib_chunk['mask'] &= ~bragg_mask
            modified_mask = bragg_counts.astype(bool)
            filtered_mask = np.zeros(modified_mask.shape,dtype = bool)
        #log.info('Bragg filter counts = {}'.format(bragg_counts))   
        if self.save_bragg_pixel_count:
            old_counts = self.data.get('n_bragg_pixels',np.array([]))
            new_counts= np.concatenate((old_counts,bragg_counts))
            self.data['n_bragg_pixels'] = new_counts
        return calib_chunk,filtered_mask, modified_mask


    def get_q_sorted_data(self,data):
        masks = self.radial_masks        
        if self.use_log_scale:
            min_data = abs(data.min())+1
            #assumes min of data is 0
            q_sorted_data = tuple(np.log(min_data+data[:,q_mask]) for q_mask in masks)
        else:
            q_sorted_data = tuple(data[:,q_mask] for q_mask in masks)
        return q_sorted_data


    def generate_thresholds(self,data):
        mean = np.mean(data,axis = 0)
        q_means = tuple(mean[q_mask] for q_mask in masks)



    def mask_bragg_pixels(self,data,data_mask,**kwargs):
        part_len = self.min_frames_per_mean
        n_parts = len(data)//part_len
        part_end_ids = [(i+1)*part_len for i in range(n_parts-1)]
        part_end_ids.append(len(data))
        log.info('part end ids = {}'.format(part_end_ids))
        fill_value = self.fill_value
        #log.info('mean shape = {} mask shape = {}'.format(mean.shape,mean_mask.shape))
        bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        very_bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        
        for part_id in range(n_parts):
            part_slice = slice(part_id*part_len,part_end_ids[part_id])
            previouse_std = np.inf
            part_data = data[part_slice]
            part_data_mask = data_mask[part_slice]
            mean, mean_mask = masked_mean(part_data, return_mask = True, axis = 0, where = part_data_mask)
            for q_id,q_mask in enumerate(self.radial_masks):
                q_data = part_data[:,q_mask]            
                stds,stds_mask = masked_std(q_data,fill_value = fill_value, return_mask = True , axis = 1, where = part_data_mask[:,q_mask])
                q_std = masked_mean(stds, fill_value = fill_value, where = stds_mask)
                q_mean = masked_mean(mean[q_mask], fill_value = fill_value, where = mean_mask[q_mask])
                #limiting growth of std for when going to higher q-bins
                if q_std > previouse_std*self.max_std_growth and previouse_std != 0.0:
                    q_std = previouse_std
                previouse_std = q_std
                #log.info('mean {} std {}'.format(q_mean,q_std))
                thresholds = q_mean + q_std*self.max_sigma
                bad_pixel_mask[part_slice,q_mask] = (q_data < -1*thresholds[0]) | (q_data > thresholds[0])
        #    very_bad_pixel_mask[:,q_mask] = (q_data < -1*thresholds[1]) |  (q_data > thresholds[1])
        #for f_id in range(len(data)):
        #    very_bad_pixel_mask[f_id] = grow_mask(very_bad_pixel_mask[f_id],2)
        #bad_pixel_mask |= very_bad_pixel_mask 
        data[bad_pixel_mask] = fill_value
        #log.info('num of bad pixels = {}'.format(np.sum(bad_pixel_mask)))
        #log.info('num of masked pixels = {}'.format(np.sum(~data_mask)))
        #log.info('bad pixels = {} %'.format(np.sum(bad_pixel_mask)/np.sum(data_mask)*100))
        return data,bad_pixel_mask
        
    def mask_bragg_pixels_old(self,data,data_mask,**kwargs):
        masks = self.radial_masks
        max_sigma = self.max_sigma
        fill_value = self.fill_value
        use_log_scale = self.use_log_scale
        if use_log_scale:
            min_data = abs(data.min())+1
            #assumes min of data is 0
            q_sorted_data = tuple(np.log(min_data+data[:,q_mask]) for q_mask in masks)
        else:
            q_sorted_data = tuple(data[:,q_mask] for q_mask in masks)

        bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        very_bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        for q_id,q_data in enumerate(q_sorted_data):
            #log.info('q_data shape = {}'.format(q_data.shape))
            non_zero_mask = q_data != 0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # gives divide by 0 error if q frame is compleatelly masked i.e. zero
                stds = np.std(q_data, axis = 1 , where = non_zero_mask)
                stds[np.isnan(stds)] = 0.0
                means = np.mean(q_data, axis = 1 , where = non_zero_mask)
                means[np.isnan(means)] = 0.0
            abs_q_data = np.abs(q_data)
            bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[0])[:,None]
            very_bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[1])[:,None]
            
        for f_id in range(len(data)):
            very_bad_pixel_mask[f_id] = grow_mask(very_bad_pixel_mask[f_id],2)
        bad_pixel_mask |= very_bad_pixel_mask 
        data[bad_pixel_mask] = fill_value
        return data, bad_pixel_mask
    
    def mask_bragg_pixels_inter_q(self,data,**kwargs):
        masks = self.radial_masks
        max_sigma = self.max_sigma
        fill_value = self.fill_value      
        max_inter_q_sigma = self.max_inter_q_sigma
        q_hist_length = self.q_hist_length

        q_sorted_data = self.get_q_sorted_data(data)
        bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        very_bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        mean_hist= np.full((self.q_hist_length,len(data)),np.nan)
        std_hist = np.full((self.q_hist_length,len(data)),np.nan)
        for q_id,q_data in enumerate(q_sorted_data):
            #log.info('q_data shape = {}'.format(q_data.shape))
            non_zero_mask = q_data != 0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # gives divide by 0 error if q frame is compleatelly masked i.e. zero
                stds = np.std(q_data, axis = 1 , where = non_zero_mask)
                stds[np.isnan(stds)] = 0.0
                means = np.mean(q_data, axis = 1 , where = non_zero_mask)
                means[np.isnan(means)] = 0.0
                
                neither_nan_nor_zero = ~np.isnan(means) & (means !=0)
                thresholds = np.mean(mean_hist, axis = 0 ,where = neither_nan_nor_zero) + np.std(mean_hist, axis = 0 ,where = neither_nan_nor_zero)*max_inter_q_sigma
            nan_thresholds = np.isnan(thresholds)
            thresholds[nan_thresholds] = means[nan_thresholds]
            invalid_means_mask = means > thresholds
            if invalid_means_mask.any():
                #log.info('\n old: {} \n new {}'.format(mean_hist[0][invalid_means_mask],means[invalid_means_mask]))
                #log.info('WARNING: there are invalid means !')
                pass
            means[invalid_means_mask] = mean_hist[0][invalid_means_mask]
            stds[invalid_means_mask] = std_hist[0][invalid_means_mask]


            mean_hist = np.roll(mean_hist,1)
            std_hist = np.roll(std_hist,1)
            mean_hist[0] = means
            std_hist[0] = stds
                    
            abs_q_data = np.abs(q_data)
            bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[0])[:,None]
            very_bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[1])[:,None]
            
        for f_id in range(len(data)):
            very_bad_pixel_mask[f_id] = grow_mask(very_bad_pixel_mask[f_id],2)
        bad_pixel_mask |= very_bad_pixel_mask 
        data[bad_pixel_mask] = fill_value
        return data, bad_pixel_mask
    
    
    @staticmethod    
    def mask_bragg_pixels_old(data, masks, max_sigma, fill_value, use_log_scale,**kwargs):
        if use_log_scale:
            min_data = abs(data.min())+1
            #assumes min of data is 0
            q_sorted_data = tuple(np.log(min_data+data[:,q_mask]) for q_mask in masks)
        else:
            q_sorted_data = tuple(data[:,q_mask] for q_mask in masks)

        bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        very_bad_pixel_mask = np.zeros(data.shape,dtype =bool)
        for q_id,q_data in enumerate(q_sorted_data):
            #log.info('q_data shape = {}'.format(q_data.shape))
            non_zero_mask = q_data != 0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # gives divide by 0 error if q frame is compleatelly masked i.e. zero
                stds = np.std(q_data, axis = 1 , where = non_zero_mask)
                stds[np.isnan(stds)] = 0.0
                means = np.mean(q_data, axis = 1 , where = non_zero_mask)
                means[np.isnan(means)] = 0.0
            abs_q_data = np.abs(q_data)
            bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[0])[:,None]
            very_bad_pixel_mask[:,masks[q_id]] = abs_q_data > (means + stds*max_sigma[1])[:,None]
            
        for f_id in range(len(data)):
            very_bad_pixel_mask[f_id] = grow_mask(very_bad_pixel_mask[f_id],2)
        bad_pixel_mask |= very_bad_pixel_mask 
        data[bad_pixel_mask] = fill_value
        return data, bad_pixel_mask
    
    @staticmethod    
    def count_bragg_pixels(data, masks, max_sigma,use_log_scale,**kwargs):
        #print((data != 0).any())
        if use_log_scale:
            min_data = abs(data.min())+1
            q_sorted_data = tuple(np.log(min_data+data[:,q_mask]) for q_mask in masks)
        else:
            q_sorted_data = tuple(data[:,q_mask] for q_mask in masks)
        stds = np.array(tuple( np.std(data_ring,axis = 1) for data_ring in q_sorted_data))
        #print(q_stored_data)
        # compare abs values in each q ring with sigma limit and count violations 
        bragg_counts_per_ring = np.array(tuple(np.sum(np.abs(data_ring)> (np.mean(data_ring, axis = 1)+std*max_sigma)[:,None],axis = 1) for data_ring,std in zip(q_sorted_data,stds)))
        bragg_counts = np.sum(bragg_counts_per_ring,axis = 0)
        #print(bragg_counts)
        return bragg_counts


class NormalizationFilter(Filter):
    def __init__(self,opt : dict):
        super().__init__(opt)
        
    def reset_data(self):
        self.data={}
        
    def _apply(self,calib_chunk,masks):
        #log.info('applying normalization')
        #log.info('masks = {}'.format(masks))
        roi_mask = self.roi_mask
        #good_frames = ~masks['total_filtered']
        #log.info('good_frames_shape = {}'.format(good_frames.shape))
        #log.info('any bad frames = {}'.format(good_frames.all()))
        data = calib_chunk['data']#[good_frames]
        #log.info('norm_data = {}'.format(data[0,0,0,:100]))
        #log.info('data_shape = {}'.format(data.shape))
        #log.info('data_chunk_shape = {}'.format(calib_chunk['data'].shape))
        #log.info('{} of points are masked by roi'.format(np.sum(~roi_mask)/np.prod(roi_mask.shape)))
        mask = calib_chunk['mask']#[good_frames]
        #mean_per_frame = masked_mean_new(data[:,roi_mask],mask[:,roi_mask],axis = 1)[0]
        #N=len(data)
        #mean_per_frame = masked_mean_new(data.reshape(N,-1),mask.reshape(N,-1),axis = 1)[0]
        #log.info('mean_per_frame = {}'.format(mean_per_frame))
        mean_per_frame = self.calc_mean(data,mask,roi_mask,axis = 1)
        scaling_factors = mean_per_frame
        calib_chunk['data'][:] /= scaling_factors[:,None,None,None]
        is_nan = np.isnan(scaling_factors)
        if is_nan.any():
            log.warning('Nan scalingfactors during application of NormalizationFilter')
            #log.info('scaling_factors shape {} first elements {}'.format(scaling_factors.shape,scaling_factors[:10]))

        out_mask = np.full(len(data),False)
        ## I cheet here a bit with out_mask. I don't count normalization as modification because otherwise all frames would always
        ## be viewd as modified if normalization is applied.
        ## In that sense normalization is not really a filter ... but its convenient to implent it as such if there are filters applied anyways.
        return calib_chunk,out_mask,out_mask

    def calc_mean(self,data,mask,roi_mask,axis = 1):
        #modules = convert_to_slice_if_possible(self.modules)
        # norm_data and norm_mask will be copies of the selected range not views becaus of numpy fancy indexing with booleans in roi_mask.
        #log.info('roi mask shape = {}'.format(roi_mask.shape))
        #log.info('mask shape = {}'.format(mask[:,modules].shape))
        #log.info('norm_data contains nan  = {} percentage = {}%'.format(np.isnan(data).any(),np.sum(np.isnan(data))/(np.prod(data.shape))*100 ))
        n_frames = len(data)
        norm_mask = (mask & roi_mask[None,...]).reshape(n_frames,-1)
        norm_data = data.reshape(n_frames,-1)
        #log.info('normalization roi is everything = {}'.format(self.normalization_mask.all()))
        #log.info('data mask containes {} % unmasked values'.format(np.sum(mask)/np.prod(mask.shape)))
        #log.info('norm mask containes {} % unmasked values'.format(np.sum(norm_mask)/np.prod(norm_mask.shape)))
        #log.info('mean ={}'.format(np.mean(norm_data,axis=axis)))
        means,counts = masked_mean(norm_data,norm_mask, axis = axis)
        #log.info(means)
        return np.atleast_1d(means)


class ADUFilter2D(Filter):
    def __init__(self,opt : DictNamespace):
        super().__init__(opt)
        self.limits = opt.limits
    def _apply(self,calib_chunk,masks):
        limits = self.limits
        data = calib_chunk['data']
        mask = calib_chunk['mask']        
        lower_mask = True
        upper_mask = True
        
        if isinstance(limits[0],(float,int)):
            lower_mask = data >= limits[0]
        if isinstance(limits[1],(float,int)):
            upper_mask = data <= limits[0]
        adu_mask = lower_mask & upper_mask
        new_mask = mask & adu_mask
        calib_chunk['mask'][:] = new_mask
        #log.info('negative data = {}'.format(data[~lower_mask].reshape(-1)[:100]))
        return calib_chunk,np.array(False),np.array(False)


class GainFilter2D(Filter):
    def __init__(self,opt : DictNamespace):
        super().__init__(opt)
        self.mask_opt = opt.mask
        self.gain_values= {'low':2,'medium':1,'high':0}
    def _apply(self,calib_chunk,masks):
        mask = calib_chunk['mask']
        gain = calib_chunk['gain']        

        gain_values = self.gain_values
        gain_masks = {'low':True,'medium':True,'high':True}

        gain_mask = False
        for key in enumerate(gain_values):
            if mask[key]:
                gain_mask = gain_mask | (gain == gain_valus[key])
                
        valid_mask = (~gain_mask)
        calib_chunk['mask'][:] = mask & valid_mask 
        return calib_chunk,False,False

    
class Filter1D(Filter):
    def __init__(self,opt : dict):
        super().__init__(opt)
        metrics = self.opt['metrics']
        if not isinstance(metrics,(tuple,list)):
            metrics = [metrics]
        self.metrics = metrics
        
        limits = self.opt['limits']
        if not isinstance(limits[0],(list,tuple)):
            limits = [limits]
        self.limits = limits
        
        
    def _apply(self,calib_chunk,masks):
        #good_frames = ~masks['total_filtered']
        data = calib_chunk['data']#[good_frames]
        mask = calib_chunk['mask']#[good_frames]
        metrics= self.metrics
        metric_values = self.calc_metric_values(data,mask)
        #log.info(metric_values)
        #log.info('max {} min {}'.format(metric_values[0].max(),metric_values[0].min()))
        #filtered_mask = np.zeros(len(calib_chunk['data']),dtype = bool)
        filtered_mask = self.apply_limits(metric_values[0])
        #log.info('{} of {} filtered'.format(np.sum(filtered_mask),len(filtered_mask)))
        return calib_chunk,filtered_mask,np.full(len(data),False)

    def calc_metric_values(self,data,mask,*args,**kwargs):
        metrics= self.metrics
        metric_values=np.zeros((len(data),len(metrics)))
        #log.info('data shape = {}'.format(data.shape))
        #log.info('mean data = {} '.format(np.mean(data, axis = tuple(range(1,len(data.shape))), where = mask)))
        for m_id in range(len(metrics)):
            metric=metrics[m_id]
            #metric_values[:,m_id] = np.mean(data, axis = tuple(range(1,len(data.shape))), where = mask)
            for f_id in range(len(data)):
                f_mask = mask[f_id]
                f_dat=data[f_id][f_mask]
                if len(f_dat) > 0:
                    metric_values[f_id,m_id] = metric(f_dat)
                else:
                    metric_values[f_id,m_id] = 0
        #log.info('metric values = {} '.format(metric_values))
        return  [metric_values]

    def apply_limits(self,metric_values):
        masks = np.zeros(metric_values.shape,dtype = bool)
        for index,limit in enumerate(self.limits):
            values = metric_values[:,index]
            if limit[0] == None:
                limit[0] = values.min()
            if limit[1] == None:
                limit[1] = values.max()
            #log.info('{} metric values are grater than {}'.format(np.sum(metric_values>limit[1]),limit[1]))
            #log.info('{} metric values are smaller than {}'.format(np.sum(metric_values<limit[0]),limit[0]))
            masks[:,index] = (values<limit[0]) | (values>limit[1])
        # logical and
        #log.info('number of unmasked values = {}'.format(np.sum(masks)))
        combined_mask = np.sum(np.array(masks), axis = 1).astype(bool)
        #log.info('number of unmasked values combined = {}'.format(np.sum(combined_mask)))
        return combined_mask


class LitPixels(Filter1D):
    def __init__(self,opt : dict):
        opt['metrics'] = [self.metric]
        super().__init__(opt)
        self.lit_threshold = self.opt['lit_threshold']

        
        limits = self.opt['limits']
        if not isinstance(limits[0],(list,tuple)):
            limits = [limits]
        self.limits = limits
    def metric(self,data,n_pixels,*args,**kwargs):
        lit_pixel_fraction = np.sum(data>self.lit_threshold)/n_pixels
        return lit_pixel_fraction
        
    
    def calc_metric_values(self,data,mask,*args,**kwargs):
        metrics= self.metrics
        metric_values=np.zeros((len(data),len(metrics)))
        #log.info('data shape = {}'.format(data.shape))
        #log.info('mean data = {} '.format(np.mean(data, axis = tuple(range(1,len(data.shape))), where = mask)))
        for m_id in range(len(metrics)):
            metric=metrics[m_id]
            #metric_values[:,m_id] = np.mean(data, axis = tuple(range(1,len(data.shape))), where = mask)
            for f_id in range(len(data)):
                f_mask = mask[f_id]
                n_pixels = np.prod(f_mask.shape)
                f_dat=data[f_id][f_mask]
                if len(f_dat) > 0:
                    metric_values[f_id,m_id] = metric(f_dat,n_pixels)
                else:
                    metric_values[f_id,m_id] = 0
        #log.info('metric values = {} '.format(metric_values))
        return  [metric_values]
