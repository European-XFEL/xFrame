import importlib
import logging
import numpy as np

from xframe.library import physicsLibrary as pLib
from xframe.library.pythonLibrary import convert_to_slice_if_possible

from xframe.experiment.interfaces import ExperimentWorkerInterface
from xframe.experiment.interfaces import CommunicationInterface

from .detectors.agipd import AGIPD
from .analysisLibrary import filters
from .analysisLibrary.filters import Filter,FilterSequence,FilterTools
from .analysisLibrary.rois import ROIManager
from .analysisLibrary.misk import Selection
from xframe.library.pythonLibrary import split_into_simple_slices
from multiprocessing import cpu_count
from psutil import virtual_memory
from xframe import settings
from xframe import database
from xframe import Multiprocessing

log=logging.getLogger('root')


class Worker(ExperimentWorkerInterface):
    def __init__(self,detector=False,calibrator=False):
        opt = settings.experiment
        comm_module = Multiprocessing.comm_module
        self.opt=opt
        self.data_mode = opt.get('data_mode','raw')
        self.sample_distance = opt.get('sample_distance',700)# in mm
        self.x_ray_energy = opt.get('x_ray_energy',6010)# in eV
        self.x_ray_wavelength = pLib.energy_to_wavelength(self.x_ray_energy)
        self.comm_module=comm_module
        self.db = database.experiment
        if isinstance(detector,bool):
            detector = AGIPD(self.db)
        self.detector = detector
        #if isinstance(calibrator,bool):
        #    calibrator = AGIPD_VDS_Calibrator()
        #self.calibrator = calibrator
        self.info=self._generate_info()
        self.detector.origin = np.array([0,0,self.sample_distance])
        self.custom_mask = np.full(self.detector.data_shape,True)
        use_custom_mask = opt.get('custom_mask',False)
        if use_custom_mask:
            self.custom_mask = self.db.load('custom_mask')

        self.roi_manager = self.load_roi_manager()
        #self.rois = self.create_ROIs()

        #self.filter_sequence = self.load_filters()
        #background_opt = opt['calibrator']['subtract_background']
        #self.apply_background_subtraction = background_opt.get('apply',False)
        #self.background = 0
        #if self.apply_background_subtraction:
        #    self.background = self.db.load('background',path_modifiers={'name':background_opt['name']})['background']
        
        self.ids_by_run = {}
        self.cell_ids = None
        self.pulse_ids = None
        self.train_ids = None
        self.nframes = None


    ## get general information from experiment and run to be analyzed##
    def _generate_info(self):
        #data_info =self.db.load('info',runs=self.opt['runs'],data_mode=self.data_mode)
        #info = {**data_info,'sample_distance':self.sample_distance,'x_ray_energy':self.x_ray_energy,'x_ray_wavelength':self.x_ray_wavelength}
        #data_info =self.db.load('info',runs=self.opt['runs'],data_mode=self.data_mode)
        info = {'sample_distance':self.sample_distance,'x_ray_energy':self.x_ray_energy,'x_ray_wavelength':self.x_ray_wavelength}
        return info
    def set_ids_for_run(self,run):
        ids_by_run = self.ids_by_run
        run_str=str(run)
        if run_str in ids_by_run:
            self.cell_ids = ids_by_run[run_str]['cell_ids']
            self.pulse_ids = ids_by_run[run_str]['pulse_ids']
            self.train_ids = ids_by_run[run_str]['train_ids']
            self.baseline_shifts = ids_by_run[run_str]['baseline_shift']
            self.frame_mask = ids_by_run[run_str]['frame_mask']
        else:
            load = self.db.load
            pm = {'run':run,'data_mode':self.data_mode}
            cell_ids = load('cell_ids',path_modifiers = pm)
            pulse_ids = load('pulse_ids',path_modifiers = pm)
            train_ids = load('train_ids',path_modifiers = pm)
            frame_mask = load('frame_mask',path_modifiers = pm)
            baseline_shifts= load('baseline_shift',path_modifiers = pm)
            self.ids_by_run[run_str] = {'cell_ids':cell_ids,'pulse_ids':pulse_ids, 'train_ids':train_ids,'frame_mask':frame_mask,'baseline_shift':baseline_shifts}
            self.cell_ids = cell_ids
            self.pulse_ids = pulse_ids
            self.train_ids = train_ids
            self.frame_mask = frame_mask
            self.baseline_shifts = baseline_shifts
        self.nframes=len(self.cell_ids)
        
    
    ## setup filters to be applied to the data stream ##
    def load_roi_manager(self):
        used_roi_names = self.collect_used_roi_names()
        roi_manager = ROIManager(self.get_geometry(),used_rois = used_roi_names,rois_dict = self.opt['ROIs'])
        return roi_manager
    def collect_used_roi_names(self):
        filter_opt = self.opt['filters']
        roi_names = []
        for s_name in self.opt['filter_sequence']:
            try:
                roi_names += filter_opt[s_name].get('ROIs',['all'])
            except KeyError as e:
                log.warning('Quantity {} does not exist.Skipping.'.format(name))
        used_roi_names = list(np.unique(roi_names))
        return used_roi_names 
    
    def load_filters(self):
        filter_sequence = []
        for filter_name in self.opt.get('filter_sequence',[]):
            filter_opt = self.opt['filters'][filter_name]
            filter_opt['name']=filter_name
            roi_mask = self.roi_manager.get_combined_roi_mask(filter_opt.get('ROIs',['all']))
            filter_opt['roi_mask'] = roi_mask
            has_unmasked_values = roi_mask.any()
            if not has_unmasked_values:
                log.warning('The roi of filter {} has no unmasked values. Skipping.'.format(filter_name))
                continue
            class_name = filter_opt.get('class')
            try:
                filter_class = getattr(filters,class_name)
                #log.info('Creating filter from class {}'.format(filter_class))
            except AttributeError as e:
                log.warning('Filter {} not found in {}, skipping this Filter'.format(class_name,filters))
            filter_sequence.append(filter_class(filter_opt))
        filter_sequence = FilterSequence(filter_sequence)
        return filter_sequence            

    
    ## load data from database and apply filters ##
    def _selection_to_chunks(self,opt:dict):
        ex_opt = settings.experiment
        run = opt['run']
        if not str(run) in self.info:
            # The call to load info potentially creates a new vds file if non is existing.
            self.info[str(run)] = self.db.load('info',runs=[run],data_mode=self.data_mode)[str(run)]
            
        modules =np.array(opt['modules'],dtype = int)
        frame_range = opt['frame_range']
        selection = {key:Selection(key,item) for key,item in opt['selection'].items()}
        good_cells = self.opt['good_cells']
        #calib=self.calibrator
        #log.info('calibrator wants to set ids.')

        ### self set ids is iportant otherwise metadata will be wrong
        self.set_ids_for_run(run)
        
        train_selection = selection['trains']
        min_train_id = np.min(self.train_ids)
        n_trains = len(self.train_ids)
        
        
        #log.info(f'train ids min = {np.max(self.train_ids)} max = {np.max(self.train_ids)}')
        log.info('selection = {}'.format({s.name:s.data_range for s in selection.values()}))
        comm = self.comm_module
        
        chunk_opt = {
            'run':run,
            'data_mode':self.data_mode,
            'n_cpus':comm.n_cpus(),
            'free_mem':comm.free_mem()*self.opt['RAM_multiplier'],
            'selection': selection,
            'n_frames': opt['n_frames'],
            'frame_range': frame_range,
            'good_cells': good_cells,
            'in_multiples_of': opt['in_multiples_of'],
            'modules' : opt['modules']
        }
        db_load = self.db.load        
        chunks = db_load('chunks',**chunk_opt)
        if ex_opt.get('load_pump_diod_data',False):
            diod_dict = self.db.load('pump_diod',path_modifiers={'run':run})
            diod_on = diod_dict['diod_on'].astype(bool)
            diod_trains = diod_dict['trainIds']
            min_train_id = self.train_ids.min()
        #log.info(f'chunks = {chunks}')
        chunked_metadata=[]
        for chunk in chunks:
            chunk_train_ids = self.train_ids[chunk]
            _dict = {'pulse_ids':self.pulse_ids[chunk],'cell_ids':self.cell_ids[chunk],'train_ids':chunk_train_ids,'good_frames_mask':self.frame_mask[chunk],'baseline_shift':self.baseline_shifts[chunk]}
            if ex_opt.get('load_pump_diod_data',False):
                if isinstance(diod_on,np.ndarray):
                    _dict['diod_on'] = diod_on[chunk_train_ids-min_train_id]
                    _dict['diod_trains'] = diod_trains[chunk_train_ids-min_train_id]
            chunked_metadata.append(_dict)            
        return chunks,chunked_metadata

    def _process_data_chunk_worker(self,modules,slices,run,data_slices,output_slices,data_mode,filter_sequence,custom_mask,frame_mask,module_id_lookup,**opt):
        data,mask,gain,filtered_mask = opt['outputs']
        #process_events = opt['events']
        process_id = opt['local_name']
        n_processes = opt['number_of_processes']
        synchronize = opt['synchronize']

        # let database routin populate the outarrays
        self.db._load_data_chunk_worker(modules,slices,module_id_lookup,run,data_slices,output_slices,data_mode,**opt)
        #process_events[process_id][0].set()
        
        #wait until all other workers finished loading the data into the shared outputs
        #for p_id,event in enumerate(process_events):
        #    if p_id == process_id:
        #        continue
        #    success = event[0].wait(timeout = 60*20)
        #    if not success:
        #        log.warning('Process {} didnt synchronice after 20 min stop waiting in process {}'.format(p_id,process_id))
        synchronize(timeout = 60*20)
        #log.info('all processes synchronized')
        
        #take this workers pice of data: assumes module,frames,long dim,short dim layout
        n_frames = data.shape[1]
        selection = slice(process_id,None,n_processes)
        #log.info('selection = {}'.format(selection))
        data_view = np.swapaxes(data[:,selection],0,1)
        mask_view = np.swapaxes(mask[:,selection],0,1)        
        mask_view[:] &= custom_mask[None,:]
        gain_view = np.swapaxes(gain[:,selection],0,1)
        # Note that all variables called *_view are really numpy views into the big shared arrays data,mask,gain.
        # If one edits their elements the elements of the shared arrays change.
        views = {'data':data_view,'mask':mask_view,'gain':gain_view}
        init_filtered_mask = ~frame_mask[selection]
        #log.info('filtered_mask shape = {}'.format(init_filtered_mask.shape))
        init_modified_mask = np.zeros_like(init_filtered_mask)
        
        
        # The following is bad if not all workers are past the synchronisation phase yet.S        
        
        views,masks = filter_sequence(views,masks=FilterTools.init_masks(filtered_mask = init_filtered_mask,modified_mask = init_modified_mask))
        #log.info('filtered_mask shape after sequence = {}'.format(masks['total_filtered'].shape))
        filtered_mask[selection] = masks['total_filtered']
        #process_events[process_id][1].set()
        #wait until all other workers finished generating the filter_masks
        #for p_id,event in enumerate(process_events):
        #    if p_id == process_id:
        #        continue
        #    success = event[1].wait(timeout = 60*20)
        #    process_events[process_id][0].clear()
        #    if not success:
        #        log.warning('Process {} didnt synchronice after 20 min stop waiting in process {}'.format(p_id,process_id))
        synchronize(timeout = 60*20)

    def _process_data_chunk_worker2(self,modules,run,data_slices,output_slices,data_mode,filter_sequence,custom_mask,frame_mask,module_id_lookup,**opt):
        data,mask,gain,filtered_mask = opt['outputs']
        #process_events = opt['events']
        process_id = opt['local_name']
        n_processes = opt['number_of_processes']
        synchronize = opt['synchronize']

        
        # let database routin populate the outarrays
        for module in modules:
            self.db._load_data_chunk_worker2(module,module_id_lookup,run,data_slices,output_slices,data_mode,**opt)
        #process_events[process_id][0].set()
        
        #wait until all other workers finished loading the data into the shared outputs
        #for p_id,event in enumerate(process_events):
        #    if p_id == process_id:
        #        continue
        #    success = event[0].wait(timeout = 60*20)
        #    if not success:
        #        log.warning('Process {} didnt synchronice after 20 min stop waiting in process {}'.format(p_id,process_id))
        synchronize(timeout = 60*20)
        #log.info('all processes synchronized')
        
        #take this workers pice of data: assumes module,frames,long dim,short dim layout
        n_frames = data.shape[1]
        selection = slice(process_id,None,n_processes)
        #log.info('selection = {}'.format(selection))
        data_view = np.swapaxes(data[:,selection],0,1)
        mask_view = np.swapaxes(mask[:,selection],0,1)        
        mask_view[:] &= custom_mask[None,:]
        gain_view = np.swapaxes(gain[:,selection],0,1)
        # Note that all variables called *_view are really numpy views into the big shared arrays data,mask,gain.
        # If one edits their elements the elements of the shared arrays change.
        views = {'data':data_view,'mask':mask_view,'gain':gain_view}
        init_filtered_mask = ~frame_mask[selection]
        #log.info('filtered_mask shape = {}'.format(init_filtered_mask.shape))
        init_modified_mask = np.zeros_like(init_filtered_mask)
        
        
        # The following is bad if not all workers are past the synchronisation phase yet.S        
        
        views,masks = filter_sequence(views,masks=FilterTools.init_masks(filtered_mask = init_filtered_mask,modified_mask = init_modified_mask))
        #log.info('filtered_mask shape after sequence = {}'.format(masks['total_filtered'].shape))
        filtered_mask[selection] = masks['total_filtered']
        #process_events[process_id][1].set()
        #wait until all other workers finished generating the filter_masks
        #for p_id,event in enumerate(process_events):
        #    if p_id == process_id:
        #        continue
        #    success = event[1].wait(timeout = 60*20)
        #    process_events[process_id][0].clear()
        #    if not success:
        #        log.warning('Process {} didnt synchronice after 20 min stop waiting in process {}'.format(p_id,process_id))
        synchronize(timeout = 60*20)
                
    def get_data(self,opt:dict):
        ex_opt = settings.experiment
        run = opt['run']
        if not str(run) in self.info:
            # The call to load info potentially creates a new vds file if non is existing.
            self.info[str(run)] = self.db.load('info',runs=[run],data_mode=self.data_mode)[str(run)]
            
        modules =np.array(opt['modules'],dtype = int)
        module_id_lookup = {m:m_id for m_id,m in enumerate(modules)} 
        frame_range = opt['frame_range']
        good_cells = self.opt['good_cells']
        
        self.roi_manager.used_modules = np.asarray(modules)
        #log.info('modules = {}'.format(modules))
        nroi = self.roi_manager.rois['normalization_region']
        #log.info('normalization mask ={} complete mask = {}'.format(nroi.mask.shape,nroi.mask_complete.shape))
        
        filter_sequence = self.load_filters()
        apply_filter_sequence = filter_sequence.apply
        chunks,chunked_metadata = self._selection_to_chunks(opt)
            
        comm = self.comm_module        
        n_frames_to_process=np.sum(tuple(chunk.shape[0] for chunk in chunks))        
        custom_mask = self.custom_mask[modules]
        
        data_shape=self.info[str(run)]['data_shape']        
        n_processes = ex_opt.n_processes
        
        if self.opt['data_mode'] == 'proc':
            def mp_process_data(c_id,chunk):                
                frame_mask = chunked_metadata[c_id]['good_frames_mask']
                log.info('processing chunk {} of {} with {} patterns'.format(c_id+1,len(chunks),len(chunk)))                
                chunk_shape = (len(modules),len(chunk))+data_shape[1:]
                len_chunk = len(chunk)
                frame_slices,out_slices = split_into_simple_slices(chunk,return_sliced_args=True)
                #log.info(frame_slices)
                slice_length = [(s.stop-s.start) for s in frame_slices]
                #log.info('mean slice_length = {} std = {}'.format(np.mean(slice_length),np.std(slice_length)))
                
                output_shapes =  [chunk_shape,chunk_shape,chunk_shape,(len_chunk,)]
                output_dtypes = [np.dtype('float32'),np.dtype('bool'),np.dtype('uint8'),np.dtype('bool')] 
                mp_mode = Multiprocessing.MPMode_SharedArray(output_shapes,output_dtypes)
                data,mask,gain,filtered_mask= self.comm_module.request_mp_evaluation(self._process_data_chunk_worker,mp_mode,input_arrays = [modules,np.arange(len(frame_slices))], const_inputs = [run,frame_slices,out_slices,self.data_mode,apply_filter_sequence,custom_mask,frame_mask,module_id_lookup], call_with_multiple_arguments = True, split_mode = 'modulus', n_processes = n_processes)
                #data,mask,gain,filtered_mask= self.comm_module.request_mp_evaluation(self._process_data_chunk_worker2,mp_mode,input_arrays = [modules], const_inputs = [run,frame_slices,out_slices,self.data_mode,apply_filter_sequence,custom_mask,frame_mask,module_id_lookup], call_with_multiple_arguments = True, split_mode = 'modulus', n_processes = n_processes)                
                data = data.swapaxes(0,1)
                mask = mask.swapaxes(0,1)
                gain = gain.swapaxes(0,1)
                out_dict =  {'data':data,'mask':mask,'gain':gain,'frame_ids':chunk,'filtered_frames_mask':filtered_mask,**chunked_metadata[c_id]}
                #log.info('removing filtered_elements')
                #out_dict = FilterTools.remove_filtered_elements(out_dict,filtered_mask)
                #log.info('ex worker mask containes {} % unmasked values'.format(np.sum(mask_chunk)/np.prod(mask_chunk.shape)))
                #log.info('datatype data_chunk = {}'.format(raw_chunk[0].dtype))
                #log.info('datatype mask_chunk = {}'.format(mask_chunk[0].dtype))
                return out_dict

        else:
            raise NotImplementedError('Currently only loading from processed data is supported. (data_mode has to be "proc")')
        
        for c_id,chunk in enumerate(chunks):
            yield mp_process_data(c_id,chunk)



    ## setup detector geometry in reciprocal space ##
    ## i.e. the reciprocal space coordinates associated with each pixel id the detector ##
    def get_pixel_grid_reciprocal(self,approximation='None',out_coord_sys='spherical'):
        grid = self.detector.pixel_grid
        pixel_grid = pLib.pixel_grid_to_scattering_grid(grid,self.x_ray_wavelength,approximation = approximation, out_coord_sys = out_coord_sys)
        return pixel_grid

    def _get_geometry_rois(self, approximation='None'):
        mask = self.detector.sensitive_pixel_mask
        data_shape = self.detector.data_shape
        grid = self.detector.pixel_grid[:,:-1,:-1][mask].reshape(data_shape+(3,))        
        data_grid_spher = pLib.pixel_grid_to_scattering_grid(grid,self.x_ray_wavelength,approximation = approximation, out_coord_sys = 'spherical')
        return {'data_grid_spherical':data_grid_spher,'data_shape':data_shape,'asic_slices':self.detector.asic_slices}

    def get_geometry(self, approximation='None',out_coord_sys='spherical'):
        grid = self.detector.pixel_grid
        framed_grid = self.detector.framed_pixel_grid
        framed_centers = self.detector.framed_pixel_centers
        pixel_grid = pLib.pixel_grid_to_scattering_grid(grid,self.x_ray_wavelength,approximation = approximation, out_coord_sys = out_coord_sys)
        framed_pixel_grid = pLib.pixel_grid_to_scattering_grid(framed_grid,self.x_ray_wavelength,approximation = approximation, out_coord_sys = out_coord_sys)
        framed_pixel_centers = pLib.pixel_grid_to_scattering_grid(framed_centers,self.x_ray_wavelength,approximation = approximation, out_coord_sys = out_coord_sys)

        mask = self.detector.sensitive_pixel_mask
        data_shape = self.detector.data_shape
        grid = self.detector.pixel_grid[:,:-1,:-1][mask].reshape(data_shape+(3,))        
        data_grid_spher = pLib.pixel_grid_to_scattering_grid(grid,self.x_ray_wavelength,approximation = approximation, out_coord_sys = 'spherical')
        
        return {'pixel_grid':pixel_grid,'lab_pixel_grid':grid,'framed_pixel_grid':framed_pixel_grid,'framed_lab_pixel_grid':framed_grid,'framed_pixel_centers':framed_pixel_centers,'framed_lab_pixel_centers':framed_centers,'mask':mask,'framed_mask':self.detector.framed_sensitive_pixel_mask,'data_shape':data_shape,'asic_slices':self.detector.asic_slices,'data_grid_spherical':data_grid_spher}
    

    ## satisfy interface ##
    ## currently empty since experiment and analysis workers are not disconnected in separate processes yet##
    def start_working(self):
        pass           
