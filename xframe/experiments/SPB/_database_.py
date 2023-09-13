import numpy as np
import os
from os import path as op
import time
import logging
import traceback
import re
import glob


from xframe.interfaces import DatabaseInterface
from xframe.library.mathLibrary import plane3D
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library.gridLibrary import GridFactory
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import double_first_dimension
from xframe.library.pythonLibrary import convert_to_slice_if_possible
from xframe.library.pythonLibrary import split_into_simple_slices
from xframe.library.pythonLibrary import split_ids_by_unique_values
from xframe.library.pythonLibrary import getArrayOfArray
from xframe.database.database import DefaultDB
from xframe import Multiprocessing


class ExperimentDB(DefaultDB,DatabaseInterface):
    def __init__(self,**folders_files):
        super().__init__(**folders_files)
        self.data_path = self.get_path('vds_agipd_data',is_file=False)
        self.mask_path = self.get_path('vds_agipd_mask',is_file=False)
        self.gain_path = self.get_path('vds_agipd_gain',is_file=False)
        self.train_path = self.get_path('vds_agipd_trainId',is_file=False)
        self.pulse_path = self.get_path('vds_agipd_pulseId',is_file=False)
        self.cell_path = self.get_path('vds_agipd_cellId',is_file=False)
        self.baseline_shift_path = self.get_path('vds_agipd_baseline_shift',is_file=False)
        self.frame_mask_path = self.get_path('vds_agipd_frame_mask',is_file=False)
        self.pump_diod_path = self.get_path('h5_pump_diod',is_file=False)
        self.comm_module = Multiprocessing.comm_module


    ## load_default, save_default will be depricated use save_direct,load_direct inherited from default_DB
    def load_default(self,name,**kwargs):
        return super().load(name,skip_custom_methods=True,**kwargs)
    def save_default(self,name,data,**kwargs):
        super().save(name,data,skip_custom_methods=True,**kwargs)
        

    ## general methods
    def load_vds(self,name,**opt):
        '''Custom load routine for agipd vds files. If vds files exest load them otherwise if options allow vds creation create vds and load it.'''
        path_modifiers = opt['path_modifiers']
        file_path = self.get_path('vds',path_modifiers = path_modifiers)
        #log.info(file_path)
        if op.exists(file_path):
            data = self.load_direct('vds',**opt)
        else:
            log.info('vds_file: {} \n does not exist yet.'.format(file_path))
            if opt['allow_vds_creation']:
                log.info('creating vds file. data_mode = {}'.format(path_modifiers['data_mode']))
                run = path_modifiers['run']
                from_raw_data = (path_modifiers['data_mode'] == 'raw')                
                self.create_vds(run, from_raw_data,n_processes=opt.get('n_processes',False))
                log.info('finished vds creation.')
                data = self.load_direct('vds',**opt)
        #log.info('finished lading for {}.  exiting load vds'.format(path_modifiers['data_mode']))
        return data
    def _load_vds_old(self,name,**opt):
        path_modifiers = opt['path_modifiers']
        file_path = self.get_path('vds',path_modifiers = path_modifiers)
        if op.exists(file_path):
            data = self.load_default('vds',**opt)
        else:
            log.info('vds_file: {} \n does not exist.'.format(file_path))
            if opt['allow_vds_creation']:
                log.info('creating vds file. mode = {}'.format(path_modifiers['data_mode']))
                run = path_modifiers['run']
                from_raw_data = (path_modifiers['data_mode'] == 'raw')
                self.create_vds(run, from_raw_data,create_modulewise_vds = opt.get('create_modulewise_vds',False),create_complete_vds=True,n_processes=opt.get('n_processes',False))
                log.info('finished vds creation call load_default')
                data = self.load_default('vds',**opt)
        log.info('finished lading for {}.  exiting load vds'.format(path_modifiers['data_mode']))
        return data
    
    def load_info(self,name,**opt):
        runs = opt['runs'] #int or iterable
        data_mode = opt['data_mode']
        
        if isinstance(runs,int):
            runs = (runs,)
        infos = {}
        for run in runs:
            #log.info('type of run = {}'.format(run))
            with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
                #log.info('opened {} file'.format(data_mode))
                ## loads general information about the dataset assumes modules,n_patterns,x_pixel_axis(long),y_pixel_axis(short) data shape
                #valid_frames_mask = self._get_valid_frames_mask(h5_file)
                data_set = h5_file[self.data_path]
                shape = data_set.shape
                infos[str(run)]={'n_frames': shape[1],
                                 'frame_size_Bytes': np.prod(data_set.shape[2:])*data_set.dtype.itemsize,
                                 'data_shape':shape,
                                 'cells': np.unique(h5_file[self.cell_path][:1000])
                }
            #log.info('finished generating info.')
        return infos

    def load_cell_ids(self,name, **opt):
        run = opt['path_modifiers']['run']
        data_mode = opt['path_modifiers']['data_mode']
        
        frames = opt.get('frames',slice(None))
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            data=h5_file[self.cell_path][frames]#frame,cell_id
        return data
    def load_pulse_ids(self,name, **opt):
        run = opt['path_modifiers']['run']
        data_mode = opt['path_modifiers']['data_mode']
                
        frames = opt.get('frames',slice(None))
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            data=h5_file[self.pulse_path][frames]#frame,pulse_id
        return data
    def load_train_ids(self,name, **opt):
        run = opt['path_modifiers']['run']
        data_mode = opt['path_modifiers']['data_mode']
                
        frames = opt.get('frames',slice(None))
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            data=h5_file[self.train_path][frames]#frame,train_id
        return data
    
    def load_baseline_shift(self,name, **opt):
        run = opt['path_modifiers']['run']
        data_mode = opt['path_modifiers']['data_mode']
                
        frames = opt.get('frames',slice(None))
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            data=h5_file[self.baseline_shift_path][frames]#frame,train_id
        return data
    def load_frame_mask(self,name, **opt):
        run = opt['path_modifiers']['run']
        data_mode = opt['path_modifiers']['data_mode']
                
        frames = opt.get('frames',slice(None))
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            data=h5_file[self.frame_mask_path][frames]#frame,train_id
        return data
    
    def load_custom_mask(self,name,**opt):
        mask_dict = super().load(name,skip_custom_methods=True,path_modifiers={'name':opt['file_name']})
        return mask_dict[opt['key']].astype(bool)
    
    def load_geometry(self,name,**kwargs):
        modules = range(16)
        geom_path = super().get_path('geometry')
        file_type = os.path.splitext(geom_path)[1] 
        #log.info('geom_cryst_path = {}'.format(file_type))
        if os.path.isfile(geom_path):
            if file_type == '.h5':
                plane_dict = super().load('geom')['modules']
                planes = tuple( plane3D.from_array(plane_dict[str(module)]) for module in modules)
            elif file_type == '.geom':
                #log.info('yay')
                search_phrases=tuple('p{}a0/fs = '.format(module) for module in range(16))
                with open(geom_path, "r") as geom:
                    lines = geom.readlines()
                r_float = '[\-]*[0-9]+\.[0-9]*'
                r_module = '[0-9]+[a-z]'

                data={}
                for id,line in enumerate(lines):
                    if any(tuple(phrase in line for phrase in search_phrases)):
                        module = re.search(r_module,line)[0][:-1]
                        y_direction = np.array(re.findall(r_float,line)+['0.0']).astype(float)
                        x_direction = np.array(re.findall(r_float,lines[id+1])+['0.0']).astype(float)
                        base = np.array(re.findall(r_float,lines[id+2])+re.findall(r_float,lines[id+3])+['0.0']).astype(float)*0.2 #0.2mm pixelwidth | geom file etries are measured in pixel width
                        data[module]=plane3D(base=base,x_direction =x_direction,y_direction = y_direction)

                planes= tuple(data[str(module)] for module in modules)
            else:
                raise IOError('Geometry file type {} not .h5 or .geom'.format(file_type))
        else:
            raise IOError('Geometry file not found at {}'.format(geom_path))
        return planes

    
    ## select specific frame range and split it into chunks  ##
    def _process_frame_range(self,frame_range):
        'if its a string threat it as datapath to some fole that contains a frame range'
        if isinstance(frame_range,str):
            frame_range = self.load(frame_range)
        return frame_range

    def get_frame_ids_from_selection(self,frame_range,selection,good_cells,h5_file):
        n_frames = len(h5_file[self.cell_path])
        frame_mask = np.zeros(n_frames,dtype = bool)
        frame_mask[frame_range] = True 
        
        dsets = {}
        u_dsets = {}
        
        dsets['cells'] = h5_file[self.cell_path][:]
        dsets['trains'] = h5_file[self.train_path][:]
        dsets['pulses'] = h5_file[self.pulse_path][:]


        bad_cell_mask = np.isin(dsets['cells'],good_cells)
        sel_mask = bad_cell_mask
        for key in dsets:
            sel = selection[key]
            sel_mask &= sel.mask(dsets[key])
            
        mask = sel_mask & frame_mask
        #mask[:frame_range.start] = False
        #mask[frame_range.stop:] = False
        frame_ids = np.nonzero(mask)[0]
        n_frames = len(frame_ids)
        return frame_ids,n_frames

    def load_chunks(self, name ,**opt):
        '''
        Gets the frame Ids corresponding to kwargs['selection'] and splits them in chunks according to the available memory.
        '''
        run = opt['run']
        data_mode = opt['data_mode']
        frame_range = self._process_frame_range(opt['frame_range'])
        #log.info('frame_range = {}'.format(frame_range))
        selection = opt['selection']
        modules = opt['modules']
        n_out_frames = opt['n_frames']
        free_mem = opt['free_mem']
        good_cells = opt['good_cells']
        in_multiples_of = opt['in_multiples_of']
        
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:            
            frame_ids,n_frames = self.get_frame_ids_from_selection(frame_range,selection,good_cells,h5_file)
            if not isinstance(n_out_frames,bool):                
                try:
                    assert (n_frames >= n_out_frames), 'Not enough frames beeing selected. Required {} frames got {} frames. Try to continue.'.format(n_out_frames,n_frames)
                except AssertionError as e:
                    log.error(e)
                    n_out_frames = n_frames
                    #log.info('n_frames {} nout frames {}'.format(n_frames,n_out_frames))
                    #raise e
                frame_ids = frame_ids[:n_out_frames]#np.random.choice(frame_ids,size = n_out_frames, replace=False)
                #log.info('n frames after = {}'.format(len(frame_ids)))
                n_frames= n_out_frames            
            #log.info('frame_ids shape ={}'.format(frame_ids.shape))
            dataset = h5_file[self.data_path]
            frame_shape = dataset.shape[1:]
            #log.info('frame shape = {}'.format(frame_shape))
            frame_size_B = np.prod(frame_shape)*len(modules)*np.dtype(float).itemsize
            #log.info('frame size = {} [MBytes]'.format(frame_size_B/1024**2))
        simultanous_frames_in_mem = 1 + free_mem//frame_size_B
        #log.info('simultaneouse frames in mem = {} [MBytes]'.format(simultanous_frames_in_mem))
        #make sure all but the last chunk part are in multiples of 'in_multiples_of'
        #log.info('in multiples of = {}'.format(in_multiples_of))
        if not isinstance(in_multiples_of,bool):
            simultanous_frames_in_mem = simultanous_frames_in_mem//in_multiples_of *in_multiples_of
            #log.info('simultanous_frames_in_mem = {}'.format(simultanous_frames_in_mem))
        #log.info('n_frames = {}'.format(n_frames))
        split_ids = np.concatenate((np.arange(0,n_frames,simultanous_frames_in_mem),[n_frames-1])).astype(int)
        #log.info('split_ids={}'.format(split_ids))
        #if last chunk part is smaller than 'in_multiples_of' combine the last two chunk parts.
        #i.e make sure that each chunk is bigger than in_multiples_of
        if not isinstance(in_multiples_of,bool):
            last_chunk_size = n_frames%simultanous_frames_in_mem
            if last_chunk_size < in_multiples_of:
                split_ids=split_ids[:-1]
                
        log.info("Maximal chunk size is {} GB or {} frames".format(simultanous_frames_in_mem*frame_size_B/1024**3,simultanous_frames_in_mem))

        if len(frame_ids) !=0:
            chunks=np.split(frame_ids,split_ids[1:-1])
        else:
            chunks = []
        #log.info('n frames = {}'.format(len(frame_ids)))
        #log.info('frames in chunks = {}'.format(np.sum([len(chunk) for chunk in chunks])))
        return chunks
            

    ## load data from VDS files 
    def _load_data_chunk_worker(self,modules,slices,module_id_lookup,run,data_slices,output_slices,data_mode,**opt):
        for m,s_id in zip(modules,slices):
            #log.info('mdule {}'.format(m))
            m_id = module_id_lookup[m]
            data,mask,gain = opt['outputs'][:3]
            d_slice = data_slices[s_id]
            o_slice = output_slices[s_id]
            with super().load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':m},as_h5_object = True) as h5_file:                
                vds = h5_file[self.data_path]
                vds_mask = h5_file[self.mask_path]
                vds_gain = h5_file[self.gain_path]                
                #for d_slice,o_slice in zip(data_slices,output_slices):
                #log.info('out_slice shape = {}'.format(o_slice))
                #log.info('data_slice shape = {}'.format(vds[d_slice].shape))
                vds.read_direct(data,d_slice,(m_id,o_slice))
                nan_mask = np.isnan(vds[d_slice])
                if nan_mask.any():
                    log.info('module {}, slice {}, out_slice {}'.format(m,d_slice,o_slice))
                    log.info('{}% nans found '.format(np.sum(nan_mask)/np.prod(nan_mask.shape)*100))
                if data_mode =='proc':
                    vds_mask.read_direct(mask,d_slice,(m_id,o_slice))
                    vds_gain.read_direct(gain,d_slice,(m_id,o_slice))
                # inverting the mask so that unmasked values are 1/True and masked values are 0/False
                mask[m_id,o_slice] = ~mask[m_id,o_slice]
    def _load_data_chunk_worker2(self,module,module_id_lookup,run,data_slices,output_slices,data_mode,**opt):
        m = module
        for d_slice,o_slice in zip(data_slices,output_slices):
            #log.info('mdule {}'.format(m))
            m_id = module_id_lookup[m]
            data,mask,gain = opt['outputs'][:3]
            #d_slice = data_slices[s_id]
            #o_slice = output_slices[s_id]
            with super().load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':m},as_h5_object = True) as h5_file:
                vds = h5_file[self.data_path]
                vds_mask = h5_file[self.mask_path]
                vds_gain = h5_file[self.gain_path]                
                #for d_slice,o_slice in zip(data_slices,output_slices):
                #log.info('out_slice shape = {}'.format(o_slice))
                #log.info('data_slice shape = {}'.format(vds[d_slice].shape))
                vds.read_direct(data,d_slice,(m_id,o_slice))
                nan_mask = np.isnan(vds[d_slice])
                if nan_mask.any():
                    log.info('module {}, slice {}, out_slice {}'.format(m,d_slice,o_slice))
                    log.info('{}% nans found '.format(np.sum(nan_mask)/np.prod(nan_mask.shape)*100))
                if data_mode =='proc':
                    vds_mask.read_direct(mask,d_slice,(m_id,o_slice))
                    vds_gain.read_direct(gain,d_slice,(m_id,o_slice))
                # inverting the mask so that unmasked values are 1/True and masked values are 0/False
                mask[m_id,o_slice] = ~mask[m_id,o_slice]                
    def load_data_chunk(self,name, **opt):
        run = opt['run']
        modules = opt['modules']
        frame_ids = opt['frame_ids']
        data_mode = opt['data_mode']
        n_processes = opt.get('n_processes',False)
        #outputs = opt['outputs']        
        
        n_frames = len(frame_ids)
        with self.load('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':0},as_h5_object = True) as h5_file:
            chunk_shape = (len(modules),n_frames) + h5_file[self.data_path].shape[1:]
        
        frame_slices,out_slices = split_into_simple_slices(frame_ids,return_sliced_args=True)
        data,mask,gain = self.comm_module.request_mp_evaluation(self._load_data_chunk_worker , mp_type='shared_array_multi_new' , out_shapes = [chunk_shape,chunk_shape,chunk_shape], out_dtypes = [np.dtype('float32'),np.dtype('bool'),np.dtype('uint8')],argArrays = [modules,np.arange(len(frame_slices))], const_args = [run,frame_slices,out_slices,data_mode], callWithMultipleArguments = True,splitMode='modulus',n_processes = n_processes)
        data = np.swapaxes(data,0,1)
        if data_mode == 'proc':
            mask = np.swapaxes(mask,0,1)
            gain = np.swapaxes(gain,0,1)
            return data,mask,gain
        else:
            return data


    ## HDF5 VDS Generation ##
    def create_vds(self, run, from_raw_data,modules = np.arange(16), n_processes=False):
        ## There is a multiprocessing bug in the following code right now only works of one module is given as input
        '''
        This function creates assembled vds files for data from the AGIPD detector at the European XFEL.
        It supports to generate sepparate vds per detector module or a combined vds for all modules (this is however slower).
        It automatically filters out trainId==0, doubled train_ids and pulseId or cellId == 65535
        input:
            run int : The Experimental run to generatevds files for
            from_raw_data bool: Whether to generate vds_data for raw or proc files
            create_modulewise_vds bool: Whether or not to create module wise vds files
            create_complete_vds bool: Whether or not to create fully combined vds_files
            n_processes bool or int: If bool it uses the non-hyperthreading threads on your machine (i.e. number of cpu cores). 
        '''
        comm_module = Multiprocessing.comm_module
        if from_raw_data:
            data_mode = 'raw'
        else:
            data_mode = 'proc'
            
        ## get file path
        #paths = tuple( sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%module))for module in range(16) )
        
        paths = [sorted(glob.glob(self.get_path('vds_regexpr',path_modifiers = {'data_mode':data_mode,'run':run,'module':module}))) for module in modules]
        m_ids = np.concatenate([np.full(len(module_paths),m) for m,module_paths in zip(modules,paths)])
        #m_ids = np.concatenate(tuple(np.full(len(file_list),module,dtype = np.int) for module,file_list in enumerate(paths)))        
        #flist = np.array( tuple( path for paths_per_module in paths for path in paths_per_module))
        flist = np.array(tuple(path for module_paths in paths for path in module_paths))
        assert len(flist)>0,'No input files found at {} check file "vds_regexpr" in experimental settings.'.format(self.get_path('run_data',is_file=False,path_modifiers = {'data_mode':data_mode,'run':run}))
        #log.info(tuple(zip(m_ids,flist)))
        ## setup general information about datasets
        log.info('Loading general dataset info.')
        general_data = self.get_general_vds_data(flist,m_ids,from_raw_data,n_processes = n_processes)        
            
        ## Count number of frames and create a way to find the part of frames corresponding to a given file.     
        lookup_array = general_data['train_lookup_array']
        start_frames = [np.sum(lookup_array[:i,0]) for i in range(len(lookup_array)+1)]
        frame_slices = [slice(start_frames[i],start_frames[i+1]) for i in range(len(lookup_array))]        
        n_frames = general_data['n_frames']
        frame_ids = np.arange(n_frames)
        
        log.info('Generating VDS layouts.')
        mp_mode = Multiprocessing.MPMode_Queue(assemble_outputs = False)
        results = comm_module.request_mp_evaluation(self._vds_worker,mp_mode,input_arrays=[flist,m_ids],const_inputs=[frame_slices,frame_ids,lookup_array,general_data,data_mode],split_together = True,n_processes = n_processes)
        #log.info('len(m_ids) = {}'.format(len(m_ids)))
        results = tuple( elem for result in results.values() for elem in result)
        
        result_module_ids = np.array([r['module'] for r in results])
        log.info('Populating VDS layouts and saving VDS files.')
        comm_module.request_mp_evaluation(self._generate_module_wise_vds_files,mp_mode,input_arrays=[modules],const_inputs = [run,results,result_module_ids,frame_slices,general_data,from_raw_data],n_processes = n_processes)
        
    def get_general_vds_data(self,flist,m_ids,module,from_raw_data= False,n_processes =False):
        if from_raw_data:
            data_mode = 'raw'
        else:
            data_mode = 'proc'
        flist = np.asarray(flist)
        def _extract_general_data_worker(file_names,module_ids,**kwargs):
            n_frames=[]
            n_trains = []
            last_train_ids = []
            max_train_ids = []
            first_train_ids = []
            for fname,m_id in zip(file_names,module_ids):
                train_path = self.get_path('h5_agipd_trainId',is_file = False,path_modifiers = {'module':m_id})
                pulse_path = self.get_path('h5_agipd_pulseId',is_file = False,path_modifiers = {'module':m_id})
                data_path = self.get_path('h5_agipd_data',is_file = False,path_modifiers = {'module':m_id})
                with h5.File(fname, 'r') as f:
                    n_frames.append(f[data_path].shape[0])
                    train_ids = f[train_path][:]
                    min_train_id = int(np.min(train_ids[train_ids>0]))
                    unique_pulses,counts = np.unique(f[pulse_path][:],return_counts = True)
                    train_count = int(np.median(counts))
                    n_trains.append(train_count)
                    first_train_ids.append(min_train_id)
                    last_train_ids.append(min_train_id+train_count-1)
                    max_train_ids.append(np.max(train_ids))
                    #log.info(f'module {m_id} n_frames: {n_frames[-1]} file name = {fname}')
            #log.info(f'outputs = { np.array([n_frames,first_train_ids,last_train_ids,max_train_ids,n_trains,module_ids])}')
            return [n_frames,first_train_ids,last_train_ids,max_train_ids,n_trains,module_ids]
        mp_mode = Multiprocessing.MPMode_Queue(assemble_outputs = False)
        results2 = Multiprocessing.comm_module.request_mp_evaluation(_extract_general_data_worker,mp_mode,input_arrays = [flist,m_ids],split_together=True, call_with_multiple_arguments = True,n_processes = n_processes)
        #log.info('{}'.format([np.asarray(a).T.shape for a in results.values()]))
        #log.info('results shape = {}'.format(results.shape))
        results2 = np.concatenate([np.asarray(a).T for a in results2.values()],axis = 0)

        results = Multiprocessing.comm_module.request_mp_evaluation(_extract_general_data_worker,input_arrays = [flist,m_ids],split_together=True, call_with_multiple_arguments = True,n_processes = n_processes)
        results = np.concatenate([np.asarray(a).T for a in results],axis = 0)
        #log.info(f'{[len(a) for a in results2]}, len flist = {len(flist)} len results = {len(results2)} len mids = {len(m_ids)}')

        #Split results by module
        #args_sorted_by_modules = np.argsort(results[:,-1])
        #results = results[args_sorted_by_modules]
        results_per_module=np.array(np.split(results,split_ids_by_unique_values(results[:,-1])))
        
        log.info(f'{np.array(results_per_module)[:,:,0]}')
        #Check whether all modules have same number of frames and trains
        n_frames_per_module = results_per_module[:,:,0].sum(axis = 1)
        n_trains_per_module = results_per_module[:,:,4].sum(axis = 1)
        #nnumber_of_frames_equal = [np.sum(r[:,4]) for r in results_per_module]
        all_modules_have_same_number_of_frames = not ((n_frames_per_module-n_frames_per_module[0]).any())
        all_modules_have_same_number_of_trains = not ((n_trains_per_module-n_trains_per_module[0]).any())                
        #assert all_modules_have_same_number_of_trains,'Not all modules have same number of trains. \n{}'.format(n_trains_per_module)
        try:
            assert all_modules_have_same_number_of_frames,'Not all modules have same number of frames. \n{}'.format(n_frames_per_module)
        except AssertionError as e:
            log.error(e)
            meta_data_equal = (results_per_module[:,:,0] != results_per_module[0,None,:,0])
            differences_in = np.stack(np.nonzero(meta_data_equal),axis=-1) 
            log.info(f'Differences in modules/files {differences_in}')
            raise e
        

        #From now on consider only results of 0 module and remove the module id data 
        results = results_per_module[0][:,:-1]
        
        #log.info('results shape = {}'.format(results.shape))
        train_lookup_array = results[np.argsort(results[:,2])][:,:3].astype(int)
        #log.info('n_frames = {}'.format(train_lookup_array[:,0]))
        #log.info('lookup_array = {}'.format(train_lookup_array))
        n_frames = int(np.sum(results[:,0]))
        n_trains = int(np.sum(results[:,-1]))
        first_train = int(np.min(results[:,1]))
        last_train = int(first_train + n_trains -1)
        max_train_id = int(np.max(results[:,3]))
        
        out_dict = {'n_frames': n_frames,
                    'n_trains': n_trains,
                    'first_train_id':first_train,
                    'last_train_id':last_train,
                    'max_train_id':max_train_id,
                    'train_lookup_array':train_lookup_array
                    }
        #log.info('first train = {}, last train = {} ,n_trains = {}, max_train = {}'.format(first_train,last_train,n_trains,max_train_id))
        data_path = self.get_path('h5_agipd_data',is_file = False,path_modifiers = {'module':m_ids[0]})
        cell_path = self.get_path('h5_agipd_cellId',is_file = False,path_modifiers = {'module':m_ids[0]})
        train_path = self.get_path('h5_agipd_trainId',is_file = False,path_modifiers = {'module':m_ids[0]})
        pulse_path = self.get_path('h5_agipd_pulseId',is_file = False,path_modifiers = {'module':m_ids[0]})
        with h5.File(flist[0], 'r') as f:
            #log.info(f['INSTRUMENT/'+det_name +'/DET/'].keys())
            data = f[data_path]                
            data_shape = data.shape
            pulses = f[pulse_path][:]
            cells = f[cell_path]
            trains = f[train_path]
            uniq_pulses = np.unique(pulses)
            out_dict['data_shape'] = (n_frames,)+data_shape[1:]
            out_dict['uniq_pulses'] = uniq_pulses
            out_dict['n_pulses']=len(uniq_pulses)
            out_dict['data_dtype']=data.dtype
            out_dict['pulse_dtype']=pulses.dtype
            out_dict['cell_dtype']=cells.dtype
            out_dict['train_dtype']=trains.dtype
            if not from_raw_data:
                mask_path = self.get_path('h5_agipd_mask',is_file = False,path_modifiers = {'module':m_ids[0]})
                gain_path = self.get_path('h5_agipd_gain',is_file = False,path_modifiers = {'module':m_ids[0]})
                blShift_path = self.get_path('h5_agipd_baseline_shift',is_file = False,path_modifiers = {'module':m_ids[0]})
                #log.info(f['INSTRUMENT/'+det_name +'/DET/{}CH0:xtdf/image/'.format(module)].keys())
                gain = f[gain_path]
                mask = f[mask_path]
                baseline_shift = f[blShift_path]
                out_dict['mask_dtype'] = mask.dtype
                out_dict['gain_dtype'] = gain.dtype
                out_dict['basline_shift_dtype'] = baseline_shift.dtype
        return out_dict 
        
    def _get_global_slice_id(self,train_id,lookup_array):
        slice_id = np.argmin(lookup_array[:,1]<=train_id)-1
        #log.info(slice_id)
        #log.info(frame_slices[slice_id])
        if (train_id>lookup_array[-1,2]) or (train_id<lookup_array[0,1]):
            raise AssertionError('train_id {} outside of known train id values [{},{}]'.format(train_id,lookup_array[0,1],lookup_array[-1,2]))
        else:
            return slice_id
    def _vds_worker(self,file_name,module,frame_slices,frame_ids,slice_lookup_array,general_data,data_mode,**kwargs):
        data_path = self.get_path('h5_agipd_data',is_file = False,path_modifiers = {'module':module})
        train_path = self.get_path('h5_agipd_trainId',is_file = False,path_modifiers = {'module':module})
        pulse_path = self.get_path('h5_agipd_pulseId',is_file = False,path_modifiers = {'module':module})
        cell_path = self.get_path('h5_agipd_cellId',is_file = False,path_modifiers = {'module':module})
        with h5.File(file_name, 'r') as f:
            #log.info(f[train_path][:])
            # Load 1D datasets of traint/pulse/cell ids which will be combined and stored directly without virtualization
            # Annoyingly, raw data has an extra dimension for the IDs
            # (which is why we need the ravel)
            train_ids = f[train_path][:].ravel() # equivalent to  f[train_path][:].reshape(-1)
            cell_ids = f[cell_path][:].ravel() # equivalent to  f[cell_path][:].reshape(-1)
            pulse_ids = f[pulse_path][:].ravel() # equivalent to  f[pulse_path][:].reshape(-1)
            part_id = self._get_global_slice_id(train_ids[0],slice_lookup_array)
            #log.info('module = {} train_id = {} part_id ={}'.format(module,train_ids[0],part_id))
            part_slice = frame_slices[part_id]
            frame_ids_part = frame_ids[part_slice]
                
            # Generate frame mask by excluding bad trains cells and pulses:                
            # Remove the following bad data:
            #   Train ID = 0, suggesting no input from AGIPD
            #   Train ID out of range, for bit flips from the trainID server
            #   Repeated train IDs: Keep only first train with that ID
            first_global_train_id = general_data['first_train_id']
            last_global_train_id = general_data['last_train_id']
            n_pulses =  general_data['n_pulses']
            train_mask = (train_ids>0) & (train_ids>=first_global_train_id) & (train_ids<=last_global_train_id)
            uniq_trains, uniq_trains_counts = np.unique(train_ids, return_counts=True, return_index=True)[1:]
            for i in uniq_trains[uniq_trains_counts > n_pulses]:
                log.warning('WARNING: Repeated train IDs in {} from ind {}'.format((op.basename(file_name), i)))
                train_mask[np.nonzero(train_ids==train_ids[i])[0][n_uniq_pulses:]] = False                 
            local_frame_mask = train_mask

            if (~local_frame_mask).any():
                log.info('{} of {} trains are masked.'.format(np.sum(~local_frame_mask),uniq_trains.shape))
                log.info("first train = {} last train = {} train_ids:\n {} ".format(first_global_train_id,last_global_train_id,train_ids))
            
            # Setting masked Ids to default values
            # pulse and cellIds are stored as uint64 dtype while the cell ids are stored as uint16
            # We set the default value to be the maximal possible integer for the given datatypes which are:
            # 2**64-1 for uint64 and 2**16-1 for uint16
            bad_data_mask = ~local_frame_mask
            max_uint64 = int(2**64-1)
            max_uint16 = int(2**16-1)
            train_ids[bad_data_mask]=max_uint64
            pulse_ids[bad_data_mask]=max_uint64
            cell_ids[bad_data_mask]=max_uint16
            #good_local_frame_ids = np.arange(len(train_ids))[local_frame_mask]
            #good_global_frame_ids = frame_ids_part[local_frame_mask]
            #n_good_frames = len(good_global_frame_ids)
            
            # Initialize results
            result = {
                'module':module,
                'cells':cell_ids,
                'pulses':pulse_ids,
                'trains':train_ids,
                'frame_ids':frame_ids_part,
                'local_frame_mask':local_frame_mask,
                'part_id': part_id,
            #    'n_good_frames':n_good_frames
            }
            
            # Load further 1D datasets that will be stored directly
            # baseline shift: Set masked valus to np.nan snce the data is of type float32
            if data_mode == 'proc':
                blShift_path = self.get_path('h5_agipd_baseline_shift',is_file = False,path_modifiers = {'module':module})
                baseline_shift = f[blShift_path][:]
                baseline_shift[bad_data_mask]=np.nan
                result['baseline_shift']=baseline_shift
                
            # Create VDS virtualsource data for 2d virtual datasets 
            data = f[data_path]
            vsource_data = h5.VirtualSource(data)
            result['v_source'] = vsource_data
            if data_mode == 'proc':
                mask_path = self.get_path('h5_agipd_mask',is_file = False,path_modifiers = {'module':module})
                gain_path = self.get_path('h5_agipd_gain',is_file = False,path_modifiers = {'module':module})             
                mask = f[mask_path]
                vsource_mask = h5.VirtualSource(mask)                    
                gain = f[gain_path]
                vsource_gain = h5.VirtualSource(gain)                
                result['v_source_mask'] = vsource_mask
                result['v_source_gain'] = vsource_gain
            return result
    def _generate_module_wise_vds_files(self,module_id,run,results,module_ids,frame_slices,general_data,from_raw_data,**args):
        if from_raw_data:
            data_mode = 'raw'
        else:
            data_mode = 'proc'
        result_ids = np.nonzero(module_ids==module_id)[0]
        #log.info('module_id = {}'.format(module_id))
        data_shape = general_data['data_shape']
        data_dtype = general_data['data_dtype']
        n_frames = data_shape[0]
        shape_1d = (n_frames,)
        layouts = {
            'data':h5.VirtualLayout(shape=data_shape, dtype = general_data['data_dtype']),
            # the following virtualLayouts are only used if data_mode = 'proc' in which case 'mask_dtype' and 'gain_dtype' are contained in general_data
            # I'm generating the following vds layouts also in case of data_mode = 'raw' with mock datatypes to eliminate if statements in this routine
            'mask':h5.VirtualLayout(shape=data_shape, dtype = general_data.get('mask_dtype',data_dtype)),
            'gain':h5.VirtualLayout(shape=data_shape, dtype = general_data.get('gain_dtype',data_dtype))
        }
        #log.info('data layout shape = {}'.format(layouts['data'].shape))
        cell_ids = np.zeros(shape_1d,dtype = general_data['cell_dtype'])
        pulse_ids = np.zeros(shape_1d,dtype= general_data['pulse_dtype'])
        train_ids = np.zeros(shape_1d,dtype= general_data['train_dtype'])
        frame_mask = np.zeros(shape_1d,dtype= bool)
            
        # Same comment as for the above as for gain and mask virtual layouts
        baseline_shifts = np.zeros(shape_1d,dtype = general_data.get('baseline_shift_dtype',data_dtype))
                             
        frame_mask = np.zeros(shape_1d,dtype= bool)
        for r_id in result_ids:
            result = results[r_id]
            part_id = result['part_id']
            frame_slice = frame_slices[part_id]
            #log.info('module {} partid = {}'.format(module_id,part_id))
            self._populate_vds(result,layouts,frame_slice,from_raw_data,index_module = False)
            cell_ids[frame_slice] = result['cells']
            pulse_ids[frame_slice] = result['pulses']
            train_ids[frame_slice] = result['trains']
            frame_mask[frame_slice] = result['local_frame_mask']
            baseline_shifts[frame_slice] = result.get('baseline_shift',0)
        
        vds_file_name = self.get_path('vds',path_modifiers={'run':run,'data_mode':data_mode,'module':module_id})
        #log.info('vds_file name = {}'.format(vds_file_name))
        self._save_vds_file(layouts,cell_ids,pulse_ids,train_ids,frame_mask,baseline_shifts,vds_file_name,from_raw_data)
        #log.info('done')            
        return None
    def _populate_vds(self,result,layouts,frame_slice,from_raw_data,index_module=True):
        module = result['module']
        vsource_data = result['v_source']
        layout_data = layouts['data']
        #log.info('layout_shape = {} vsource shape ={} slice = {} len slice = {}'.format(layout_data.shape,vsource_data.shape,frame_slice,frame_slice.stop-frame_slice.start))
        #log.info('slice in populate_vds ={}'.format(frame_slice))
        #log.info('max_local frame_id ={}'.format(local_frame_ids.max()))
        #log.info('layout  shape {}'.format(layout_data.shape))
        #log.info('vsource shape {}'.format(vsource_data.shape))             
        if index_module:
            layout_data[module,frame_slice] = vsource_data
        else:
            #log.info('vsource_shape = {}'.format(vsource_data.shape))
            #log.info('vsource_shape = {}'.format(vsource_data.shape))
            #log.info('layout_shape = {}'.format(layout_data[frame_slice].shape))
            #layout_data[frame_slice] = vsource_data[combined_good_frames_part]
            layout_data[frame_slice] = vsource_data
        if not from_raw_data:
            layout_mask = layouts['mask']
            vsource_mask = result['v_source_mask']                            
            layout_gain = layouts['gain']
            vsource_gain = result['v_source_gain']
            #log.info('layout gain shape {}'.format(layout_gain.shape))
            #log.info('vsource gain shape {}'.format(vsource_gain.shape))
            if index_module:
                layout_mask[module,frame_slice] = vsource_mask
                layout_gain[module,frame_slice] = vsource_gain
            else:
                #log.info('len frame_ids={}'.format(len(frame_ids)))
                layout_mask[frame_slice] = vsource_mask
                layout_gain[frame_slice] = vsource_gain
                
    def _save_vds_file(self,layouts,cell_ids,pulse_ids,train_ids,frame_mask,baseline_shifts,file_name,from_raw_data):
        data_path = self.get_path('vds_agipd_data',is_file = False)
        train_path = self.get_path('vds_agipd_trainId',is_file = False)
        pulse_path = self.get_path('vds_agipd_pulseId',is_file = False)
        cell_path = self.get_path('vds_agipd_cellId',is_file = False)
        
        frame_mask_path = self.get_path('vds_agipd_frame_mask',is_file = False)
        self.create_path_if_nonexistent(file_name)
        outf = h5.File(file_name, 'w', libver='latest')
        outf.create_dataset(train_path,data = train_ids) 
        outf.create_dataset(cell_path,data = cell_ids)
        outf.create_dataset(pulse_path,data = pulse_ids)
        outf.create_dataset(frame_mask_path,data = frame_mask)
        outf.create_virtual_dataset(data_path, layouts['data'], fillvalue=np.nan)
        if not from_raw_data:
            mask_path = self.get_path('vds_agipd_mask',is_file = False)
            gain_path = self.get_path('vds_agipd_gain',is_file = False)
            baseline_shift_path = self.get_path('vds_agipd_baseline_shift',is_file = False)
            outf.create_virtual_dataset(mask_path, layouts['mask'], fillvalue=np.nan)
            outf.create_virtual_dataset(gain_path, layouts['gain'], fillvalue=np.nan)
            outf.create_dataset(baseline_shift_path,data = baseline_shifts)
        outf.close()
        

    def create_vds_module_old(self, run, from_raw_data,modules = np.arange(16), create_modulewise_vds = True, create_complete_vds = False, n_processes=False):
        ## There is a multiprocessing bug in the following code right now only works of one module is given as input
        '''
        This function creates assembled vds files for data from the AGIPD detector at the European XFEL.
        It supports to generate sepparate vds per detector module or a combined vds for all modules (this is however slower).
        It automatically filters out trainId==0, doubled train_ids and pulseId or cellId == 65535
        input:
            run int : The Experimental run to generatevds files for
            from_raw_data bool: Whether to generate vds_data for raw or proc files
            create_modulewise_vds bool: Whether or not to create module wise vds files
            create_complete_vds bool: Whether or not to create fully combined vds_files
            n_processes bool or int: If bool it uses the non-hyperthreading threads on your machine (i.e. number of cpu cores). 
        '''
        comm_module = Multiprocessing.comm_module
        if from_raw_data:
            data_mode = 'raw'
            folder = self.get_path('raw_data_base',path_modifiers={'run':run},is_file = False)
        else:
            data_mode = 'proc'
            folder = self.get_path('proc_data_base',path_modifiers={'run':run},is_file = False)
            
        ## get file path
        #paths = tuple( sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%module))for module in range(16) )
        paths = [sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%module)) for module in modules]
        m_ids = np.concatenate([np.full(len(module_paths),m) for m,module_paths in zip(modules,paths)])
        #m_ids = np.concatenate(tuple(np.full(len(file_list),module,dtype = np.int) for module,file_list in enumerate(paths)))        
        #flist = np.array( tuple( path for paths_per_module in paths for path in paths_per_module))
        flist = np.array(tuple(path for module_paths in paths for path in module_paths))
        #log.info(tuple(zip(m_ids,flist)))
        ## setup general information about datasets
        general_data = self.get_general_vds_data(flist[0],modules[0],from_raw_data)
        detector_name = general_data['detector_name']
        data_shape = general_data['data_shape']
        uniq_pulses = general_data['uniq_pulses']
        n_uniq_pulses = len(uniq_pulses)
        data_dtype = general_data['data_dtype']
        if not from_raw_data:
            mask_dtype = general_data['mask_dtype']
            gain_dtype = general_data['gain_dtype']
        cell_dtype = general_data['cell_dtype']
        pulse_dtype = general_data['pulse_dtype']
        train_dtype = general_data['train_dtype']
            
        ## Count number of frames and create a way to find the part of frames corresponding to a given file.     
        lookup_array = self.generate_frames_lookup_array(folder,n_processes = n_processes)
        last_train = lookup_array[-1,2]
        start_frames = [np.sum(lookup_array[:i,0]) for i in range(len(lookup_array)+1)]
        frame_slices = [slice(start_frames[i],start_frames[i+1]) for i in range(len(lookup_array))]
        def get_slice_id(train_id):
            slice_id = np.argmin(lookup_array[:,1]<=train_id)-1
            #log.info(slice_id)
            #log.info(frame_slices[slice_id])
            if (train_id>lookup_array[-1,2]) or (train_id<lookup_array[0,1]):
                raise AssertionError('train_id {} outside of known train id values [{},{}]'.format(train_id,lookup_array[0,1],lookup_array[-1,2]))
            return slice_id
        
        n_frames = np.sum(lookup_array[:,0])
        frame_ids = np.arange(n_frames)
        
        def vds_worker(file_name,module,**kwargs):
            dset_prefix = 'INSTRUMENT/'+detector_name+'/DET/%dCH0:xtdf/image/'%module
            with h5.File(file_name, 'r') as f:
                # Annoyingly, raw data has an extra dimension for the IDs
                # (which is why we need the ravel)
                train_ids = f[dset_prefix+'trainId'][:].ravel() # equivalent to  f[dset_prefix+'trainId'][:].reshape(-1)
                cell_ids = f[dset_prefix+'cellId'][:].ravel() # equivalent to  f[dset_prefix+'trainId'][:].reshape(-1)
                pulse_ids = f[dset_prefix+'pulseId'][:].ravel() # equivalent to  f[dset_prefix+'trainId'][:].reshape(-1)
                part_id = get_slice_id(train_ids[0])
                part_slice = frame_slices[part_id]
                frame_ids_part = frame_ids[part_slice]
                
                # Generate frame mask by excluding bad trains cells and pulses:                
                # Remove the following bad data:
                #   Train ID = 0, suggesting no input from AGIPD
                #   Train ID out of range, for bit flips from the trainID server
                #   Repeated train IDs: Keep only first train with that ID
                train_mask = (train_ids>0) & (train_ids<last_train) # gives true only for trIDs in the given range
                uniq_trains, uniq_trains_counts = np.unique(train_ids, return_counts=True, return_index=True)[1:]
                for i in uniq_trains[uniq_trains_counts > n_uniq_pulses]:
                    log.warning('WARNING: Repeated train IDs in {} from ind {}'.format((op.basename(file_name), i)))
                    train_mask[np.nonzero(train_ids==train_ids[i])[0][n_uniq_pulses:]] = False                 
                local_frame_mask = train_mask 
                good_local_frame_ids = np.arange(len(train_ids))[local_frame_mask]
                good_global_frame_ids = frame_ids_part[local_frame_mask]
                n_good_frames = len(good_global_frame_ids)
                
                # Initialize results
                result = {
                    'module':module,
                    'cells':cell_ids[local_frame_mask],
                    'pulses':pulse_ids[local_frame_mask],
                    'trains':train_ids[local_frame_mask],
                    'frame_ids':good_global_frame_ids,
                    'local_frame_mask':local_frame_mask,
                    'part_id': part_id,
                    'n_good_frames':n_good_frames
                }
                
                # Create VDS virtualsource data 
                data = f[dset_prefix+'data']
                vsource_data = h5.VirtualSource(data)
                result['v_source'] = vsource_data
                if not from_raw_data:
                    mask = f[dset_prefix+'mask']
                    vsource_mask = h5.VirtualSource(mask)                    
                    gain = f[dset_prefix+'gain']
                    vsource_gain = h5.VirtualSource(gain)                
                    result['v_source_mask'] = vsource_mask
                    result['v_source_gain'] = vsource_gain
                #if (data[0,0,0]<16.6) and (data[0,0,0]>16.5):
                #    log.info('found!!!!! data = {}'.format(data[0,0,0]))
                #    log.info('part_slice = {}'.format(part_slice))
                #    log.info('good_local_frame_ids = {}'.format(good_local_frame_ids[:10]))
                #    log.info('module = {}'.format(module))
                #    log.info('vsource_data ={}'.format(vsource_data[0,0,0]))
            return result
        def save_vds_file(layouts,cell_ids,pulse_ids,train_ids,n_good_frames,file_name):
            self.create_path_if_nonexistent(file_name)
            outf = h5.File(file_name, 'w', libver='latest')
            outf['INSTRUMENT/'+detector_name+'/DET/image/trainId'] = train_ids # create a dataset with all train numbers and write it to the h5file (considering pulse structure of the trains)
            # create datasets for cellid and pulseid and fill them with a number 65535
            default_value=65535
            outdset_cid = outf.create_dataset('INSTRUMENT/'+detector_name+'/DET/image/cellId',
                                              shape=(n_good_frames,), dtype='u2',
                                              data=np.full(n_good_frames,default_value, dtype=cell_dtype))
            outdset_cid[:]=cell_ids
            outdset_pid = outf.create_dataset('INSTRUMENT/'+detector_name+'/DET/image/pulseId',
                                              shape=(n_good_frames,), dtype='u8',
                                              data=np.full(n_good_frames,default_value, dtype=pulse_dtype))
            outdset_pid[:]=pulse_ids
            #assert (cell_ids != default_value).any(), 'There seem to be unset cell ids. Check vds_creation algorithm.'
            #assert (pulse_ids != default_value).any(), 'There seem to be unset pulse ids. Check vds_creation algorithm'
            outf.create_virtual_dataset('INSTRUMENT/'+detector_name+'/DET/image/data', layouts['data'], fillvalue=np.nan)
            if not from_raw_data:
                outf.create_virtual_dataset('INSTRUMENT/'+detector_name+'/DET/image/mask', layouts['mask'], fillvalue=np.nan)
                outf.create_virtual_dataset('INSTRUMENT/'+detector_name+'/DET/image/gain', layouts['gain'], fillvalue=np.nan)                
            outf.close()
        def generate_module_wise_vds_files(module_id,results,module_ids,n_good_frames,local_good_frame_ids_by_part,good_frame_slices,**args):
            result_ids = np.nonzero(module_ids==module_id)[0]
            #log.info('module_id = {}'.format(module_id))
            layouts = {
                'data':h5.VirtualLayout(shape=(n_good_frames,) + data_shape[1:], dtype = data_dtype),
                'mask':h5.VirtualLayout(shape=(n_good_frames,) + data_shape[1:], dtype = mask_dtype),
                'gain':h5.VirtualLayout(shape=(n_good_frames,) + data_shape[1:], dtype = gain_dtype)
            }
            #log.info('data layout shape = {}'.format(layouts['data'].shape))
            cell_ids = np.zeros((n_good_frames,),dtype= cell_dtype)
            pulse_ids = np.zeros((n_good_frames,),dtype= pulse_dtype)
            train_ids = np.zeros((n_good_frames,),dtype= train_dtype)
            for r_id in result_ids:
                result = results[r_id]
                part_id = result['part_id']
                good_frame_slice = good_frame_slices[part_id]
                local_good_frame_ids = local_good_frame_ids_by_part[part_id]
                populate_vds(result,layouts,local_good_frame_ids,good_frame_slice,index_module = False)
                cell_ids[good_frame_slice] = result['cells']
                pulse_ids[good_frame_slice] = result['pulses']
                train_ids[good_frame_slice] = result['trains']
            
            vds_file_name = self.get_path('vds_module',path_modifiers={'run':run,'data_mode':data_mode,'module':module_id})
            #log.info('vds_file name = {}'.format(vds_file_name))
            save_vds_file(layouts,cell_ids,pulse_ids,train_ids,n_good_frames,vds_file_name)
            #log.info('done')            
            return None
        
        def populate_vds(result,layouts,local_good_frame_ids,good_frame_slice,index_module=True):
            module = result['module']
            part_id = result['part_id']
            frame_slice = frame_slices[part_id]
            vsource_data = result['v_source']
            layout_data = layouts['data']
            #log.info('slice in populate_vds ={}'.format(good_frame_slice))
            #log.info('max_local frame_id ={}'.format(local_good_frame_ids.max()))
            #log.info('layout  shape {}'.format(layout_data.shape))
            #log.info('vsource shape {}'.format(vsource_data.shape))             
            if index_module:
                layout_data[module,good_frame_slice] = vsource_data[local_good_frame_ids]
            else:
                #log.info('vsource_shape = {}'.format(vsource_data.shape))
                #log.info('vsource_shape = {}'.format(vsource_data.shape))
                #log.info('layout_shape = {}'.format(layout_data[good_frame_slice].shape))
                #layout_data[good_frame_slice] = vsource_data[combined_good_frames_part]
                layout_data[good_frame_slice] = vsource_data[local_good_frame_ids]
            if not from_raw_data:
                layout_mask = layouts['mask']
                vsource_mask = result['v_source_mask']                            
                layout_gain = layouts['gain']
                vsource_gain = result['v_source_gain']
                #log.info('layout gain shape {}'.format(layout_gain.shape))
                #log.info('vsource gain shape {}'.format(vsource_gain.shape))
                if index_module:
                    layout_mask[module,good_frame_slice] = vsource_mask[local_good_frame_ids]
                    layout_gain[module,good_frame_slice] = vsource_gain[local_good_frame_ids]
                else:
                    #log.info('len frame_ids={}'.format(len(frame_ids)))
                    layout_mask[good_frame_slice] = vsource_mask[local_good_frame_ids]
                    layout_gain[good_frame_slice] = vsource_gain[local_good_frame_ids]

        
        n_files = len(flist)
        stime = time.time()
        
        #log.info('creating_vds in multiprocessing mode')
        results = comm_module.request_mp_evaluation(vds_worker,argArrays=[flist,m_ids],split_together = True,assemble_outputs = False,n_processes = n_processes)
        #log.info('len(m_ids) = {}'.format(len(m_ids)))
        results = tuple( elem for result in results.values() for elem in result)

        # Calculate frames that are not mask for all modules and use them to generate the vds File
        combined_frame_mask = np.full(n_frames,True)
        for r in results:
            combined_frame_mask[frame_slices[r['part_id']]] &= r['local_frame_mask']
        local_good_frame_ids_by_part = []
        for s in frame_slices:
            local_n_frames = s.stop-s.start
            local_frame_ids = np.arange(local_n_frames)
            local_good_frame_ids_by_part.append(local_frame_ids[combined_frame_mask[s]])
        
        # Calculate slices in combined_good_frames that correspond to those in frame_slices
        n_good_frames_per_slice = [np.sum(combined_frame_mask[s]) for s in frame_slices]
        #log.info('n_good_frames_per_slice ={}'.format(n_good_frames_per_slice))
        temp = [np.sum(n_good_frames_per_slice[:i]) for i in range(len(frame_slices)+1)]
        #log.info('temp ={}'.format(temp))
        good_frame_slices = [slice(int(temp[i]),int(temp[i+1])) for i in range(len(frame_slices))]            
        #log.info('good_frame_slices ={}'.format(good_frame_slices))
        n_good_frames = np.sum(n_good_frames_per_slice)
        log.info('From {} inital frames {} remain after masking, that is {}%'.format(n_frames,n_good_frames,n_good_frames/n_frames*100))
        
        if create_modulewise_vds:
            #layouts_per_module = comm_module.request_mp_evaluation(generate_module_wise_vds_files,argArrays=[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])],const_args = [results,m_ids,n_good_frames,local_good_frame_ids_by_part,good_frame_slices] ,assemble_outputs = False,n_processes = n_processes)
            layouts_per_module = comm_module.request_mp_evaluation(generate_module_wise_vds_files,argArrays=[modules],const_args = [results,m_ids,n_good_frames,local_good_frame_ids_by_part,good_frame_slices] ,assemble_outputs = False,n_processes = n_processes)
        if create_complete_vds:
            raise NotImplementedError('omplete vds not implemented yet.')


    ## misc saving routines ##
    def save_hit_specifier_vds(self,name,data,**kwargs):
        runs = opt['runs'] #iterable
        modules = opt['modules']
        info = self.load_info({'runs':runs})
        n_frames=tuple(info[str(run)]['n_frames'] for run in runs)
        n_frames_total = np.sum(n_frames)
        l_hit_specifiers = tuple(h5.VirtualLayout(shape=(n_frames_total,3), dtype=float) for module in modules)

        runs_array = np.concatenate(
            tuple(
                np.concatenate(( np.full((n_frames[id],),run)[:,None] , np.arange(n_frames[id])[:,None]),axis=1) for id,run in enumerate(runs)),axis = 0)
        
        current_frame = 0
        for run in runs :
            filename = super().get_path('hit_specifier',path_modifiers={'run':'%04d'%run})
            length = info[str(run)]['n_frames']            
            vsrc_hit_specifiers = tuple( h5.VirtualSource(filename, 'litpixels_{}'.format('%02d'%module), shape=(length,)) for module in modules)
            c=current_frame
            for m_id in range(len(modules)):
                l_hit_specifiers[m_id][c:c+length] = vsrc_hit_specifiers[m_id]
            current_frame += length
            
        value_dict={
            'module_%02d'%module:l_hit_specifiers[m_id] for m_id,module in enumerate(modules)
        }
        value_dict['runs']=runns_array

        super().save('vds_hit_specifier',value_dict)
        
    def save_hits_vds(self,name,data,**kwargs):
        runs = opt['runs'] #iterable
        n_hits=[]
        for run in runs:
            with  h5.File(super().get_path('hits',path_modifiers={'run':run}) ,mode = 'r') as h5_file:
                n_hits.append(h5_file['hits/assembled'].shape[0])

        n_hits_total = np.sum(n_hits)
        l_hit_specifiers = tuple(h5.VirtualLayout(shape=(n_hits_total,1296,1138), dtype=float) for module in modules)

        runs_array = np.concatenate(tuple( np.concatenate((np.full((n_frames[id],),run)[:,None],np.arange(n_frames[id])[:,None]),axis=1) for id,run in enumerate(runs)),axis = 0)
        current_frame = 0
        for run in runs :
            filename = super().get_path('hit_specifier',path_modifiers={'run':'%04d'%run})
            length = info[str(run)]['n_frames']            
            vsrc_hit_specifiers = tuple( h5.VirtualSource(filename, 'litpixels_{}'.format('%02d'%module), shape=(length,)) for module in modules)
            c = current_frame
            for m_id in range(len(modules)):
                l_hit_specifiers[m_id][c:c+length] = vsrc_hit_specifiers[m_id]
            current_frame += length
            
        value_dict={
            'module_%02d'%module:l_hit_specifiers[m_id] for m_id,module in enumerate(modules)
        }
        value_dict['runs']=runns_array

        super().save('vds_hit_specifier',value_dict)
