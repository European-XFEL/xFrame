import sys
import h5py
import numpy as np
import glob
import time
import logging
import traceback
log=logging.getLogger('root')

import ctypes
from psutil import virtual_memory
from scipy import ndimage


from xframe.experiment.interfaces import CalibratorInterface
from xframe.library import pythonLibrary as pyLib
from xframe import database
from xframe import settings

class MockCalibrator(CalibratorInterface):
    def calibration_worker(self,data_generator,out_modifier = False):
        pass
    
class AGIPD_VDS_Calibrator(CalibratorInterface):
    '''
    Interface to get frames interactively
    Initially specify path to folder with raw/proc data
    Then use get_frame(num) to get specific frame
    '''
    def __init__(self):
        #            geom_fname='/gpfs/exfel/exp/SPB/201802/p002145/scratch/geom/b1.geom'):
        #log.info(opt)
        data_mode = settings.experiment.data_mode
        self.good_cells = settings.experiment.good_cells
        opt = settings.experiment.get('calibrator',{})
        self.db=database.experiment
        self.n_modules = 16
        self.verbose = opt.get('verbose',0)
        self.sum_h5cells = len(self.good_cells)
        #        self.geom_fname = opt.get('geom_fname','../geometry/b2.geom')
        self.mask_lg_mg = opt.get('mask_low_and_mid_gain',False)
        self.dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data'
        self.train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/trainId'
        self.pulse_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/pulseId'
        self.cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/cellId'
        if data_mode=='raw':
            self.calib = self.db.load('calibration_constants')
            self.cmode = opt.get('common_mode_correction',False)
            log.info(opt.keys())
            log.info('common mode = {}'.format(self.cmode))
            self.photon_threshold= opt.get('photon_threshold',0.5)
            self.gain_levels=self.calib['DigitalGainLevel']#np.concatenate((([self.calib[module]['DigitalGainLevel'] for module in range(16)],dtype=np.float32)
            self.analog_offset=self.calib['AnalogOffset']#np.array([self.calib[module]['AnalogOffset'] for module in range(16)],dtype=np.float32)
            self.relative_gain=self.calib['RelativeGain']#np.array([self.calib[module]['RelativeGain'] for module in range(16)],dtype=np.float32)
            badpix=self.calib['Badpixel']
            if self.mask_lg_mg:
                badpix[(self.gain_levels[1] | self.gain_levels[2])]=1 # mask all pixels in MG and LG, with some positive value    # notice , all masked values are >0, all good values ==0 in the mask    
            self.bad_pixel=badpix#np.array([self.calib[module]['Badpixel'] for module in range(16)])        
        
        self.frame = np.empty((16,512,128))
        self.frame_shape=(512,128)
        
        self.ids_by_run = {}
        self.cell_ids = None
        self.pulse_ids = None
        self.train_ids = None
        self.nframes = None

        log.info("opt keys ={}".format(opt.keys()))
        self.gain_mode = opt.get('gain_mode','adaptive')
        
        
        # bd.load(calibration)  [h5py.File(f, 'r') for f in sorted(glob.glob(calib_glob))] '/gpfs/exfel/exp/SPB/201802/p002145/scratch/calib/r%.4d/Cheetah*.h5'
        #dir_calib = '/gpfs/exfel/u/scratch/SPB/201802/p002145/kurta/calibData/'
        #calib_str=dir_calib+'Cheetah-AGIPD-calib.h5'
        #mask_str=dir_calib+'bad-pixel-mask.h5'
        
    def set_ids_for_run(self,run):
        ids_by_run = self.ids_by_run
        run_str=str(run)
        if run_str in ids_by_run:
            self.cell_ids = ids_by_run[run_str]['cell_ids']
            self.pulse_ids = ids_by_run[run_str]['pulse_ids']
            self.train_ids = ids_by_run[run_str]['train_ids']
        else:
            load = self.db.load
            pm = {'run':run,'data_mode':'raw'}
            cell_ids = load('cell_ids',path_modifiers = pm)
            pulse_ids = load('pulse_ids',path_modifiers = pm)
            train_ids = load('train_ids',path_modifiers = pm)
            self.ids_by_run[run_str] = {'cell_ids':cell_ids,'pulse_ids':pulse_ids, 'train_ids':train_ids}
            self.cell_ids = cell_ids
            self.pulse_ids = pulse_ids
            self.train_ids = train_ids
        self.nframes=len(self.cell_ids)
            
    def load_calib_constants(self):
        calib_mode = self.calib_mode.lower()
        calib_glob = None
        calib ={}
        if calib_mode == 'cheetah':
            if self.calib_run is None:
                run_str='latest'
            else:
                run_str= 'r%.4d' % calib_run
            f = self.db.load('calibration', opt = {'mode':'cheetah'}, path_modifiers={'run':run_str})
            #(gain,cell,module,many_pixel_ax,few_pixel_ax)
            calib['DigitalGainLevel'] = np.concatenate(tuple(f[module]['DigitalGainLevel'][:][:,:,None,...] for module in range(self.n_modules)),axis = 2,dtype = np.float32)
            calib['AnalogOffset']= np.concatenate(tuple(f[module]['AnalogOffset'][:][:,:,None,...] for module in range(self.n_modules)),axis = 2,dtype = np.float32)
            calib['RelativeGain'] = np.concatenate(tuple(f[module]['RelativeGain'][:][:,:,None,...] for module in range(self.n_modules)),axis = 2,dtype = np.float32)
            calib['Badpixel'] = np.concatenate(tuple(f[module]['Badpixel'][:][:,:,None,...] for module in range(self.n_modules)),axis = 2)
        elif calib_mode == 'alireza2020ver1':
            calib_data = self.db.load('calibration',opt = {'mode':'alireza'})
            data = calib_data['data']
            masks = calib_data['masks']
            npls = self.sum_h5cells
            calib = {}
            npls = 1#data['AnalogOffset'][:]
            calib['AnalogOffset']=data['AnalogOffset'][:].reshape(3,npls,16,512,128) 
            calib['RelativeGain']=data['RelativeGain'][:].reshape(3,npls,16,512,128) 
            gain_modes=np.asarray(data['DigitalGainLevel']).reshape(3,npls,16,512,128)
            calib['DigitalGainLevel'] = gain_modes
            
    
            calib['Badpixel']=np.asarray(masks['badPixelMask']).reshape(3,npls,16,512,128) # mask from dark run Analysis
            print('Mask generated from dark runs analysis will be applied')
            if 'badPixelMask_ByWater' in masks:
                calib['Badpixel']=np.add(calib['Badpixel'], masks['badPixelMask_ByWater'][:].reshape(3,npls,16,512,128)) # mask from Water analysis
                print('Mask generated from water data analysis will be applied')
            if self.mask_lg_mg:
                badpix[(gain_modes[1] | gain_modes[2])]=1 # mask all pixels in MG and LG, with some positive value    # notice , all masked values are >0, all good values ==0 in the mask    
        else:
            print('Error: Calibration Mode {} not known.'.format(calib_mode))
        return calib


    def _calibrate_frames_adaptive_new(self,data_file,frames,module):
        nan_ids=np.isnan(data_file).nonzero()
        if len(nan_ids) != 0:
            #print('nans in datafile to -5')
            data_file[nan_ids]=-5
            
        cmode = self.cmode
        photonThresh = self.photon_threshold
        t_len=len(data_file)
        n_frames=len(frames)
        gains=data_file[:,1]
        raw_data=data_file[:,0]
        cells=self.cell_ids[frames]
        
        
        no_data_mask=(cells==65535)
        raw_data=raw_data[~no_data_mask]
        gains=gains[~no_data_mask]
        cells=cells[~no_data_mask]
        #print('data2 shape={}'.format(data.shape))
        #        print('cells={}'.format(frames))
        gain_modes = self._threshold2(gains, module, cells)
        del(gains)
        offset = np.empty(gain_modes.shape[1:])
        gain = np.empty(gain_modes.shape[1:])
        badpix = np.empty(gain_modes.shape[1:])
        for gain_level,gain_mask in enumerate(gain_modes):
            #print('analog_offset shape={}'.format(self.analog_offset[module,gain_level,cells].shape))
            offset[gain_mask] = self.analog_offset[gain_level,cells,module][gain_mask]
            gain[gain_mask] = self.relative_gain[gain_level,cells,module][gain_mask]
            badpix[gain_mask] = self.bad_pixel[gain_level,cells,module][gain_mask]
            #        print('data dtype= {} offset dtype={} gain dtype={}'.format(data.dtype,offset.dtype,gain.dtype))
        #data_and_mask = np.zeros_like(data_file,dtype = np.float32)
        data = np.zeros_like(data_file[:,0],dtype=np.float32)
        mask = np.full(data_file[:,1].shape,True)
        #data = data_and_mask[:,0]
        #mask = data_and_mask[:,1]
        #mask[:]=1
        data[:] = (np.float32(raw_data) - offset)*gain
        del(raw_data)
        del(data_file)

        #assumes badpix is True for bad pixels and reverses this convention
        bad_pixel_mask = ~(badpix != 0)
        mask[~bad_pixel_mask] = False
        data[~bad_pixel_mask] = 0
        del(bad_pixel_mask)
        del(badpix)
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))
            
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)            
        if isinstance(photonThresh,(list,tuple)):
            photon_threshold_mask = (data < photonThresh[0]*45*gain) | (data>photonThresh[1]*45*gain)
        elif isinstance(photonThresh,(int,float)):
            photon_threshold_mask = (data < photonThresh*45*gain)
                        
        data[photon_threshold_mask] = 0
        #mask[photon_threshold_mask] = 1        
        
        if cmode:
            # Median subtraction by 64x64
            #log.info('applying common mode correction!')
            tmp_data = data.reshape(n_frames,8,64,2,64).transpose(0,2,4,1,3).reshape(n_frames,64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
#            print('new data shape={}'.format(data.shape))
#            print('median shape ={}'.format(np.median(data,axis=(1,2)).shape))
            tmp_data -= np.median(tmp_data, axis=(1,2))[:,None,None,:]
            tmp_data = tmp_data.reshape(n_frames,64,64,8,2).transpose(0,3,1,4,2).reshape(n_frames,512,128)
            data[:]=tmp_data
#        print('data shape={}'.format(data.shape))

        
            #print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        #data[data < photonThresh*45*gain] = 0
        #data[data > 10000] = 10000
        #print('data shape={}'.format(data.shape))
        return data, mask
    def _calibrate_frames_fixed_gain_medium_new(self,data_file,frames,module): 
        nan_ids=np.isnan(data_file).nonzero()
        if len(nan_ids) != 0:
            #print('nans in datafile to -5')
            data_file[nan_ids]=-5
            
        cmode = self.cmode
        photonThresh = self.photon_threshold
        t_len=len(data_file)
        n_frames=len(frames)
        #gains=data_file[:,1]
        raw_data=data_file[:,0]
        cells=self.cell_ids[frames]
        
        
        no_data_mask=(cells==65535)
        raw_data=raw_data[~no_data_mask]
        #gains=gains[~no_data_mask]
        cells=cells[~no_data_mask]
        #print('data2 shape={}'.format(data.shape))
        #        print('cells={}'.format(frames))
        #ain_modes = self._threshold2(gains, module, cells)
        #del(gains)
        offset = self.analog_offset[1,cells,module]
        gain = self.relative_gain[1,cells,module]
        badpix = self.bad_pixel[1,cells,module]

        data = np.zeros_like(data_file[:,0],dtype=np.float32)
        mask = np.full(data_file[:,1].shape,True)
        #data_and_mask = np.zeros_like(data_file,dtype = np.float32)
        #data = data_and_mask[:,0]
        #mask = data_and_mask[:,1]
        #mask[:]=1
        data[:] = (np.float32(raw_data) - offset)*gain
        del(raw_data)
        del(data_file)

        #assumes badpix is True for bad pixels and reverses this convention
        bad_pixel_mask = ~(badpix != 0)
        mask[~bad_pixel_mask] = False
        data[~bad_pixel_mask] = 0
        del(bad_pixel_mask)
        del(badpix)
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))
            
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)            
        if isinstance(photonThresh,(list,tuple)):
            photon_threshold_mask = (data < photonThresh[0]*45*gain) | (data>photonThresh[1]*45*gain)
        elif isinstance(photonThresh,(int,float)):
            photon_threshold_mask = (data < photonThresh*45*gain)
                        
        data[photon_threshold_mask] = 0
        #mask[photon_threshold_mask] = 1        
        
        if cmode:
            # Median subtraction by 64x64
            #log.info('applying common mode correction!')
            tmp_data = data.reshape(n_frames,8,64,2,64).transpose(0,2,4,1,3).reshape(n_frames,64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
#            print('new data shape={}'.format(data.shape))
#            print('median shape ={}'.format(np.median(data,axis=(1,2)).shape))
            tmp_data -= np.median(tmp_data, axis=(1,2))[:,None,None,:]
            tmp_data = tmp_data.reshape(n_frames,64,64,8,2).transpose(0,3,1,4,2).reshape(n_frames,512,128)
            data[:]=tmp_data
#        print('data shape={}'.format(data.shape))

        
            #print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        #data[data < photonThresh*45*gain] = 0
        #data[data > 10000] = 10000
        #print('data shape={}'.format(data.shape))
        return data,mask

    def _calibrate_frames_adaptive_old(self,data_file,frames,module):
        nan_ids=np.isnan(data_file).nonzero()
        if len(nan_ids) != 0:
            #print('nans in datafile to -5')
            data_file[nan_ids]=-5
            
        cmode = self.cmode
        photonThresh = self.photon_threshold
        t_len=len(data_file)
        n_frames=len(frames)
        gains=data_file[:,1]
        raw_data=data_file[:,0]
        cells=self.cell_ids[frames]
        
        
        no_data_mask=(cells==65535)
        raw_data=raw_data[~no_data_mask]
        gains=gains[~no_data_mask]
        cells=cells[~no_data_mask]
        #print('data2 shape={}'.format(data.shape))
        #        print('cells={}'.format(frames))
        gain_modes = self._threshold2(gains, module, cells)
        del(gains)
        offset = np.empty(gain_modes.shape[1:])
        gain = np.empty(gain_modes.shape[1:])
        badpix = np.empty(gain_modes.shape[1:])
        for gain_level,gain_mask in enumerate(gain_modes):
            #print('analog_offset shape={}'.format(self.analog_offset[module,gain_level,cells].shape))
            offset[gain_mask] = self.analog_offset[gain_level,cells,module][gain_mask]
            gain[gain_mask] = self.relative_gain[gain_level,cells,module][gain_mask]
            badpix[gain_mask] = self.bad_pixel[gain_level,cells,module][gain_mask]
            #        print('data dtype= {} offset dtype={} gain dtype={}'.format(data.dtype,offset.dtype,gain.dtype))
        data_and_mask = np.zeros_like(data_file,dtype = np.float32)
        data = data_and_mask[:,0]
        mask = data_and_mask[:,1]
        mask[:]=1
        data[:] = (np.float32(raw_data) - offset)*gain
        del(raw_data)
        del(data_file)

        #assumes badpix is True for bad pixels and reverses this convention
        bad_pixel_mask = ~(badpix != 0)
        mask[~bad_pixel_mask] = 0
        data[~bad_pixel_mask] = 0
        del(bad_pixel_mask)
        del(badpix)
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))
            
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)            
        if isinstance(photonThresh,(list,tuple)):
            photon_threshold_mask = (data < photonThresh[0]*45*gain) | (data>photonThresh[1]*45*gain)
        elif isinstance(photonThresh,(int,float)):
            photon_threshold_mask = (data < photonThresh*45*gain)
                        
        data[photon_threshold_mask] = 0
        #mask[photon_threshold_mask] = 1        
        
        if cmode:
            # Median subtraction by 64x64
            #log.info('applying common mode correction!')
            tmp_data = data.reshape(n_frames,8,64,2,64).transpose(0,2,4,1,3).reshape(n_frames,64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
#            print('new data shape={}'.format(data.shape))
#            print('median shape ={}'.format(np.median(data,axis=(1,2)).shape))
            tmp_data -= np.median(tmp_data, axis=(1,2))[:,None,None,:]
            tmp_data = tmp_data.reshape(n_frames,64,64,8,2).transpose(0,3,1,4,2).reshape(n_frames,512,128)
            data[:]=tmp_data
#        print('data shape={}'.format(data.shape))

        
            #print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        #data[data < photonThresh*45*gain] = 0
        #data[data > 10000] = 10000
        #print('data shape={}'.format(data.shape))
        return data_and_mask
    def _calibrate_frames_fixed_gain_medium_old(self,data_file,frames,module): 
        nan_ids=np.isnan(data_file).nonzero()
        if len(nan_ids) != 0:
            #print('nans in datafile to -5')
            data_file[nan_ids]=-5
            
        cmode = self.cmode
        photonThresh = self.photon_threshold
        t_len=len(data_file)
        n_frames=len(frames)
        #gains=data_file[:,1]
        raw_data=data_file[:,0]
        cells=self.cell_ids[frames]
        
        
        no_data_mask=(cells==65535)
        raw_data=raw_data[~no_data_mask]
        #gains=gains[~no_data_mask]
        cells=cells[~no_data_mask]
        #print('data2 shape={}'.format(data.shape))
        #        print('cells={}'.format(frames))
        #ain_modes = self._threshold2(gains, module, cells)
        del(gains)
        offset = self.analog_offset[1,cells,module]
        gain = self.relative_gain[1,cells,module]
        badpix = self.bad_pixel[1,cells,module]

        data_and_mask = np.zeros_like(data_file,dtype = np.float32)
        data = data_and_mask[:,0]
        mask = data_and_mask[:,1]
        mask[:]=1
        data[:] = (np.float32(raw_data) - offset)*gain
        del(raw_data)
        del(data_file)

        #assumes badpix is True for bad pixels and reverses this convention
        bad_pixel_mask = ~(badpix != 0)
        mask[~bad_pixel_mask] = 0
        data[~bad_pixel_mask] = 0
        del(bad_pixel_mask)
        del(badpix)
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))
            
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)            
        if isinstance(photonThresh,(list,tuple)):
            photon_threshold_mask = (data < photonThresh[0]*45*gain) | (data>photonThresh[1]*45*gain)
        elif isinstance(photonThresh,(int,float)):
            photon_threshold_mask = (data < photonThresh*45*gain)
                        
        data[photon_threshold_mask] = 0
        #mask[photon_threshold_mask] = 1        
        
        if cmode:
            # Median subtraction by 64x64
            #log.info('applying common mode correction!')
            tmp_data = data.reshape(n_frames,8,64,2,64).transpose(0,2,4,1,3).reshape(n_frames,64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
#            print('new data shape={}'.format(data.shape))
#            print('median shape ={}'.format(np.median(data,axis=(1,2)).shape))
            tmp_data -= np.median(tmp_data, axis=(1,2))[:,None,None,:]
            tmp_data = tmp_data.reshape(n_frames,64,64,8,2).transpose(0,3,1,4,2).reshape(n_frames,512,128)
            data[:]=tmp_data
#        print('data shape={}'.format(data.shape))

        
            #print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        #data[data < photonThresh*45*gain] = 0
        #data[data > 10000] = 10000
        #print('data shape={}'.format(data.shape))
        return data_and_mask
    
    def _calibrate_frames_old(self,data_file,frames,module):
        nan_ids=np.isnan(data_file).nonzero()
        if len(nan_ids) != 0:
            #print('nans in datafile to -5')
            data_file[nan_ids]=-5
            
        cmode = self.cmode
        photonThresh = self.photon_threshold
        t_len=len(data_file)
        n_frames=len(frames)
        gains=data_file[:,1]
        data=data_file[:,0]
        cells=self.cell_ids[frames]
        
        
        no_data_mask=(cells==65535)
        data=data[~no_data_mask]
        gains=gains[~no_data_mask]
        cells=cells[~no_data_mask]
        #print('data2 shape={}'.format(data.shape))
        #        print('cells={}'.format(frames))
        gain_modes = self._threshold2(gains, module, cells)
        offset = np.empty(gain_modes.shape[1:])
        gain = np.empty(gain_modes.shape[1:])
        badpix = np.empty(gain_modes.shape[1:])
        for gain_level,gain_mask in enumerate(gain_modes):
            #print('analog_offset shape={}'.format(self.analog_offset[module,gain_level,cells].shape))
            offset[gain_mask] = self.analog_offset[gain_level,cells,module][gain_mask]
            gain[gain_mask] = self.relative_gain[gain_level,cells,module][gain_mask]
            badpix[gain_mask] = self.bad_pixel[gain_level,cells,module][gain_mask]
#        print('data dtype= {} offset dtype={} gain dtype={}'.format(data.dtype,offset.dtype,gain.dtype))
        data = (np.float32(data) - offset)*gain
        data[badpix != 0] = 0        
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))    
        if cmode:
            # Median subtraction by 64x64
            #log.info('applying common mode correction!')
            data = data.reshape(n_frames,8,64,2,64).transpose(0,2,4,1,3).reshape(n_frames,64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
#            print('new data shape={}'.format(data.shape))
#            print('median shape ={}'.format(np.median(data,axis=(1,2)).shape))
            data -= np.median(data, axis=(1,2))[:,None,None,:]
            data = data.reshape(n_frames,64,64,8,2).transpose(0,3,1,4,2).reshape(n_frames,512,128)
#        print('data shape={}'.format(data.shape))
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)

        if isinstance(photonThresh,(list,tuple)):                
            data[(data < photonThresh[0]*45*gain) | (data>photonThresh[1]*45*gain)] = 0
        elif isinstance(photonThresh,(int,float)):
            data[data < photonThresh*45*gain] = 0
        
            #print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        #data[data < photonThresh*45*gain] = 0
        #data[data > 10000] = 10000
        #print('data shape={}'.format(data.shape))
        return data
    
    def calibration_worker(self,chunked_data,out_modifier = False):
        
        out_dict={}
        modules,c_id,chunk,n_chunks,data = chunked_data

        start_time = time.time()
        print('calibrating chunk {}/{} for modules {}]'.format(c_id+1,n_chunks,modules))
        for m_id,module in enumerate(modules):
            calib_data_chunk=self._calibrate_frames(data,chunk[m_id],module)
            if callable(out_modifier):
                out_data_part=out_modifier(calib_data_chunk,m_id,module,c_id)
            else:
                out_data_part=calib_data_chunk

            if not str(m_id) in out_dict:
                out_dict[str(m_id)]={str(c_id):out_data_part}
            else:
                out_dict[str(m_id)][str(c_id)] = out_data_part            
        end_time=time.time()
        print('{} seconds per chunk'.format(end_time-start_time))
        n_modules = len(out_dict)
        out_data_per_module = tuple(np.concatenate(tuple(out_dict[str(m_id)].values()),axis = 0)[:,None] for m_id in range(n_modules))
        out_data = np.concatenate(out_data_per_module,axis = 1)
        return out_data

    def calibrate_new(self,data,frame_ids,modules,out_modifier = False):
        gain_mode= self.gain_mode
        try:
            if gain_mode == 'adaptive':
                calibration_routine = self._calibrate_frames_adaptive_new
            elif gain_mode == 'fixed_gain_medium':
                calibration_routine = self._calibrate_frames_fixed_gain_medium_new
            else:
                e = AssertionError('gain mode {} not known'.format(gain_mode))
                raise(e)
        except AssertionError as e:
            log.error(e)
            traceback.print_exc()
            
        out_dict={}
        for m_id,module in enumerate(modules):
            out_dict[str(m_id)]= calibration_routine(data[m_id],frame_ids,module)
        n_modules = len(out_dict)
        out_data_sets_per_module = zip(*tuple([ out_dict[str(m_id)][0][:,None],out_dict[str(m_id)][1][:,None] ] for m_id in range(n_modules)))
        out_data_sets = [np.concatenate(dataset,axis = 1) for dataset in out_data_sets_per_module]
        if callable(out_modifier):
            out_data_sets = out_modifier(out_data_sets,frame_ids,modules)
        return out_data_sets

    def calibrate_old(self,data,frame_ids,modules,out_modifier = False):
        gain_mode= self.gain_mode
        try:
            if gain_mode == 'adaptive':
                calibration_routine = self._calibrate_frames_adaptive_old
            elif gain_mode == 'fixed_gain_medium':
                calibration_routine = self._calibrate_frames_fixed_gain_medium_old
            else:
                e = AssertionError('gain mode {} not known'.format(gain_mode))
                raise(e)
        except AssertionError as e:
            log.error(e)
            traceback.print_exc()
            
        out_dict={}
        for m_id,module in enumerate(modules):
            out_dict[str(m_id)]= calibration_routine(data[m_id],frame_ids,module)
        n_modules = len(out_dict)
        out_data_per_module = tuple(out_dict[str(m_id)][:,None] for m_id in range(n_modules))
        out_data = np.concatenate(out_data_per_module,axis = 1)
        if callable(out_modifier):
            out_data = out_modifier(out_data,frame_ids,modules)
        return out_data

    def _threshold2(self, gains,module,cells):
        thresholds=self.gain_levels[:,cells,module]
        low_gains = gains > thresholds[2,:]
        high_gains = gains < thresholds[1,:]
        medium_gains= (~low_gains)*(~high_gains)
#        print('447,126, low gain={} medium gain {} high gain {}'.format(low_gains[:,447,126],medium_gains[:,447,126],high_gains[:,447,126]))
        return np.concatenate((high_gains[None,...],medium_gains[None,...],low_gains[None,...]),axis=0)

    


