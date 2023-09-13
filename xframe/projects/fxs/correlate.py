import time
import logging
import sys
import os
import math
import traceback
import numpy as np
import scipy as sp
import struct
from itertools import repeat
file_path = os.path.realpath(__file__)
plugin_dir = os.path.dirname(file_path)
os.chdir(plugin_dir)

from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
from xframe import database
from xframe.library.pythonLibrary import xprint
from xframe.library.physicsLibrary import scattering_angle_to_reciprocal_radii
from xframe import Multiprocessing
from .projectLibrary.cross_correlation import ccf_analysis
Pi = np.pi
log=logging.getLogger('root')

def profile_modifier(func):
    def outer(*args,**kwargs):
        profile = settings.project.profile.use
        process_id=settings.project.profile.process_id
        if profile:
            if Multiprocessing.get_process_name() == process_id:
                out_folder=database.project.get_path('out_base',is_file=False)
                #log.info(f'out_folder = {out_folder}')
                database.project.create_path_if_nonexistent(out_folder)
                path = out_folder+"correlation_worker_{}-2.stats".format(process_id)
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
        result = func(*args,**kwargs)
        if profile:
            if Multiprocessing.get_process_name() == process_id:
                profiler.disable()
                profiler.dump_stats(path)
        return result
    return outer
    

class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        args = []
        arg_names = [
            'compute',
            'max_n_patterns',
            'batch_size',
            'n_processes',
            'list_inp',
            'image_dimensions',
            'intensity_pixel_threshold', 
            'intensity_radial_pixel_filter',
            'use_binary_mask',
            'background_subtraction',
            'ROI_normalization',
            'ROI_mean_filter',
            'pixel_size',
            'sample_distance',
            'wavelength',
            'polarization_correction',
            'solid_angle_correction',
            'qrange',
            'qrange_xcca',
            'fc_n_max',
            'phi_range',
            'ccf_2p_symmetrize',
            'detector_origin',
            'interpolation_order' 
        ]
        opt = settings.project
        if isinstance(opt.qrange,bool) or isinstance(opt.qrange_xcca,bool):
            wavelength = opt.wavelength
            sample_distance = opt.sample_distance
            detector_edge = opt.image_dimensions[0]/2*(opt.pixel_size/1000)
            scattering_angle = np.arctan(detector_edge/sample_distance)
            q_max = scattering_angle_to_reciprocal_radii(scattering_angle,wavelength)
            qrange = (0,q_max,q_max/(opt.image_dimensions[0]/2-1))
            qrange_xcca = ((0,q_max,1),(0,q_max,1))
            if isinstance(opt.qrange,bool):
                opt.qrange = qrange
            if isinstance(opt.qrange_xcca,bool):
                opt.qrange_xcca = qrange_xcca
            
        for n in arg_names:
            args.append(settings.project.get(n))

        
        args[4] = database.project.get_path('input_file_list')
        #log.info(f'len arg names - {len(arg_names)} len args = {len(args)}')            
        self.data_reader = DataReader(*args)
        
    def run(self):
        result = self.data_reader.run_processing_in_parallel()
        database.project.save('ccd',result)
        return {},locals()
    
# reading, processing and storing the data                
class DataReader():
    def __init__(self, compute, patterns_max, batch_size, numcpus_max,list_inp, image_dimensions,intensity_pixel_threshold, intensity_radial_pixel_filter, mask_binary, background_subtraction, ROInormalization, ROImeanfilter,pixsz, det_sam, wavelng, xpolarization, solid_angle_correction, qrange, qrange_xcca, fc_n_max, phirange, ccf_2p_symmetrize, dpcenter, interp_order):
        self.start_time = time.time()
        self.compute=compute
        self.patterns_max=patterns_max
        self.batch_size=batch_size
        self.numcpus_max=numcpus_max
        self.split_mode = settings.project.split_mode
        self.list_inp=list_inp
        self.intensity_pixel_threshold=intensity_pixel_threshold
        self.intensity_radial_pixel_filter=intensity_radial_pixel_filter
        self.mask_binary_inp=mask_binary
        self.background_subtraction=background_subtraction
        self.ROInormalization=ROInormalization
        self.ROImeanfilter=ROImeanfilter
        self.interp_order=interp_order
        self.fc_n_max=fc_n_max
        self.ccf_2p_symmetrize=ccf_2p_symmetrize
        self.pixelsize=pixsz
        self.det_sam=det_sam
        self.wavelng=wavelng
        self.dpcenter=dpcenter
        self.xpolarization=xpolarization
        self.solid_angle_correction=solid_angle_correction
        self.img_shape=image_dimensions
        
        # analyze dependencies of computations and update the "compute" list
        self._analyse_dependencies()
        
        #if os.path.exists(self.dir_save) is not True:
        #        os.makedirs(self.dir_save)
        
        #print('\nResults output directory: {}'.format(self.dir_save))
        print('Input file with a file list: {}'.format(self.list_inp))
         
        self.file_list=self._read_line_text(self.list_inp)
        self.M=len(self.file_list)
        print('Number of files specified in the list : {}'.format(self.M))
        cnt_r=0
        for i in range(self.M):
            if os.path.exists(self.file_list[i]):
                cnt_r+=1
            else:
                print('File {} does not exist '.format(self.file_list[i]))
        
        if cnt_r != self.M:
            print('Error: only {} files of {} specified in the list {} have been found.\nCorrect the input file list. \n'. format(cnt_r, self.M, self.list_inp))        
            sys.exit(1)
        
        self.good_frames=np.ones(self.M, dtype=int) # array with labels for each frame "1"-good, "0"-bad; initially we assume that all frames are good
        
        # read binary mask that will be applied to all images
        if self.mask_binary_inp==True:
            path = database.project.get_path('binary_mask')
            if os.path.exists(path):
                self.mask_binary=self._read_binary_2D_arr(path, self.img_shape)
            else:
                print('Error: Input file {} with binary mask have not been found.\n'.format(path))        
                sys.exit(1)
                
        # read background data from the file
        if self.background_subtraction:
            path = database.project.get_path('background')
            if os.path.exists(path):
                self.background_data=self._read_binary_2D_arr(path, self.img_shape)
            else:
                print('Error: Input file {} with background data have not been found.\n'. format(path))        
                sys.exit(1)
            
        # determime the reciprocal space geometry and related exp qunatities
        self._prepare_polar_representation(qrange, qrange_xcca, phirange) 
        
        
        # determine polarization and solid angle correction factors
        if self.xpolarization[0]:
            self._determine_polarization_correction()
        if self.solid_angle_correction==True:
            self._determine_solid_angle_correction()
                
        # define ROI for normalizing data
        if self.ROInormalization[0] or self.ROImeanfilter[0]:
            self.ROInorm_qpos1=np.abs(self.qvals - self.ROInormalization[1]).argmin()
            self.ROInorm_qpos2=np.abs(self.qvals - self.ROInormalization[2]).argmin()             
            if (self.ROInorm_qpos1==self.ROInorm_qpos2):
                print("WARNING: ROI normalization range ({},{}) contains only 1 radial point".format(self.qvals[self.ROInorm_qpos1],self.qvals[self.ROInorm_qpos2]))
            else:
                print("ROI normalization range ({},{}) contains {} radial points".format(self.qvals[self.ROInorm_qpos1],self.qvals[self.ROInorm_qpos2], self.ROInorm_qpos2-self.ROInorm_qpos1+1))    
        
        #initialize xcca functionality
        if "xcca" in self.compute:
            self.xcca_data=ccf_analysis(self.n_q1, self.n_q2, self.n_phi, self.q1vals_pos, self.q2vals_pos)
    
    
    # initiating multiprocessing, collecting the results
    #
    def run_processing_in_parallel(self):
        self.M=min(self.M, self.patterns_max)
        numcpus= Multiprocessing.get_free_cpus()
        self.numcpus=min(self.numcpus_max, numcpus)
    
        # create a list of batch indices
        #if 
        #batches = [(i, min(i + self.batch_size, self.M)) for i in range(0, self.M, self.batch_size)]
        #self.num_batches=len(batches)
        
        xprint("Number of CPUs available: {}, requested: {},  to be used: {} ".format(numcpus, self.numcpus_max, Multiprocessing._read_number_of_processes(self.numcpus_max)))
        xprint("Number of images to be processed: {}".format(self.M))
         
        # Process data batches in parallel
        mode = Multiprocessing.MPMode_Queue(assemble_outputs=False)
        results = Multiprocessing.process_mp_request(self.process_batch,mode,input_arrays=[np.arange(self.M)],n_processes = self.numcpus,split_mode=self.split_mode,call_with_multiple_arguments=True)
        
        if "is_good" in self.compute: 
                isgood_vals = np.zeros(self.M, dtype=int)
        if "waxs" in self.compute:    
                waxs_vals = np.zeros((self.M, self.n_q))
        if "waxs_aver" in self.compute:     
                waxs_aver_vals = np.zeros(self.n_q)
        if "xcca" in self.compute:   
            if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute):
                ccf_q1q2 = np.zeros((self.n_q1, self.n_q2, self.n_phi))
                ccf_q1q2_mask = np.zeros((self.n_q1, self.n_q2, self.n_phi), dtype=int)
        n_processes = len(results)
        log.info(f'len results = {len(results)}')
        split_ids = Multiprocessing.split_mp_arguments([np.arange(self.M)],n_processes,mode=self.split_mode)['indices']
        for i, batch in results.items():
            if not isinstance(batch,dict):
                batch = batch[0]
            #start_idx = i * self.batch_size
            #end_idx = min(start_idx + self.batch_size, self.M)
            #log.info(f'batch = {batch}')            
            if "is_good" in self.compute: 
                isgood_vals[split_ids[i]] = batch["is_good"]
            
            if "waxs" in self.compute:    
                waxs_vals[split_ids[i],:] = batch["waxs"]
            
            if "xcca" in self.compute: 
                if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute):   
                    np.add(ccf_q1q2, batch["ccf_q1q2"], out=ccf_q1q2)
                    np.add(ccf_q1q2_mask, batch["ccf_q1q2_mask"], out=ccf_q1q2_mask)
                
            
        Mgood=int(np.sum(isgood_vals))     #normalize properly        
        print("CCfs were averaged over {} selected images".format(Mgood))
        
        if "waxs_aver" in self.compute:    
            np.mean(waxs_vals, axis=0, where=(isgood_vals[:,None]==1), out=waxs_aver_vals)
                                
                
        if "xcca" in self.compute:      
            if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute):
                np.divide(ccf_q1q2, ccf_q1q2_mask, out=ccf_q1q2, where=(ccf_q1q2_mask!=0))
                ccf_q1q2[ccf_q1q2_mask==0]=np.nan
        
        if self.ccf_2p_symmetrize==True:
            posPi2=np.abs(self.phi - Pi/2.0).argmin()
            posPi=np.abs(self.phi - Pi).argmin()
            pos3Pi2=np.abs(self.phi - 3*Pi/2.0).argmin()
            ccf_q1q2[...]=self.xcca_data.symmetrize_ccf(ccf_q1q2, posPi2, posPi, pos3Pi2) 
            print("Two-point CCF has been symmetrized using the angular range (Pi/2, 3Pi/2)")
            
                   
        if "ccf_q1q2_fc" in self.compute:    
            ccf_q1q2_fc=self.xcca_data.ccf_fcs(ccf_q1q2)[...,:self.fc_n_max]
        

        # save results to output file
        result = {}
         
        if "waxs_aver" in self.compute:    
            prefix_save="iaverage"
            result['average_intensity'] = waxs_aver_vals
                  
        if "ccf_q1q2" in self.compute:
            prefix_save="ccf_q1q2_2p"
            result['cross_correlation'] = {'I1I1':ccf_q1q2}
            log.info(f'nans in cc = {np.isnan(ccf_q1q2).any()}')
            log.info(f'all nans = {np.isnan(ccf_q1q2).all()}')
            log.info(ccf_q1q2)
            
        if "ccf_q1q2_fc" in self.compute: 
            result['cross_correlation'] = {'I1I1_fc':ccf_q1q2_fc}
                        
        result['radial_points']=self.qvals
        result['angular_points']= self.phi#*(180.0/np.pi)
        result['num_images_processed']= self.M
        result['num_images_good'] = Mgood
        result['xray_wavelength']= settings.project.wavelength
        return result 
         

    
    # processing a batch of images in parallel
    #
    @profile_modifier
    def process_batch(self, frames,**kwargs):
        #read a batch of images from the files
        n_frames = len(frames)
        isgood_vals = np.zeros(n_frames)
        waxs_vals = np.zeros((n_frames,self.n_q),dtype = float)
        if "xcca" in self.compute:    
            if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute): 
                ccf_q1q2=np.zeros((self.n_q1, self.n_q2, self.n_phi))
                ccf_q1q2_mask = np.zeros((self.n_q1, self.n_q2, self.n_phi), dtype=int)
        if isinstance(self.batch_size,bool) or (self.batch_size ==0):
            n_parts = 1
        elif n_frames%self.batch_size!=0:
            n_parts = n_frames//self.batch_size+1
        else:
            n_parts = n_frames//self.batch_size
        n_parts = max(n_parts,1)
        
        batches = np.array_split(frames,n_parts)
        batch_ids = np.array_split(np.arange(len(frames)),n_parts)
        batch_nr = 1
        for batch,ids in zip(batches,batch_ids):
            data=np.empty((len(batch),self.img_shape[0], self.img_shape[1]))
            mask=np.ones(data.shape, dtype=int)
            for i,idx in enumerate(batch):
                #log.info(f'idx = {idx}')
                data[i,:,:]=self._read_binary_2D_arr(self.file_list[idx], self.img_shape)
            # perform processing of each frame in the batch
            for i,idx in enumerate(batch):            
                if self.good_frames[idx]==0: # if image was already labelled as bad
                    vals_dictionary = dict(zip(self.compute, repeat(0)))                    
                    if "is_good" in self.compute: 
                        vals_dictionary["is_good"]=0
                    if "xcca" in self.compute:    
                        vals_dictionary["image_polar"]=0
                        vals_dictionary["mask_polar"]=0    
                       
                else:
                    vals_dictionary=self.process_image(data[i], mask[i])
                    
                if "is_good" in self.compute: 
                    isgood_vals[ids[i]]=vals_dictionary["is_good"]
            
                if "waxs" in self.compute:    
                    waxs_vals[ids[i],:]=vals_dictionary["waxs"]
                        
                if (isgood_vals[i]==1) and  ("xcca" in self.compute):    
                    image_polar=vals_dictionary["image_polar"]
                    mask_polar=vals_dictionary["mask_polar"]
                    
                    if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute): 
                        valtmp,valmask=self.xcca_data.ccf_twopoint_q1_q2_mask_corrected(image_polar, mask_polar)
                        #isnotnone=~np.isnan(valtmp)
                        np.add(ccf_q1q2, valtmp.real, out=ccf_q1q2, where=valmask) 
                        np.add(ccf_q1q2_mask, 1, out=ccf_q1q2_mask, where=valmask)
            xprint(f'Process {kwargs["local_name"]}: Completed batch {batch_nr}/{n_parts}.')
            batch_nr+=1
        #return a combined result
        result_batch={}
        if "is_good" in self.compute: 
            result_batch["is_good"]=isgood_vals
            
        if "waxs" in self.compute:    
            result_batch["waxs"]=waxs_vals
       
        if "xcca" in self.compute: 
            if ("ccf_q1q2" in self.compute) or ("ccf_q1q2_fc" in self.compute):    
                result_batch["ccf_q1q2"]=ccf_q1q2
                result_batch["ccf_q1q2_mask"]=ccf_q1q2_mask
        
        #print("batch {} has been processed".format(frames))      
          
        return result_batch
    
    # processing an individual image 
    #
    def process_image(self, image, mask):
        
        vals_dictionary = dict(zip(self.compute, repeat(0))) # output results
        
        # update the mask if necessary
        if self.intensity_pixel_threshold[0]:    # mask pixels with intensities outside of the specified thresholds
                mask[(image<self.intensity_pixel_threshold[1]) | (image>self.intensity_pixel_threshold[2])]=0            
        if self.mask_binary_inp==True:
            np.multiply(mask, self.mask_binary, out=mask)
        
        # background correction
        if self.background_subtraction:
            image=np.subtract(image, self.background_data)
            
        #apply mask to the image 
        np.multiply(image, mask, out=image)
        
        # interpolate to polar coordinates
        image=sp.ndimage.map_coordinates(image, [self.cart_x.ravel(), self.cart_y.ravel()], order=self.interp_order, mode='constant', cval=0, prefilter=True)
        image.shape=(self.n_q, self.n_phi)
        mask=sp.ndimage.map_coordinates(mask, [self.cart_x.ravel(), self.cart_y.ravel()], order=self.interp_order, mode='constant', cval=0, prefilter=True)
        mask.shape=(self.n_q, self.n_phi)
        
        
        # radial pixel filter (can be applied only to the data in polar coordinates)
        if self.intensity_radial_pixel_filter[0]:
            if self.intensity_radial_pixel_filter[1][0]=='average_sigma':
                val_av, val_sig = self.i_average_and_sigma_azimuthal(image, mask)  
            elif self.intensity_radial_pixel_filter[1][0]=='median_mad':
                val_av, val_sig = self.i_median_and_mad(image, mask)
            else:
                print("Error: incorrect settings for intensity radial pixel filter. The filter will not function properly")
            
            
            #apply filter
            mask[np.abs(image-val_av[:,None])>self.intensity_radial_pixel_filter[1][1]*val_sig[:,None]]=0
            np.multiply(image, mask, out=image)      
            
                               
        is_good=1 # initially consider all images as good
        
        if (np.sum(mask)==0):
            print("found a completely masked image")
            vals_dictionary["is_good"]=0 #update dictionary
            return vals_dictionary
        
                 
        if self.ROImeanfilter[0] or self.ROInormalization[0] or ("mean_roi" in self.compute):
            meav_roi_val=np.mean(image[self.ROInorm_qpos1:self.ROInorm_qpos2], where=(mask[self.ROInorm_qpos1:self.ROInorm_qpos2]==1)) # average intensity/pixel
        
        if self.ROImeanfilter[0]:
            if (meav_roi_val<self.ROImeanfilter[1]) or (meav_roi_val>self.ROImeanfilter[2]):
                is_good=0
                
        if self.ROInormalization[0]: 
            image=np.divide(image, meav_roi_val)
                
        if self.xpolarization[0]==True:
                image=np.multiply(image, self.Pfactor)
                
        if self.solid_angle_correction==True:
                image=np.multiply(image, self.SolAngCorr)

                
        #compose results
        if "is_good" in self.compute: 
            vals_dictionary["is_good"]=is_good
        
        if "waxs" in self.compute: 
            val_av = self.i_average_azimuthal(image, mask)   
            vals_dictionary["waxs"]=val_av
        if "xcca" in self.compute:    
            vals_dictionary["image_polar"]=image
            vals_dictionary["mask_polar"]=mask
                  
        return  vals_dictionary 
    
   
    
    # compute the mean value and standard deviation for an image determined on a polar grid
    #
    def i_average_and_sigma_azimuthal(self, data_polar, mask_polar):
        imean=np.mean(data_polar, axis=1, where=(mask_polar==1))
        istd=np.std(data_polar, axis=1, where=(mask_polar==1))
        return imean, istd 
    
    # compute the mean value for an image determined on a polar grid
    #
    def i_average_azimuthal(self, data_polar, mask_polar):
        imean=np.mean(data_polar, axis=1, where=(mask_polar==1))
        return imean
        
    # compute median value and median absolute deviation for an image determined on a polar grid
    #
    def i_median_and_mad(self, data_polar, mask_polar):
        imedval=np.nanmedian(np.where(mask_polar==1, data_polar, np.nan),axis=1)
        imad=spst.median_abs_deviation(np.where(mask_polar==1, data_polar, np.nan), axis=1, nan_policy='omit')
        return imedval, imad    
                    
    # analyze dependencies and update the compute list
    #
    def _analyse_dependencies(self):
        if ("ccf_q1q2" in self.compute): 
            if ("xcca" not in self.compute): 
                self.compute.append("xcca")         
        if "waxs_aver" in self.compute:
            if not ("waxs" in self.compute):
                 self.compute.append("waxs")
   
                
    # reciprocal space geometry
    #
    def _prepare_polar_representation(self, qrange, qrange_xcca, phirange):
    
        self.q_min=qrange[0]
        self.q_max=qrange[1]
        self.q_step=qrange[2]
        pixsz=self.pixelsize*0.001 # conversion to [mm]
        self.n_q=int((self.q_max-self.q_min)/self.q_step+1)    # number of points in the raial direction (radial sampling)
        self.phi_min=phirange[0]
        self.phi_max=phirange[1]
        n_phi=phirange[2]
        
        detq=(2*Pi*pixsz)/(self.wavelng*self.det_sam) #  Detector resolution in reciprocal space [ Angstroem^(-1)] 
       
        self.qvals=np.arange(self.n_q)*self.q_step + self.q_min    # qvals for each ring
        self.theta=2.0*np.arcsin(self.qvals*self.wavelng/(4.0*Pi)) # scattering angle 2theta for each q-ring
      
        # maximum feasible azimuthal sampling
        min_circ_rad_pix=int(math.floor(self.det_sam*math.tan(self.theta[0])/pixsz)) # circle radius [pixels] determined for q_min
        max_circ_rad_pix=int(round(self.det_sam*math.tan(self.theta[-1])/pixsz)) # circle radius [pixels] determined for q_max
        step_circ_rad_pix=self.det_sam*math.tan(self.theta[1]-self.theta[0])/pixsz # circle radius [pixels] determined for q_step (at the lowest resoltion)
        
        self.maxpix=round(2*Pi*max_circ_rad_pix) # number of pixels on the circumference of the radius q_max, determines the maximum feasible azimuthal sampling (1 pixel step)
        if (self.maxpix % 2) != 0:
            self.maxpix+=1
        self.maxpix=int(self.maxpix)
        
        
        # number of points in the azimuthal direction(azimuthal sampling) to be used, the same for all resolution rings
        if phirange[3]=='max':
            self.n_phi=min(self.maxpix, n_phi)
        elif phirange[3]=='min':
            self.n_phi=max(self.maxpix, n_phi)
        else:
            self.n_phi=n_phi
                      
        self.phi_step=(self.phi_max-self.phi_min)/self.n_phi
        self.phi=np.arange(self.n_phi)*(self.phi_max-self.phi_min)/float(self.n_phi) + self.phi_min # azimuthal angles for each point
             
        print("Reciprocal space parameters: q_min={:.6f} [A^-1], q_max={:.6f} [A^-1], q_step={:.6f} [A^-1], N_q={}".format(self.qvals[0], self.qvals[-1], self.q_step, self.n_q))
        print("Reciprocal space parameters: q_min={} [pixels], q_max={} [pixels], q_step={:.3f} [pixels]".format(min_circ_rad_pix, max_circ_rad_pix, step_circ_rad_pix))
        
        if (self.q_step<detq):
            print("WARNING: the specified q_step={:.6f} is smaller than the detector resolution detq={:.6f} (q_step-detq={:.8f}). To save computation time set q_step>=detq".format(self.q_step, detq, self.q_step-detq))
        print("Scattering angles: 2theta_min={:.4f} [deg], 2theta_max={:.4f} [deg], 2theta_step={:.4f} [deg]".format(self.theta[0]*180.0/Pi, self.theta[-1]*180.0/Pi, (self.theta[1]-self.theta[0])*180.0/Pi))
        print("Maximum feasible number of azimuthal points at q_max: max_n_phi={}; requested : {}; will be used={}".format(self.maxpix, n_phi, self.n_phi)) 
  
        #polar grid [pixel coordinates]
        self.pol_q = np.outer(np.tan(self.theta)*self.det_sam/pixsz, np.ones(self.n_phi) )
        self.pol_phi = np.outer( np.ones(self.n_q), np.arange(self.n_phi)*(self.phi_max-self.phi_min)/float(self.n_phi) + self.phi_min)
        
        # cartesian coordinates
        self.cart_x_orig = self.pol_q*np.cos(self.pol_phi)
        self.cart_y_orig = self.pol_q*np.sin(self.pol_phi)
        
        self.cart_x = self.cart_x_orig + self.dpcenter[0]
        self.cart_y = self.cart_y_orig + self.dpcenter[1]
            
        # computations of the [q1,q2] ranges for xcca
        q1pos1=np.abs(self.qvals - qrange_xcca[0][0]).argmin()# find the positons of the q1_min and q1_max in the array  self.qvals
        q1pos2=np.abs(self.qvals - qrange_xcca[0][1]).argmin()
        self.q1vals_pos=np.arange(q1pos1,q1pos2+1,qrange_xcca[0][2]) # indices of the q1 values in the array of self.qvals
        self.q1vals=self.qvals[self.q1vals_pos]
        self.n_q1=self.q1vals.shape[0]
        print("q1 values for computing the CCFs: q1_min={:.6f} [A^-1], q1_max={:.6f} [A^-1], q1_step={:.6f} [A^-1], N_q1={}".format(self.q1vals[0], self.q1vals[-1], self.q1vals[1]-self.q1vals[0], self.n_q1))
        
        q2pos1=np.abs(self.qvals - qrange_xcca[1][0]).argmin()# find the positons of the q1_min and q1_max in the array  self.qvals
        q2pos2=np.abs(self.qvals - qrange_xcca[1][1]).argmin()
        self.q2vals_pos=np.arange(q2pos1,q2pos2+1,qrange_xcca[1][2]) # indices of the q1 values in the array of self.qvals
        self.q2vals=self.qvals[self.q2vals_pos]
        self.n_q2=self.q2vals.shape[0]
        print("q2 values for computing the CCFs: q2_min={:.6f} [A^-1], q2_max={:.6f} [A^-1], q2_step={:.6f} [A^-1], N_q2={}".format(self.q2vals[0], self.q2vals[-1], self.q2vals[1]-self.q2vals[0], self.n_q2))
        
  
    # determine polarization factor
    # the experimental intensity should be multiplied by self.Pfactor[i,j]
    #
    def _determine_polarization_correction(self):
        
        self.Pfactor=np.ones((self.n_q, self.n_phi))
        
        # here 'h' and 'v' polarisations are implemented in the way to be compatible with array indexing (firt index - verical direction, second, horisontal)
        if self.xpolarization[1]=='v':
          for i in range(self.n_q):
            th=self.theta[i]
            for j in range(self.n_phi):
                fi=self.phi[j]
                self.Pfactor[i,j]=1.0/(math.cos(th)**2+(math.sin(th)**2)*(math.sin(fi)**2))

        elif self.xpolarization[1]=='h':
          for i in range(self.n_q):
            th=self.theta[i]
            for j in range(self.n_phi):
                fi=self.phi[j]
                self.Pfactor[i,j]=1.0/(math.cos(th)**2+(math.sin(th)**2)*(math.cos(fi)**2))
        
    # determine solid_angle_correction factor
    # the experimental intensity should be multiplied by self.SolAngCorr[i] 
    #
    def _determine_solid_angle_correction(self):
        self.SolAngCorr=np.empty((self.n_q, self.n_phi)) # multiply raw intensity by this factor 
        for i in range(self.n_q):
            th=self.theta[i]
            self.SolAngCorr[i]=1.0/(math.cos(th)**3)
    
    # read text file content line-by-line
    def _read_line_text(self, fname):
        with open(fname) as f:
            text_lines = f.readlines()
            text_lines = [x.strip() for x in text_lines] 
            text_lines = list(filter(None, text_lines))
            base_path = os.path.dirname(database.project.get_path('input_file_list'))
            for _id,f in enumerate(text_lines):
                if f[:2]=='./':
                    text_lines[_id]=os.path.join(base_path,f[2:])
            return text_lines

    # read 2D binary array of a given shape from a file
    def _read_binary_2D_arr(self, fname, shape, dtype='f', bo='<' ): 
        with open( fname, "rb") as file:
            data = file.read()
        if len(data)!=0:  
            fmt=bo+str(shape[0]*shape[1])+dtype
            dtuple=struct.unpack(fmt, data)
            data=np.asarray(dtuple).reshape(shape)
            #data = database.project.load(fname).reshape(shape)
            log.info(f'data contains {np.isnan(data).sum()} nan values')
            data[np.isnan(data)]=0
            #log.info(f'datatype = {data.dtype}')
            #data = data.astype(float)
            #log.info(f'data = {data[0][:3]}')
            #log.info(f'data db = {database.project.load(fname)[:3]}')
            return data
        else:
            return np.array([]) 
            
