import logging
import sys
import os
import numpy as np
import traceback
file_path = os.path.realpath(__file__)
plugin_dir = os.path.dirname(file_path)
os.chdir(plugin_dir)
#from xframe import dependency_inject_SOFT
#dependency_inject_SOFT()

from xframe.library import mathLibrary as mLib
from xframe.library.mathLibrary import SampleShapeFunctions
from xframe.library.mathLibrary import get_soft_obj
from xframe.library.mathLibrary import PolarIntegrator
from xframe.library.gridLibrary import uniformGrid_func
from xframe.library.gridLibrary import ReGrider
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import GridFactory
from xframe.library.pythonLibrary import plugVariableIntoFunction,create_threshold_projection
from xframe.library.pythonLibrary import selectElementOfFunctionOutput_decorator
from xframe.library.pythonLibrary import xprint
from xframe.interfaces import ProjectWorkerInterface

#from .projectLibrary.hankel_transforms import generateHT_SinCos
from .projectLibrary.hankel_transforms import generate_weightDict
from .projectLibrary.fourier_transforms import generate_ft
from .projectLibrary.misk import getAnalysisRecipeFacotry
from .projectLibrary.misk import get_analysis_process_factory
from .projectLibrary.misk import generate_load_from_dict
from .projectLibrary.misk import generate_calc_center
from .projectLibrary.classes import FTGridPair
from .projectLibrary.fxsShrinkWrap import generateSW_SupportMaskRoutine
from .projectLibrary.fxsShrinkWrap import generateSW_multiply_ft_gaussian
from .projectLibrary.fxs_Projections import generate_negative_shift_operator,generate_shift_by_operator
from .projectLibrary.harmonic_transforms import HarmonicTransform
from .projectLibrary.misk import _get_reciprocity_coefficient
from .projectLibrary.resolution_metrics import PRTF_fxs,FQCB_2D,FQCB_3D,FSC_single_fxs,FSC_bit_limit
from .projectLibrary.fxs_invariant_tools import intensity_to_deg2_invariant,harmonic_coeff_to_deg2_invariants
from xframe.externalLibraries import shtns_plugin,soft_plugin
#from analysisLibrary.classes import ReciprocalProjectionData
from xframe import settings
from xframe import database
from xframe import Multiprocessing



##### only for testing, this breaks dependency scheme !!!
from xframe import Multiprocessing

log=logging.getLogger('root')

#class Worker(RecipeInterface):
class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        comm_module = Multiprocessing.comm_module        
        self.opt = settings.project
        self.db = database.project
        self.process_factory=get_analysis_process_factory(self.db,[])
        #self.mtip=MTIP(self.process_factory,database,comm_module,opt)
        #self.mp_opt=opt['multi_process']
        #self.mtip_opt=opt
        self.comm_module=comm_module
                
    def post_processing(self,result,ft_grids,r_opt):
        log.info('Start post processing.')
        save = self.db.save
        try:
            reciprocity_coefficient = _get_reciprocity_coefficient(r_opt.fourier_transform)
            result['reciprocity_coefficient'] = reciprocity_coefficient
            result['internal_grid'] = {'real_grid':ft_grids.realGrid[:],
                                       'reciprocal_grid':ft_grids.reciprocalGrid[:]}
            #    if self.mp_opt['type'] != 'routine_construction_and_phasing_and_post_processing':
            #        processed_results=self.results.get('MTIP',{})
            #        processed_results_list=[result_dict for result_dict in processed_results]
            #        #log.info([type(result_dict) for result_dict in processed_results])
            #        #log.info([result_dict.shape for result_dict in processed_results])
            #        errors = []
            #        for result_dict in processed_results_list:
            #            grid_pair = result_dict.pop('grid_pair')
            #            errors.append(result_dict['error_dict']['real_rel_mean_square'][-1])
            #            r_ids=np.argsort(errors)
            #        log.info('error sorted reconstruction = {} \n errors ={}'.format(r_ids,np.array(errors)[r_ids]))
            #        processed_results_dict={str(_id):processed_results_list[_id] for _id in r_ids}
            #        data_dict={'configuration':{'internal_grid':grid_pair,'opt':self.mtip.opt.__dict__},'reconstruction_results':processed_results_dict}
            save('average_results',result)
        except Exception as e:
            traceback.print_exc()
            log.error(e)
    def get_averaged_projection_matrices(self,projection_matrices,average_scaling_factors):
        averaged_matrices = []
        n_matrices = len(projection_matrices[0])
        N_datasets = len(projection_matrices) 
        for i in range(n_matrices):
            temp = projection_matrices[0][i]/average_scaling_factors[0]**2
            for j in range(1,N_datasets):
                temp += projection_matrices[j][i]/average_scaling_factors[j]**2
            averaged_matrices.append(temp/N_datasets)
        return averaged_matrices
            
    def run(self):
        xprint('load reconstructions.')
        reconstructions,masks,r_opt,errors,start_ids,reference_arg,reconstructions_keys,projection_matrices,reconstruction_ids_per_file = getattr(self,self.opt['load_routine'])()        
        xprint('done.\n')
        if len(reconstructions)<2:
            log.error(f'Less than 2 reconstructions where found in the specified reconstruction files:\n {settings.project.reconstruction_files}\n There is nothing to average, abborting calculation.')
            return
        
        if r_opt['GPU'].get('use',False):
            settings.general.n_control_workers = r_opt['GPU'].get("n_gpu_workers",4)
            Multiprocessing.comm_module.restart_control_worker()
        n_reconstructions = self.opt['selection'].get('n_reconstructions','all')
        if not isinstance(n_reconstructions,int) or isinstance(n_reconstructions,bool):
            n_reconstructions = len(reconstructions)
            
        dim = r_opt.dimensions
        if dim == 2:
            result,_locals = self.run_2d(reconstructions,masks,r_opt,errors,start_ids,reference_arg,reconstructions_keys,n_reconstructions,projection_matrices,reconstruction_ids_per_file)
        elif dim == 3:
            result,_locals = self.run_3d(reconstructions,masks,r_opt,errors,start_ids,reference_arg,reconstructions_keys,n_reconstructions,projection_matrices,reconstruction_ids_per_file)
        return result,_locals
            
    def run_2d(self,reconstructions,masks,r_opt,errors,start_ids,reference_arg,reconstructions_keys,n_reconstructions,projection_matrices,reconstruction_ids_per_file):
        opt = self.opt
        xprint('setting up alignment operators.')
        #log.info('reference arg = {}'.format(reference_arg))
        align_opt = {'opt':opt,'r_opt':r_opt}
        alignment = Alignment(self.process_factory,self.db,align_opt)
        fft = self.process_factory.get_operator('fourier_transform')
        ifft = self.process_factory.get_operator('inverse_fourier_transform')
        cht = self.process_factory.get_operator('complex_harmonic_transform')
        icht = self.process_factory.get_operator('complex_inverse_harmonic_transform')               
        xprint('done.\n')
        
        if opt.get('center_reconstructions',False):            
            xprint('centering reconstructions.')
            if opt['multi_process']['use']:
                def mp_center(r_id,**kwargs):
                    alignment = Alignment(self.process_factory,self.db,align_opt)
                    r = reconstructions[r_id]
                    temp = alignment.shift_to_center(*r)                    
                    return [r_id,temp]
                Multiprocessing.MPMode_Queue(assemble_outputs=True)
                outs = Multiprocessing.comm_module.request_mp_evaluation(mp_center,input_arrays=[np.arange(len(reconstructions))],call_with_multiple_arguments=False,split_mode = 'modulus')
                
                #log.info('len outs[0] = {}'.format(len(outs[0])))
                #log.info('len outs = {}'.format(type(outs)))
                for r_id,temp in outs:
                    reconstructions[r_id] = temp[:2]
            else:
                for r_id,r in enumerate(reconstructions):
                    #log.info([d.shape for d in r])
                    temp = alignment.shift_to_center(*r)
                    if opt['use_masks']:
                        shifted_mask_threshold = self.opt['shifted_mask_threshold']
                        shift = mLib.cartesian_to_spherical(-mLib.spherical_to_cartesian(temp[-1]))
                        log.info('shift = {}, - shift = {}'.format(temp[-1], shift))
                        masks[r_id] = alignment.shift_by(masks[r_id],shift)[0]
                        density = temp[0]
                        density[masks[r_id]<shifted_mask_threshold] = 0
                        #temp[0] = density
                    reconstructions[r_id] = temp[:2]
            xprint('done.\n')
            
        scaling_factors = np.ones(len(reconstructions))
        if opt.normalize_reconstructions.use:
            xprint('normalizing reconstructions.')            
            if opt.normalize_reconstructions.mode == 'max':
                for r_id,r in enumerate(reconstructions):
                    xprint(f'Reconstruction {r_id} has real max {np.max(r[0])} and mean {np.mean(r[0])}')
                    scale = np.max(r[0][r[0]>0].real)
                    scaling_factors[r_id]=scale
                    #log.info('{}{}'.format(scale,scale2))
                    reconstructions[r_id] = [r[0]/scale,r[1]/scale]
                    
            elif opt.normalize_reconstructions.mode == 'mean':
                for r_id,r in enumerate(reconstructions):
                    scale = np.mean(r[0][r[0]>0])
                    scaling_factors[r_id]=scale
                    reconstructions[r_id] = [r[0]/scale,r[1]/scale]
            xprint('done.\n')

        average_scaling_factors_per_file = np.ones(len(reconstruction_ids_per_file))
        for file_id,r_ids in enumerate(reconstruction_ids_per_file):
            #log.info('ids per file = {}'.format(r_ids))
            average_scaling_factors_per_file[file_id] = np.mean(scaling_factors[r_ids])
                
        ft_grids = alignment.ft_grids
        rgrid = ft_grids.reciprocalGrid[:]
        integrate = PolarIntegrator(rgrid).integrate
        #log.info('leen reconstructions before reference extraction = {}'.format(len(reconstructions)))
        reference_reconstruction = reconstructions.pop(reference_arg)
        #log.info('leen reconstructions after reference extraction = {}'.format(len(reconstructions)))
        reference_mask = masks.pop(reference_arg)
        d_ref = reference_reconstruction[0].copy()
        ft_d_ref = reference_reconstruction[1].copy()
       
        if opt.get("pointinvert_reference",False):
            xprint('\tpoint inverting reference.')
            #(ift(signal[1].conj()),signal[1].conj())
            ft_d_ref = ft_d_ref.conj()
            d_ref = ifft(fft(d_ref).conj())
            #reference_d = ifft(reference_i.conj())
            #reference_reconstruction = [reference_d,reference_i]
            reference_mask[:]=True #Propper mask handling not implemented for pointinverse
            xprint('done.\n')
        
        aligned_reconstructions = [[d_ref,ft_d_ref]]
        p_inv_metric = []
        ft_d_ref_pinv = ft_d_ref.conj()
        #aligned_reconstructions = [[ifft(ft_d_ref.copy()),ft_d_ref]]
        #log.info('len reconstructions before pinv correction = {}'.format(len(reconstructions)))
        xprint('correct point inversion by comparing to reference.')
        for i,reconstruction in enumerate(reconstructions):
            d,ft_d = reconstruction            
            diff = integrate(np.abs(ft_d.imag-ft_d_ref.imag))
            diff_inv = integrate(np.abs(ft_d.imag-ft_d_ref_pinv.imag))                        
            point_inverted = diff>diff_inv
            #log.info('{} : diff = {} diff inv = {} diff2 = {} diff inv2 = {}'.format(i,diff,diff_inv,diff2,diff_inv2))
            #log.info('diff_reff = {}'.format(integrate(np.abs(ft_d_ref.imag-ft_d_ref_pinv.imag))))            
            if point_inverted:
                new_ft_d = ft_d.copy().conj()
                new_d = ifft(fft(d).conj())
            else:
                new_ft_d = ft_d
                new_d = d

            aligned_reconstructions.append([new_d,new_ft_d])
            p_inv_metric.append([diff,diff_inv,int(point_inverted)])
        xprint('done.\n')
        
        if len(aligned_reconstructions)>=n_reconstructions:
            aligned_reconstructions=aligned_reconstructions[:n_reconstructions]
        log.info('averaging over {} '.format(len(aligned_reconstructions)))

        #density_array = np.array(tuple(r[0] for r in aligned_reconstructions))
        #log.info('density_array.shape = {}'.format(density_array))
        average_density = np.mean(np.array(tuple(r[0] for r in aligned_reconstructions)),axis = 0)
        average_ft_density = np.mean(np.array(tuple(r[1] for r in aligned_reconstructions)),axis = 0)
        aligned_ft_reconstr_ftd = [fft(r[0]) for r in aligned_reconstructions]
        intensity_from_ft_density = np.mean([(r[1]*r[1].conj()).real for r in aligned_reconstructions], axis = 0)
        intensity_from_density= np.mean([(r*r.conj()).real for r in aligned_ft_reconstr_ftd], axis = 0)
        input_reconstructions = [reference_reconstruction,*reconstructions]
        input_masks = [reference_mask,*masks]

        centered_average = alignment.shift_to_center(average_density,average_ft_density)

        xprint('Calculate resolution metrics.')
        resolution_metrics = {}
        if opt.resolution_metrics.get('PRTF',False):
            prtf = PRTF_fxs(fft(average_density),intensity_from_density,averaged_projected_scattering_amplitude=average_ft_density,averaged_projected_intensity=intensity_from_ft_density)
            resolution_metrics['PRTF'] = prtf[0]
            resolution_metrics['PRTF_std'] = prtf[1]
            prtf_d = PRTF_fxs(fft(average_density),intensity_from_density)
            resolution_metrics['PRTF_from_density'] = prtf_d[0]
            resolution_metrics['PRTF_from_density_std'] = prtf_d[1]
            prtf_ftd = PRTF_fxs(average_ft_density,intensity_from_ft_density)
            resolution_metrics['PRTF_from_ft_density'] = prtf_ftd[0]
            resolution_metrics['PRTF_from_ft_density_std'] = prtf_ftd[1]

            prtf_ftI= PRTF_fxs(fft(average_density),intensity_from_ft_density)
            resolution_metrics['PRTF_ftI'] = prtf_ftI[0]
            resolution_metrics['PRTF_ftI_std'] = prtf_ftI[1]
            #log.info('PRTF shape {} PRTF dtype = {}'.format(prtf[0].shape,prtf[0].dtype))
            #log.info('PRTF {}'.format(prtf[0]))
        if opt.resolution_metrics.get('FQCB',False):
            ft_d = fft(average_density)
            b_d_rec=intensity_to_deg2_invariant(ft_d*ft_d.conj(),cht,2)
            b_ftd_rec=intensity_to_deg2_invariant(average_ft_density*average_ft_density.conj(),cht,2)
            pr_mat=np.squeeze(np.array(self.get_averaged_projection_matrices(projection_matrices,average_scaling_factors_per_file)))
            b_target = harmonic_coeff_to_deg2_invariants(2,pr_mat.T)

            #for i in range(len(b_target)):
            #    b_di=b_d_rec[i]
            #    b_ftdi=b_ftd_rec[i]
            #    b_ti=b_target[i]
            #    
            #    b_di[np.abs(b_di)<np.abs(b_di).max()/1e15]=0
            #    b_ftdi[np.abs(b_ftdi)<np.abs(b_ftdi).max()/1e15]=0
            #    b_ti[np.abs(b_ti)<np.abs(b_ti).max()/1e15]=0
                
            fqcb_d = FQCB_2D(b_d_rec,b_target,skip_odd_orders=True)
            fqcb_ftd = FQCB_2D(b_ftd_rec,b_target,skip_odd_orders=True)
            fqcb_rec = FQCB_2D(b_ftd_rec,b_d_rec,skip_odd_orders=True)
            fqcb_d_z = FQCB_2D(b_d_rec,b_target,skip_odd_orders=True,include_zero_order = True)
            fqcb_ftd_z = FQCB_2D(b_ftd_rec,b_target,skip_odd_orders=True,include_zero_order = True)
            resolution_metrics['FQCB_from_density']=fqcb_d[0]
            resolution_metrics['FQCB_from_density_std']=fqcb_d[1]
            resolution_metrics['FQCB_from_ft_density']=fqcb_ftd[0]
            resolution_metrics['FQCB_from_ft_density_std']=fqcb_ftd[1]
            resolution_metrics['FQCB_projected_vs_unprojected']=fqcb_rec[0]
            resolution_metrics['FQCB_projected_vs_unprojected_std']=fqcb_rec[1]
            
            resolution_metrics['FQCB_from_density_with_zero_order']=fqcb_d_z[0]
            resolution_metrics['FQCB_from_density_with_zero_order_std']=fqcb_d_z[1]
            resolution_metrics['FQCB_from_ft_density_with_zero_order']=fqcb_ftd_z[0]
            resolution_metrics['FQCB_from_ft_density_with_zero_order_std']=fqcb_ftd_z[1]
            
            resolution_metrics['FQCB_bl_from_density']=b_d_rec
            resolution_metrics['FQCB_bl_from_ft_density']=b_ftd_rec
            resolution_metrics['FQCB_bl_target']=b_target
            
        if opt.resolution_metrics.get('pseudo_FSC',False):
            pseudo_fsc = FSC_single_fxs(fft(average_density),average_ft_density)
            resolution_metrics['pseudo_FSC']=pseudo_fsc
            fsc_limit = FSC_bit_limit(0.5,rgrid)
            resolution_metrics['FSC_0.5bit_limit']=fsc_limit
            #input_b = 
            
        xprint('done.\n')

        xprint('Averaging completed saving results.')
        result = {
            'average':
            {
                'real_density':average_density,'normalized_real_density':self.normalize_density(average_density),'reciprocal_density':average_ft_density,'intensity_from_densities':intensity_from_density,'intensity_from_ft_densities':intensity_from_ft_density
            },
            'resolution_metrics':resolution_metrics,
            'centered_average':
            {
                'real_density':centered_average[0],'normalized_real_density':self.normalize_density(centered_average[0]),'reciprocal_density':centered_average[1]
            },
            "aligned":
            {
                #**{
                #'0':{'real_density':reference_reconstruction[0],'reciprocal_density':reference_reconstruction[0]},
                #'1':{'real_density':d,'reciprocal_density':a},
                #},
                
                **{
                    str(i):{'real_density':r[0],'reciprocal_density':r[1]} for i,r in enumerate(aligned_reconstructions)
                },
                
                #**{
                #str(i+2):{'real_density':reconstructions[i][0],'reciprocal_density':reconstructions[i][1]} for i in range(len(reconstructions))
                #}
            },
            'input':
            {
                str(i):{'real_density':input_reconstructions[i][0],
                        'reciprocal_density':input_reconstructions[i][1],
                        'support_mask':input_masks[i]} for i in range(len(input_reconstructions))
            },
            'input_meta':{
                'projection_matrices':projection_matrices,
                'reconstruction_keys':reconstructions_keys,
                'scaling_factors': np.array(scaling_factors),
                'average_scaling_factors_per_file':np.array(average_scaling_factors_per_file),
                'grids':ft_grids
            },
            'inversion_metric':
            {
                str(i+1): np.array(m) for i,m in enumerate(p_inv_metric)         
            }
        }
        self.post_processing(result,ft_grids,r_opt)
        result = 0
        return result,locals()
    def run_3d(self,reconstructions,masks,r_opt,errors,start_ids,reference_arg,reconstructions_keys,n_reconstructions,projection_matrices,reconstruction_ids_per_file):
        opt = self.opt        
        #log.info('reference arg = {}'.format(reference_arg))
        xprint('setting up alignment operators.')
        align_opt = {'opt':opt,'r_opt':r_opt}
        alignment = Alignment(self.process_factory,self.db,align_opt)
        ft_grids = alignment.ft_grids
        rgrid = ft_grids.reciprocalGrid[:]
        fft = self.process_factory.get_operator('fourier_transform')
        ifft = self.process_factory.get_operator('inverse_fourier_transform')
        xprint('done.\n')
        
        #xprint(f'sums {[np.sum(r[0]) for r in reconstructions]}')
        if opt.get('center_reconstructions',False):
            xprint('centering reconstructions.')
            if opt['multi_process']['use']:
                n_processes = opt['multi_process']['n_processes']
                log.info(f'n_processes centering {n_processes}')
                def mp_center(r_ids,**kwargs):
                    #temeporary fix to define alignment here its align_to function has problems with forking.
                    #the shift center part is likely not affected but just to be sure.
                    process_factory = get_analysis_process_factory(self.db,[])
                    alignment = Alignment(process_factory,self.db,align_opt)
                    results = []
                    for r_id in r_ids:
                        r = reconstructions[r_id]
                        temp = alignment.shift_to_center(*r)
                        results.append([r_id,temp])
                    return results
                Multiprocessing.MPMode_Queue(assemble_outputs=True)
                outs = Multiprocessing.comm_module.request_mp_evaluation(mp_center,input_arrays=[np.arange(len(reconstructions))],call_with_multiple_arguments=True,split_mode = 'modulus',n_processes = n_processes)
                
                #log.info('len outs[0] = {}'.format(len(outs[0])))
                #log.info('len outs = {}'.format(type(outs)))
                #for o in outs:
                #    log.info(f'len part = {len(o)}')
                #log.info('len outs[0] = {}'.format(len(outs[0])))
                #log.info('len outs = {}'.format(len(outs)))
                for temp in outs:
                    for r_id,result in temp:
                        reconstructions[r_id] = result[:2]
                #for r_id,temp in outs:
                #    reconstructions[r_id] = temp[:2]
            else:
                for r_id,r in enumerate(reconstructions):
                    #log.info([d.shape for d in r])
                    temp = alignment.shift_to_center(*r) 
                    if opt['use_masks']:
                        shifted_mask_threshold = self.opt['shifted_mask_threshold']
                        shift = mLib.cartesian_to_spherical(-mLib.spherical_to_cartesian(temp[-1]))
                        log.info('shift = {}, - shift = {}'.format(temp[-1], shift))
                        masks[r_id] = alignment.shift_by(masks[r_id],shift)[0]
                        density = temp[0]
                        density[masks[r_id]<shifted_mask_threshold] = 0
                        #temp[0] = density
                    reconstructions[r_id] = temp[:2]
            xprint('done.\n')
            #xprint(f'sums {[np.sum(r[0]) for r in reconstructions]}')
       
       
            
        #log.info('max mask unshifted = {}'.format(reference_mask.max()))
        #n_reconstructions= opt['selection']['n_reconstructions']
        #reconstructions = [ reconstructions[arg] for arg in np.argsort(errors)[:n_reconstructions-1]]                
        #log.info('mask shape = {}'.format(ref_mask.shape))
        #ref_mask = np.array(reference_mask,dtype=np.complex128)
        #log.info('mask shape = {}'.format(ref_mask.shape))
        #ref_r_mask = fft(ref_mask)
        #log.info('alive')

        scaling_factors = np.ones(len(reconstructions))
        if opt.normalize_reconstructions.use:
            xprint('normalizing reconstructions.')
            if opt.normalize_reconstructions.mode == 'max':                
                for r_id,r in enumerate(reconstructions):
                    if np.max(r[0]).real<=0:                        
                        xprint(f'Reconstruction {r_id} has zero max! continue.')
                        continue
                    scale = np.max(r[0][r[0]>0].real)
                    #xprint(f'Reconstruction {r_id} has max {scale} ref is {reference_arg}')
                    scaling_factors[r_id]=scale
                    #log.info('{}{}'.format(scale,scale2))
                    reconstructions[r_id] = [r[0]/scale,r[1]/scale]
            
            elif opt.normalize_reconstructions.mode == 'mean':
                for r_id,r in enumerate(reconstructions):
                    scale = np.mean(r[0][r[0]>0])
                    scaling_factors[r_id]=scale
                    reconstructions[r_id] = [r[0]/scale,r[1]/scale]
            xprint('done.\n')        
        average_scaling_factors_per_file = np.ones(len(reconstruction_ids_per_file))
        for file_id,r_ids in enumerate(reconstruction_ids_per_file):
            #log.info('ids per file = {}'.format(r_ids))
            average_scaling_factors_per_file[file_id] = np.mean(scaling_factors[r_ids])

        reference_reconstruction = reconstructions.pop(reference_arg)
        reference_mask = masks.pop(reference_arg)
        #log.info("opt pinv =  {}".format(opt.get("pointinvert_reference",False)))
        #log.info("opt =  {}".format(opt))
        if opt.get("pointinvert_reference",False):
            xprint('\tpoint inverting reference.')
            #(ift(signal[1].conj()),signal[1].conj())
            reference_i = reference_reconstruction[1].conj()
            reference_d = ifft(reference_i)
            reference_reconstruction = [reference_d,reference_i]
            reference_mask[:]=True #Propper mask handling not implemented for pointinverse
            xprint('done.\n')
        #       
        #        n_reconstructions = len(reconstructions)
        
        #reference_reconstruction = alignment.shift_by(reconstructions[0][0],np.array((100,np.pi/2,0)))
        #reference_reconstruction = alignment.rotate_by(r2[0],np.array([0.5*np.pi,0.7*np.pi,1.5*np.pi])) #alignment.shift_by(r2[0],np.array((100,np.pi/2,0)))
        #print('ref = {} sig = {}'.format(2,2))
        #reference_reconstruction = r2 #alignment.shift_by(r2[0],np.array((100,np.pi/2,0)))
        
        xprint('rotationaly aligning reconstructions to reference.')
        outs = []
        valid_errors = []
        densities = [reference_reconstruction]
        rotation_metrics = []
        rotation_angles = []
        valid_alignments= [reference_reconstruction]
        valid_alignment_ids = [0]
        alignment_error_limit = settings.project.alignment_error_limit        
        if opt['multi_process']['use']:
            n_processes = opt['multi_process']['n_processes']
            log.info(f'n_processes align {n_processes}')
            def mp_align_pattern(_ids,**kwargs):
                results = []
                #temeporary fix: We create a new alignment object here because its apply_to function has problems with forking.
                alignment = Alignment(self.process_factory,self.db,align_opt)
                for _id in _ids:
                    result = [_id,alignment.apply_to(reference_reconstruction[0].copy(),reconstructions[_id])]
                    results.append(result)
                del(alignment)
                return results
            Multiprocessing.MPMode_Queue(assemble_outputs=True)
            outs = Multiprocessing.comm_module.request_mp_evaluation(mp_align_pattern,input_arrays=[np.arange(len(reconstructions))],call_with_multiple_arguments=True,split_mode='modulus',n_processes = n_processes)
            errors = []
            for temp in outs:
                for out in temp:
                    #log.info(out.keys())
                    out = out[1]
                    densities.append(out['densities'])
                    rotation_metrics.append(out['rotation_metrics'])
                    rotation_angles.append(out['rotation_angles'])
                    #log.info('error = {}'.format(out['errors'][-1]))
                    if out['errors'][-1] < alignment_error_limit:
                        valid_errors.append(out['errors'][-1])
                        valid_alignments.append(out['densities'])
                        valid_alignment_ids.append(r_id)
                    errors.append(out['errors'][-1])
        else:
            for r_id,reconstruction in enumerate(reconstructions):
                out = alignment.apply_to(reference_reconstruction[0].copy(),reconstruction)
                outs.append(out)
                densities.append(out['densities'])
                rotation_metrics.append(out['rotation_metrics'])
                rotation_angles.append(out['rotation_angles'])
                #log.info('error = {}'.format(out['errors'][-1]))
                if out['errors'][-1] < alignment_error_limit:
                    valid_errors.append(out['errors'][-1])
                    valid_alignments.append(out['densities'])
                    valid_alignment_ids.append(r_id)
            errors = [out['errors'][-1] for out in outs]

        xprint('done.\n')        
        sorted_errors = np.argsort(valid_errors)
        aligned_reconstr = [valid_alignments[i] for i in sorted_errors]
        if len(aligned_reconstr)>=n_reconstructions:
            aligned_reconstr=aligned_reconstr[:n_reconstructions]
        else:
            log.info('Only {} reconstructions of the wanted {} satisfy error constraint. Averaging over viewer reconstructions than targeted.'.format(len(valid_errors),n_reconstructions))

        xprint('Averaging over {} aligned patterns with error smaller than {}!'.format(len(aligned_reconstr),alignment_error_limit))
        xprint('allignment errors are = {}\n'.format(np.sort(errors)))
            
        #average = [np.mean(d,axis = 0) for d in zip(*aligned_reconstr)]
        average = np.squeeze(np.mean(aligned_reconstr,axis = 0))
        aligned_ft_reconstr_ftd = [fft(r[0]) for r in aligned_reconstr]
        intensity_from_ft_density = np.mean([(r[1]*r[1].conj()).real for r in aligned_reconstr], axis = 0)
        intensity_from_density= np.mean([(r*r.conj()).real for r in aligned_ft_reconstr_ftd], axis = 0)
        input_reconstructions = [reference_reconstruction,*reconstructions]
        input_masks = [reference_mask,*masks]
        #log.info("{}, {}".format(len(input_reconstructions),len(input_masks)))

        centered_average = alignment.shift_to_center(*average)
        average_normalization_min = opt.get('average_normalization_min',False)

        xprint('calculating resolution metrics.')
        resolution_metrics = {}
        resolution_metrics = {}
        if opt.resolution_metrics.get('PRTF',False):
            prtf = PRTF_fxs(fft(average[0]),intensity_from_density,averaged_projected_scattering_amplitude=average[1],averaged_projected_intensity=intensity_from_ft_density)
            resolution_metrics['PRTF'] = prtf[0]
            resolution_metrics['PRTF_std'] = prtf[1]
            prtf_d = PRTF_fxs(fft(average[0]),intensity_from_density)
            resolution_metrics['PRTF_from_density'] = prtf_d[0]
            resolution_metrics['PRTF_from_density_std'] = prtf_d[1]
            prtf_ftd = PRTF_fxs(average[1],intensity_from_ft_density)
            resolution_metrics['PRTF_from_ft_density'] = prtf_ftd[0]
            resolution_metrics['PRTF_from_ft_density_std'] = prtf_ftd[1]

            prtf_ftI= PRTF_fxs(fft(average[0]),intensity_from_ft_density)
            resolution_metrics['PRTF_ftI'] = prtf_ftI[0]
            resolution_metrics['PRTF_ftI_std'] = prtf_ftI[1]
            #log.info('PRTF shape {} PRTF dtype = {}'.format(prtf[0].shape,prtf[0].dtype))
            #log.info('PRTF {}'.format(prtf[0]))
        if opt.resolution_metrics.get('FQCB',False):
            pass            
        if opt.resolution_metrics.get('pseudo_FSC',False):
            pseudo_fsc = FSC_single_fxs(fft(average[0]),average[1])
            resolution_metrics['pseudo_FSC']=pseudo_fsc
            fsc_limit = FSC_bit_limit(0.5,rgrid)
            resolution_metrics['FSC_0.5bit_limit']=fsc_limit
        xprint('done.\n')

        xprint('Saving results.\n')
        result = {
            'average':
            {
                'real_density':average[0],'normalized_real_density':self.normalize_density(average[0],d_min=average_normalization_min),'reciprocal_density':average[1],'intensity_from_densities':intensity_from_density,'intensity_from_ft_densities':intensity_from_ft_density
            },
            'resolution_metrics':resolution_metrics,
            'centered_average':
            {
                'real_density':centered_average[0],'normalized_real_density':self.normalize_density(centered_average[0],d_min=average_normalization_min),'reciprocal_density':centered_average[1]
            },
            "average_ids":valid_alignment_ids,
            "aligned":
            {
                #**{
                #'0':{'real_density':reference_reconstruction[0],'reciprocal_density':reference_reconstruction[0]},
                #'1':{'real_density':d,'reciprocal_density':a},
                #},
                
                **{
                    str(i):{'real_density':aligned_reconstr[i][0],'reciprocal_density':aligned_reconstr[i][1]} for i in range(len(aligned_reconstr))
                },
                
                #**{
                #str(i+2):{'real_density':reconstructions[i][0],'reciprocal_density':reconstructions[i][1]} for i in range(len(reconstructions))
                #}
            },
            'input':
            {
                str(i):{'real_density':input_reconstructions[i][0],'reciprocal_density':input_reconstructions[i][1],'support_mask':input_masks[i]} for i in range(len(input_reconstructions))
            },
            'input_meta':{
                'projection_matrices':projection_matrices,
                'reconstruction_keys':reconstructions_keys,
                'scaling_factors': np.array(scaling_factors),
                'average_scaling_factors_per_file':np.array(average_scaling_factors_per_file),
                'grids':ft_grids
            },
            'rotation_metric':
            {
                str(i+1): rotation_metrics[i] for i in range(len(rotation_metrics))         
            },
            'rotation_angles':
            {
                str(i+1): rotation_angles[i] for i in range(len(rotation_angles))         
            },
            'so3_grid': alignment.soft_grid,
        }
        self.post_processing(result,alignment.ft_grids,r_opt)
        result = 0
        return result,locals()

    
    def load_reconstructions(self):
        opt = self.opt
        db = self.db        
        error_metric = opt['selection']['error_metric']
        error_limit = opt['selection']['error_limit']        
        reconstructions = []
        reconstruction_ids_per_file = [[] for i in range(len(opt['reconstruction_files']))]
        reconstruction_keys = {}
        reconstruction_keys2 = {}
        support_masks = []
        errors = []
        file_start_ids = []
        projection_matrices=[]
        start_id = 0
        r_opt = False
        
        file_paths = [0]*len(opt['reconstruction_files'])
        for num,file_path in enumerate(opt['reconstruction_files']):
            file_path=file_path.format(today=db.get_time_string())
            file_paths[num]=file_path
        
        for num,file_path in enumerate(file_paths):
            xprint('\tfile {} of {}'.format(num+1,len(opt['reconstruction_files'])))
            with db.load('reconstructions',path_modifiers={'path':file_path},as_h5_object = True) as h5_file:
                file_start_ids.append(start_id)
                reconstruction_keys[num] = {}
                start_id += len(h5_file['reconstruction_results'])
                max_order = len(h5_file['projection_matrices'])
                pr_group = h5_file['projection_matrices']
                pr_mat = [pr_group[str(i)][:] for i in range(max_order)]
                projection_matrices.append(pr_mat)
                #log.info(start_id)
                for key,value in h5_file['reconstruction_results'].items():
                    error = value['error_dict'][error_metric][-1]
                    if error < error_limit:
                        reconstr = (value['last_real_density'][:],value['last_reciprocal_density'][:])
                        if self.valid_maximal_density(reconstr[0].real.max()):
                            errors.append(error)
                            reconstruction_keys[num][key] = len(reconstructions)
                            pos = len(reconstructions)
                            reconstruction_keys2[pos] = (num,key)
                            reconstruction_ids_per_file[num].append(pos)
                            reconstructions.append(reconstr)                            
                            support_masks.append(np.array(value['support_mask'][:],dtype=float))
                if num == 0:                    
                    r_opt = db.load('reconstruction_settings',path_modifiers= {'path':os.path.dirname(file_path)})
                    r_opt['internal_grid'] = db.get_db('file://_.h5').recursively_load_dict_from_group(h5_file,'/configuration/')['internal_grid']
        errors = np.array(errors)

        #sort reconstructions by error_metric
                                
                
        #support_masks = [ support_masks[rid] for rid in min_args]
        #errors = errors[min_args]
        
        xprint("\t{} loaded reconstruction with error metrics smaller than {}, error values are: \n\t{}".format(len(reconstructions),error_limit,errors))
        g_dict = r_opt['internal_grid']
        #log.info(g_dict.real_grid)
        ft_grids = FTGridPair(g_dict['real_grid'],g_dict['reciprocal_grid'])
        r_opt['internal_grid'] = ft_grids
        reference_arg = self.get_reference_arg(errors,reconstruction_keys)
        #log.info('reference reconstruction={}'.format(reconstruction_keys2[reference_arg]))
        reference_file_path = db.get_path('reconstructions',path_modifiers={'path':file_paths[reconstruction_keys2[reference_arg][0]]})
        xprint("\tselected reference: reconstruction {} in {}".format(reconstruction_keys2[reference_arg][1],reference_file_path))
        return reconstructions,support_masks,r_opt,errors,file_start_ids,reference_arg,reconstruction_keys2,projection_matrices,reconstruction_ids_per_file

    def extract_aligned_reconstructions(self,alignment_results):
        aligned_reconstructions = []
        for align_dict in alignment_results:
            aligned_reconstructions.append(align_dict['reconstruction'])
        return aligned_reconstructions

    def get_reference_arg(self,errors,lookup_dict):
        selection_method = self.opt['selection']['method']
        if selection_method == 'least_error':
            reconstruction_arg = np.argmin(errors)
        elif selection_method == 'manual':
            m_spec = self.opt['selection']['manual_specifier']
            reconstruction_arg = lookup_dict[m_spec[0]][str(m_spec[1])]
        return reconstruction_arg

    def valid_maximal_density(self,max_density):
        lower_bound,upper_bound = settings.project.selection.get('max_density_range',[None,None])   
        is_valid = True
        if isinstance(lower_bound,(int,float)):
            if max_density < lower_bound:
                is_valid=False
        if isinstance(upper_bound,(int,float)):
            if max_density > upper_bound:
                is_valid=False
        return is_valid

    def normalize_density(self,d,d_min=False):
        if isinstance(d_min,bool):
            d_min = d.real.min()
        #log.info('dmin = {}'.format(d_min ))
        d_max = np.max(d.real)
        d_normalized = (d-d_min)/(d_max-d_min)
        return d_normalized
    
class Alignment():
    def __init__(self,process_factory,db,opt):
        self.opt = opt['opt'] # alignment options
        self.r_opt = opt['r_opt'] # options of the reconstructions
        if self.r_opt['GPU'].get('use',False):
            if settings.general.n_control_workers ==0:
                settings.general.n_control_workers = self.r_opt['GPU'].get("n_gpu_workers",4)
                Multiprocessing.comm_module.restart_control_worker()
        self.ft_grids = self.r_opt.internal_grid
        self.dimension = self.ft_grids.realGrid[:].shape[-1]
        self.db = db
        self.process_factory = process_factory
        self._reference_reconstruction = 'not set'
        self.results = {}
        # Load the library that takes carre of rotations
        if self.dimension ==3:
            self.soft = get_soft_obj(self.r_opt['grid']['max_order']+1)
            self.soft_grid = self.soft.make_SO3_grid()
        self.sht = 'not set' #spherical harmonic transform object will be set by generate_operators
        # The next function call generates all necessary operators and adds them
        # to the process_factory.
        self.generate_operators()
        self.shift_to_center = self.assemble_shift_to_center().run
        self.shift_by = self.assemble_shift_by()
        if self.dimension ==3:
            self.rotate_by = self.assemble_rotate_by()
            self.align = self.assemble_align()
            self.apply_to = self.generate_alignment_loop()
        

    @property
    def reference_reconstruction(self):
        return self._reference_reconstruction
    @reference_reconstruction.setter
    def reference_reconstruction(self,value):
        assert (len(value)==2 and isinstance(value[0],np.ndarray) ),'Reference reconstruction needs to be a tuple of length 2 containing np.ndarrays for real_density and reciprocal_density but {} wsr given'.format(value)
        self._reference_reconstruction = value
        self._update_alignment()

    def _update_alignment(self):        
        self.align = self.assemble_align().run
        self.apply_to = self.generate_alignment_loop()
        

    def generate_operators(self):
        r_opt = self.r_opt
        if self.dimension ==3:
            self.harm_lm_split_ids, self.harm_ml_split_ids,self.sht = self.assemble_transform_op(r_opt)
        elif self.dimension ==2 :
            self.sht = self.assemble_transform_op(r_opt)
        find_center = generate_calc_center(self.ft_grids.realGrid)
        shift = generate_shift_by_operator(self.ft_grids.reciprocalGrid)
        negative_shift = generate_shift_by_operator(self.ft_grids.reciprocalGrid,opposite_direction = True)
        self.process_factory.addOperators({'find_center':find_center,'shift':shift,'negative_shift':negative_shift})
        if self.dimension ==3:
            rotate = self.generate_rotate()
            self.process_factory.addOperators({'rotate':rotate})


    def generate_fourier_transforms(self,grid_pair,ft_opt,harm_trf):
        opt = settings.project
        db = database.project
        dimensions = grid_pair.realGrid[:].shape[-1]
        ft_type = ft_opt['type']
        max_order=self.r_opt['grid']['max_order']
        harmonic_orders = np.arange(max_order+1)
        n_orders=len(harmonic_orders)
        reciprocity_coefficient = _get_reciprocity_coefficient(ft_opt)
        rs=grid_pair.realGrid[:].__getitem__((slice(None),)+(0,)*self.dimension) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        r_max = np.max(rs)
        n_radial_points = len(rs)
    
        name_postfix='N'+str(n_radial_points)+'mO'+str(max_order)+'nO'+str(n_orders)+'rc'+str(reciprocity_coefficient)
        try:
            weights_dict = db.load('ft_weights',path_modifiers={'postfix':name_postfix,'type':ft_type+'_'+str(dimensions)+'D'})
        except FileNotFoundError as e:
            weights_dict = generate_weightDict(max_order, n_radial_points,reciprocity_coefficient=reciprocity_coefficient,dimensions=dimensions,mode=ft_type)
        
        
        use_gpu = self.r_opt['GPU']['use']
        
        fourierTransform,inverseFourierTransform=generate_ft(r_max,weights_dict,harm_trf,dimensions,pos_orders=harmonic_orders,reciprocity_coefficient=reciprocity_coefficient,use_gpu = use_gpu,mode = ft_type)
        return fourierTransform,inverseFourierTransform

    def generate_fourier_transforms_old(self, grid_pair,ft_opt , harm_trf):                    
        db = database.project
        dimensions = grid_pair.realGrid[:].shape[-1]
        ft_type = ft_opt.type
        data_type = ft_opt.data_type
        max_order=self.r_opt['grid']['max_order']
        harmonic_orders = np.arange(max_order+1)
        n_orders=len(harmonic_orders)

        pi_in_q = ft_opt.get('pi_in_q',None)
        reciprocity_coefficient = _get_reciprocity_coefficient(ft_opt)

        rs=grid_pair.realGrid[:].__getitem__((slice(None),)+(0,)*self.dimension) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        r_max = np.max(rs)
        n_radial_points = len(rs)
            
        use_gpu = self.r_opt['GPU']['use']
        name_postfix='N'+str(n_radial_points)+'mO'+str(max_order)+'nO'+str(n_orders)+'rc'+str(reciprocity_coefficient)
        try:
            weights_dict = db.load('ft_weights',path_modifiers={'postfix':name_postfix,'type':ft_type+'_'+str(dimensions)+'D'})
        except FileNotFoundError as e:
            if ft_opt.allow_weight_calculation:
                weights_dict = generate_weightDict(max_order, n_radial_points,reciprocity_coefficient=reciprocity_coefficient,dimensions=dimensions,mode=ft_type)
                if ft_opt.allow_weight_saving:
                    db.save('ft_weights',weights_dict,path_modifiers={'postfix':name_postfix,'type':ft_type+'_'+str(dimensions)+'D'})

        
        fourierTransform,inverseFourierTransform=generate_ft(r_max,weights_dict,harm_trf,dimensions,pos_orders=harmonic_orders,reciprocity_coefficient=reciprocity_coefficient,use_gpu = use_gpu,mode = ft_type)

        #fourierTransform,inverseFourierTransform=generate_ft(r_max,weights_dict,harm_trf,dimensions,reciprocity_coefficient,use_gpu = use_gpu,mode = ft_type)
        
        return fourierTransform,inverseFourierTransform

    def assemble_transform_op(self,r_opt):
        ht_opt={
            'dimensions':self.dimension,
            **r_opt['grid']
        }
        # Create harmonic transforms (needs to happen before grid selection since harmonic transform plugin can choose the angular part of the grid.)
        #log.info('create harmonic transforms.')
        cht=HarmonicTransform('complex',ht_opt)
        #ht_data_type = opt['harmonic_transform']['data_type']
        if self.dimension == 2:
            ht =  HarmonicTransform('real',ht_opt)
        elif self.dimension == 3:
            ht=cht
        cht_forward,cht_inverse = cht.forward, cht.inverse
        ht_forward,ht_inverse = ht.forward, ht.inverse
        #log.info("ht grid param = {}".format(cht.grid_param))
        # fourier transforms
        ft_forward,ft_inverse=self.generate_fourier_transforms(self.ft_grids,r_opt['fourier_transform'],cht)

        transform_op={'fourier_transform':ft_forward,'inverse_fourier_transform':ft_inverse,'harmonic_transform':ht_forward,'inverse_harmonic_transform':ht_inverse,'complex_harmonic_transform':cht_forward,'complex_inverse_harmonic_transform':cht_inverse}
        # fourier transforms on SO(3)
        if self.dimension == 3:
            l_max= np.max(r_opt['grid']['max_order'])
            soft = mLib.get_soft_obj(l_max)
            harmonic_transpose_lm_to_ml, harmonic_transpose_ml_to_lm = self.generate_harmonic_transposes(cht)
            transform_op.update({'soft':soft.forward_cmplx,'inverse_soft':soft.inverse_cmplx,'lm_to_ml':harmonic_transpose_lm_to_ml,'ml_to_lm':harmonic_transpose_ml_to_lm})

        self.process_factory.addOperators(transform_op)
        if self.dimension == 3:
            return cht.l_split_indices,cht.m_split_indices,cht
        elif self.dimension ==2:
            return cht

    def assemble_align_parts(self):         
        find_shift = self.generate_find_shift()
        find_rotation = self.generate_find_rotation()
        save_shift = self.generate_save_shift()
        parts = {'find_shift':find_shift,'find_rotation':find_rotation,'save_shift':save_shift}        
        self.process_factory.addOperators(parts)
        
    def generate_harmonic_transposes(self,cht):
        m_indices_concat = np.concatenate(cht.m_indices)
        m_indices = cht.m_indices
        # also turns a single 1D numpy array into a tuple of 1D arrays
        def lm_to_ml(l_coeff):
            #log.info('l_coeff shape = {}'.format(l_coeff.shape))
            m_coeff = [l_coeff[:,index] for index in m_indices]
            return m_coeff
        # also turns tuple of 1Darrays into a single 1D numpy array
        def ml_to_lm(m_coeff):
            l_coeff = np.zeros((len(m_coeff[0]),cht.n_coeff),np.complex128)
            l_coeff[:,m_indices_concat] = np.concatenate(m_coeff,axis = 1)
            return l_coeff
        return lm_to_ml,ml_to_lm

    
    def generate_find_shift(self):
        ref_amplitude = self.reference_reconstruction[1]
        ift = self.process_factory.get_operator('inverse_fourier_transform')
        real_grid = self.ft_grids.realGrid
        def find_shift(ref_amplitude,amplitude):
            cc = ift(amplitude*ref_amplitude.conj())
            argmax = np.unravel_index(np.argmax(cc),cc.shape)
            shift = real_grid[argmax]
            #log.info('calculated_shift = {}'.format(shift))
            return shift
        return find_shift
    
    def generate_save_shift(self):
        def save_shift(shift):
            shift_list = self.results.get('shifts',[])
            shift_list.append(shift)
            self.results['shifts'] = shift_list
        return save_shift
            
    
    def generate_find_rotation(self):
        bw = self.soft.bw
        C_shape = (2*bw,)*3
        pi = np.pi
        calc_mean_C = self.soft.calc_mean_C
        lm_split_ids = self.harm_lm_split_ids
        angle_grid = self.soft_grid
        r_limit_ids = self.opt['find_rotation'].get('r_limit_ids', [0,self.ft_grids.realGrid.shape[0]])
        self.beta = 0
        def find_rotation(ref_lm_coeff,lm_coeff):            
            ref_lm_coeff = np.concatenate(ref_lm_coeff,axis = 1).copy()
            lm_coeff = np.concatenate(lm_coeff,axis = 1).copy()
            #print('limits = {}'.format(r_limit_ids))
            mean_C = calc_mean_C(lm_coeff,ref_lm_coeff,r_limit_ids,lm_split_ids).real
            self.results['rotation_metrics'].append(mean_C)
            #log.info('mean C shape = {},max = {}, min = {}'.format(mean_C.shape,np.max(mean_C),np.min(mean_C)))
            argmax = np.unravel_index(np.argmax(mean_C),C_shape)
            euler_angles = angle_grid[argmax[1],argmax[0],argmax[2]]
            euler_angles[0] = 2*pi - euler_angles[0]
            euler_angles[2] = 2*pi - euler_angles[2]
            self.results['rotation_angles'].append(euler_angles)
            #log.info('mean_C max ids = {},max_C = {}, min_C = {},dtype = {},mean = {},var = {}'.format(argmax,np.max(mean_C),np.min(mean_C),mean_C.dtype,np.mean(mean_C),np.median(mean_C)))
            #log.info('calculated_rotation angle = {}'.format(euler_angles/np.pi))
            return euler_angles
        return find_rotation    
    def generate_rotate(self):
        rotate_coeff = self.soft.rotate_coeff
        lm_split_ids = self.harm_lm_split_ids
        l_indices = self.sht.l_indices
        def rotate(lm_coeff,euler_angles):
            #log.info('euler_angles = {}'.format(euler_angles))
            lm_coeff = np.concatenate(lm_coeff,axis = 1)
            #log.info(lm_coeff.shape)
            rotated_coeff = rotate_coeff(lm_coeff,lm_split_ids,euler_angles)
            rotated_coeff = np.split(rotated_coeff,lm_split_ids,axis = 1)
            return rotated_coeff
        return rotate

    def assemble_align(self):
        self.assemble_align_parts()        
        # signal : the density to be aligned to the reference
        # reference : the reference density, reference_intensity
        rotate_signal_sketch =[
            ['complex_harmonic_transform','complex_harmonic_transform','complex_harmonic_transform'],
            [np.array([0,1,1,2],dtype=np.int64),['find_rotation','id','id']],
            [np.array([1,0,2,0],dtype=np.int64),['rotate','rotate']],
            ['complex_inverse_harmonic_transform','complex_inverse_harmonic_transform'],
        ]
        
        shift_reference_sketch = [
            [np.array([0,1],dtype=np.int64),['fourier_transform','fourier_transform']],
            [np.array([0,0,1],dtype=np.int64),['id','find_shift']],
            [np.array([0,1,1],dtype=np.int64),['shift','save_shift']],
            [np.array([0],dtype=np.int64),['inverse_fourier_transform']],
        ]
        rotate_signal = self.process_factory.buildProcessFromSketch(rotate_signal_sketch)
        shift_reference = self.process_factory.buildProcessFromSketch(shift_reference_sketch)
        self.process_factory.addOperators({'rotate_signal':rotate_signal,'shift_reference':shift_reference})
        
        align_sketch = [
            [np.array([0,1,2],dtype=np.int64),['rotate_signal']],
            #[np.array([0,1,1],dtype=np.int64),['shift_reference','id']],            
            # calc fourier transforms
            #[np.array([0,0,1,1],dtype=np.int64),['id','fourier_transform','id','fourier_transform']],
        ]
        
        align = self.process_factory.buildProcessFromSketch(align_sketch).run
        return align
    
    def assemble_align_old(self):
        self.assemble_align_parts()
        align_sketch = [
            'id',
            ['id','find_shift'],
            'shift',
            'inverse_fourier_transform',
            'complex_harmonic_transform',
            ['id','find_rotation'],
            'rotate',
            'complex_inverse_harmonic_transform',
            ['id','fourier_transform']                                  
        ]
        align_sketch = [
            # rotational alignment step
            ['complex_harmonic_transform','complex_harmonic_transform'],
            [np.array([0,0,1,1],dtype=np.int64),['id','find_rotation','id']],
            [np.array([0,2,1],dtype=np.int64),['id','rotate']],
            ['complex_inverse_harmonic_transform','complex_inverse_harmonic_transform'],
            [np.array([0,0,1,1],dtype=np.int64),['id','fourier_transform','id','fourier_transform']]
        ]
        align = self.process_factory.buildProcessFromSketch(align_sketch).run
        return align
    def assemble_shift_to_center(self):
        #start with reciprocal_density,real_density
        shift_to_center_sketch_old = [
            ['find_center','id'],
            [np.array([1,0,0],dtype=int),['negative_shift','id']],         
            [np.array([0,0,1],dtype=int),['inverse_fourier_transform','id','id']]
        ]
        shift_to_center_sketch = [
            [(0,0,1),['find_center','fourier_transform','id']],
            [(1,0,2,0,0),['negative_shift','negative_shift','id']],         
            [(0,1,2),['inverse_fourier_transform','id','id']]
        ]
        shift_to_center = self.process_factory.buildProcessFromSketch(shift_to_center_sketch)
        return shift_to_center
    def assemble_rotate_by(self):
        rotate_by_sketch = [
            ['complex_harmonic_transform','id'],
            'rotate',
            'complex_inverse_harmonic_transform',
            ['id','fourier_transform']
        ]
        rotate_by = self.process_factory.buildProcessFromSketch(rotate_by_sketch).run
        return rotate_by
    def assemble_shift_by(self):
        shift_by_sketch = [
            ['fourier_transform','id'],
            'shift',
            ['inverse_fourier_transform','id']
        ]
        shift_by = self.process_factory.buildProcessFromSketch(shift_by_sketch).run
        return shift_by
    
    def generate_alignment_loop(self):
        align = self.align
        max_iterations = self.opt['max_iterations']
        error_limit = self.opt['alignment_error_limit']
        integrate = mLib.SphericalIntegrator(self.ft_grids.realGrid[:]).integrate_normed
        shift = self.process_factory.get_operator('shift')
        ft = self.process_factory.get_operator('fourier_transform')
        ift = self.process_factory.get_operator('inverse_fourier_transform')
        self.results['shifts']=[np.array([0,0,0])]
            
        def alignment_loop(reference, signal):
            self.results['rotation_metrics']=[]
            self.results['rotation_angles']=[]
            ref = reference.copy()
            sig = signal[0]
            ft_sig = signal[1]
            ref_norm = integrate(ref.real**2)
            if ref_norm == 0:
                ref_norm = 1
            errors = []
            for i in range(max_iterations):
                sig,ft_sig = align(ref,sig,ft_sig)
                
                #calc error
                diff = ref.real - sig.real
                error = integrate(diff**2)/ref_norm
                errors.append(error)
                last_shift = self.results['shifts'][-1]
                #log.info('\n error = {} last_shift = {} \n'.format(error,last_shift))
                # brake the loop if shift is 0 or error limit is reached
                if (error<error_limit) or (last_shift[0]==0):
                    break

            if False: #last_shift[0] != 0:
                # calculate total shift of reference
                cart_shift = np.sum(np.array([mLib.spherical_to_cartesian(shift) for shift in self.results['shifts']]),axis = 0)
                # apply corresponding shift to signal            
                shift = mLib.cartesian_to_spherical(-cart_shift)
                sig = self.shift_by(sig,shift) #returns (density, ft(density))
            else:
                sig = [sig,ft_sig]
            out = {'densities':sig,'errors':errors,'shifts':self.results['shifts'],'rotation_metrics':self.results['rotation_metrics'],'rotation_angles':self.results['rotation_angles']}
            return out
        def alignment_routine(reference,signal):
            #log.info("type {}".format(type(signal)))
            #log.info("len {}".format(len(signal)))            
            #signal_inverted = (ift(signal[1].conj()),signal[1].conj())
            signal_inverted = (ift(ft(signal[0]).conj()),signal[1].conj())
            
            out = alignment_loop(reference,signal)
            out_inverted = alignment_loop(reference,signal_inverted)
            reference_norm = integrate(reference.real**2)
            diff = reference.real - out['densities'][0].real
            diff_inverted = reference.real - out_inverted['densities'][0].real
            error = integrate(diff**2)/reference_norm
            error_inverted = integrate(diff_inverted**2)/reference_norm
            #log.info('error = {} inverted error ={}'.format(error,error_inverted))
            if error < error_inverted:
                return_dict = out
            else:
                return_dict = out_inverted
            #log.info('normal error = {} inverted error = {},signal max = {} reference max = {}'.format(error,error_inverted,signal[0].real.max(),reference[0].real.max()))
            return_dict['diff_norm']=diff
            return_dict['diff_inverted_norm']=diff_inverted
            return return_dict
        return alignment_routine
