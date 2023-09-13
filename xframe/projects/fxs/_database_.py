import h5py as h5
import numpy as np
import os
from os.path import join as pjoin
from os import path as op
import time
import logging
import traceback
import re
import glob

from xframe.interfaces import DatabaseInterface,ExperimentWorkerInterface,ProjectWorkerInterface
from xframe.library.mathLibrary import plane3D
from xframe.library.mathLibrary import spherical_to_cartesian
#from xframe.projectRecipes import analysisLibrary as aLib
from xframe.library.gridLibrary import GridFactory
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import double_first_dimension
from xframe.library.pythonLibrary import getArrayOfArray
from xframe.library.pythonLibrary import option_switch
from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe.library.physicsLibrary import ewald_sphere_theta_pi
from xframe.database.database import DefaultDB
from .projectLibrary.classes import FXS_Data
from .projectLibrary.classes import FTGridPair
from .projectLibrary.classes import SampledFunction
from .projectLibrary.fxs_invariant_tools import ccd_associated_legendre_matrices_single_m
from .projectLibrary.fxs_invariant_tools import deg2_invariant_to_cn_3d
from .projectLibrary.fxs_invariant_tools import harmonic_coeff_to_deg2_invariants
from .projectLibrary.misk import _get_reciprocity_coefficient
from .projectLibrary.harmonic_transforms import HarmonicTransform

from xframe.presenters.matplotlibPresenter import heat2D_multi
from xframe.presenters.matplotlibPresenter import plot1D
from xframe.presenters.openCVPresenter import Polar2D
from xframe import settings
import xframe


log=logging.getLogger('root')
        
class ProjectDB(DefaultDB,DatabaseInterface):
    def __init__(self,**folders_files):
        super().__init__(**folders_files)


    def get_time_string(self):
        time_struct=time.gmtime()
        time_str=str(time_struct[2])+'_'+str(time_struct[1])+'_'+str(time_struct[0])
        return time_str
    def get_reconstruction_path(self):
        time_str=self.get_time_string()
        path=self.folders[self.files['reconstructions']['folder']]
        run= self.get_latest_run('reconstructions',path_modifiers={'time':time_str}) + 1
        run_path=path.format(time=time_str,run=run)
        return run_path
    
    def get_latest_run(self,folder_name,path_modifiers = {}):
        '''
        Searches subfolders for increasing run number.
        Stops if either the folder does not exist or does not contain a file ending with .h5 
        '''
        if folder_name in self.files:
            path=self.folders[self.files[folder_name]['folder']]
        else:
            path = folder_name
        
        run=0
        path_modifiers['run']=run
        run_path=path.format(**path_modifiers)
        log.info(f'run_path={run_path}')
        while os.path.exists(run_path):
            #print(run_path)

            h5_found = False
            for file in os.listdir(run_path):
                # check the files which are end with specific extension
                if file.endswith(".h5"):
                    h5_found=True

            if h5_found:
                run+=1
                path_modifiers['run']=run
                run_path=path.format(**path_modifiers)
            else:
                break
        return run-1

    def save_average_results(self,name,data,**kwargs):
        grid_pair = data['internal_grid']
        real_grid = grid_pair["real_grid"]
        reciprocal_grid = grid_pair["reciprocal_grid"].copy()
        dimension = int(real_grid[:].shape[-1])
        
        
        options = kwargs
        log.info(options)
        time_struct=time.gmtime()
        time_str=str(time_struct[2])+'_'+str(time_struct[1])+'_'+str(time_struct[0])
        path=self.folders[self.files['average_results']['folder']]
        path_modifiers={'time':time_str,'structure_name':settings.project.structure_name,'dimensions':dimension}
        run= self.get_latest_run('average_results',path_modifiers=path_modifiers) + 1
        path_modifiers['run']=run
        run_path=path.format(**path_modifiers)
        log.info('run_path = {}'.format(run_path))
        path = self.get_path('average_results', path_modifiers=path_modifiers)
        self.save(path,data)

        if options.get('save_settings',True):
            path = os.path.dirname(self.get_path('average_results', path_modifiers=path_modifiers))
            self._save_settings(path)

        grid_pair = data['internal_grid']
        real_grid = grid_pair["real_grid"]
        reciprocal_grid = grid_pair["reciprocal_grid"].copy()
        dimension = int(real_grid[:].shape[-1])
        qs = reciprocal_grid[:].__getitem__((slice(None),)+(int(0),)*dimension)
        #reciprocity_coefficient = data['reciprocity_coefficient']
        #reciprocal_grid[...,0]*= (np.pi/reciprocity_coefficient) 
        vtk_saver = self.get_db('file://vtk').save

        if options['generate_average_vtk']:
            try:
                real_density = data['average']['real_density']
                normalized_real_density = data['average']['normalized_real_density']
                reciprocal_density = data['average']['reciprocal_density']

                if dimension == 2:
                    grid_type = 'polar'
                elif dimension == 3:
                    grid_type = 'spherical'
                real_vtk_path=self.get_path('real_vtk',path_modifiers={**path_modifiers,'reconstruction':'average'})
                self.save(real_vtk_path,[real_density.real,normalized_real_density.real],grid = real_grid,dset_names=['density','normalized_density'],grid_type=grid_type)
                #vtk_saver([real_density],real_grid,real_vtk_path,dset_names = ['density'],grid_type='spherical')

                reciprocal_vtk_path=self.get_path('reciprocal_vtk',path_modifiers={**path_modifiers,'reconstruction':'average'})
                self.save(reciprocal_vtk_path,[np.abs(reciprocal_density).real**2],grid = reciprocal_grid,dset_names=['amplitude'],grid_type=grid_type)
                #vtk_saver([reciprocal_density],reciprocal_grid,reciprocal_vtk_path, dset_names = ['amplitude'],grid_type='spherical')
            except Exception as e:
                traceback.print_exc()
                log.error('Failed to generate aligned vtk.')

                
        if options['generate_aligned_vtk']:
            try:
                for key,result in data['aligned'].items():
                    real_density = result['real_density']
                    #log.info('averaged shape = {}'.format(real_density.shape))
                    #log.info('averaged grid shape = {}'.format(real_grid.shape))
                    reciprocal_density = result['reciprocal_density']
                    if dimension == 2:
                        grid_type = 'polar'
                    elif dimension == 3:
                        grid_type = 'spherical'                        
                    #vtk_saver([real_density],real_grid,real_vtk_path, dset_names = ['density'],grid_type='spherical')
                    real_vtk_path=self.get_path('real_vtk',path_modifiers={'time':time_str,'run':run,'reconstruction':'aligned_'+key})
                    #log.info(real_vtk_path)
                    self.save(real_vtk_path,[real_density.real],grid = real_grid,dset_names=['density'],grid_type=grid_type)
                    reciprocal_vtk_path=self.get_path('reciprocal_vtk',path_modifiers={**path_modifiers,'reconstruction':'aligned_'+key})
                    self.save(reciprocal_vtk_path,[np.abs(reciprocal_density).real**2],grid = reciprocal_grid,dset_names=['amplitude'],grid_type=grid_type)
            except Exception as e:
                traceback.print_exc()
                log.error('Failed to generate aligned vtk.')                        

        log.info('input vtk is set to = {}'.format(options['generate_input_vtk']))
        if options['generate_input_vtk']:
            try:
                log.info('len input data = {}'.format(len(data['input'].items())))
                for key,result in data['input'].items():
                    #log.info("key = {}".format(key))
                    real_density = result['real_density']
                    support_mask = result['support_mask']
                    reciprocal_density = result['reciprocal_density']

                    if dimension == 2:
                        grid_type = 'polar'
                    elif dimension == 3:
                        grid_type = 'spherical'
                    
                    real_vtk_path=self.get_path('real_vtk',path_modifiers={**path_modifiers,'reconstruction':'input_'+key})
                    self.save(real_vtk_path,[real_density.real,support_mask],grid = real_grid,dset_names=['density','mask'],grid_type=grid_type)
                    #vtk_saver([real_density,support_mask],real_grid,real_vtk_path, dset_names = ['density','mask'],grid_type='spherical')

                    reciprocal_vtk_path=self.get_path('reciprocal_vtk',path_modifiers={**path_modifiers,'reconstruction':'input_'+key})
                    #vtk_saver([reciprocal_density],reciprocal_grid,reciprocal_vtk_path , dset_names = ['amplitude'],grid_type='spherical')
                    self.save(reciprocal_vtk_path,[np.abs(reciprocal_density).real**2],grid = reciprocal_grid,dset_names=['amplitude'],grid_type=grid_type)
            except Exception as e:
                traceback.print_exc()
                log.error('Failed to generate input vtk.')
                
        if options['generate_rotation_metric_vtk']:
            try:
                if dimension == 3:
                    for key,metrics in data['rotation_metric'].items():
                        for n,metric in enumerate(metrics):
                            vtk_path = self.get_path('rotation_metric_vtk',path_modifiers={**path_modifiers,'reconstruction':key,'iteration':str(n+1)})
                            self.save(vtk_path,[metric],grid = data['so3_grid'],dset_names=['rotation_metric'],grid_type='cartesian')
                            #vtk_saver([metric],data['so3_grid'],vtk_path,dset_names = ['rotation_metric'],grid_type='cartesian')
            except Exception as e:
                traceback.print_exc()
                log.error('Failed to generate rotation metric vtk.')
        if options['plot_resolution_metrics']:
            try:
                for key,metrics in data['resolution_metrics'].items():
                    if key == "PRTF":
                        layout = {'title':'PRTF','x_label':'q [\AA$^{-1}$]','y_label':'PRTF [arb.]'}                        
                        fig = plot1D.get_fig(np.abs(metrics),grid = qs,ylim=[0.0,1.1],layout = layout,labels=['PRTF'])
                        fig_path = run_path + 'PRTF.matplotlib'
                        self.save(fig_path,fig)
            except Exception as e:
                traceback.print_exc()
                log.error('Failed to generate resolution metric plots.')

    def _save_settings(self,path,project_name='settings',exp_name='exp_settings'):
        project_settings, experiment_settings = settings.get_settings_to_save()
        if isinstance(xframe.project_worker,ProjectWorkerInterface):
            self.save(pjoin(path,project_name+'.yaml'),project_settings)
        if isinstance(xframe.experiment_worker,ExperimentWorkerInterface):
            self.save(pjoin(path,exp_name+'.yaml'),experiment_settings)
        #log.info(f'settings raw = \n {settings.raw_analysis} \n')
    
    def save_reconstructions(self,name,data,**kwargs):
        options = self.files['reconstructions']['options']
        time_str=self.get_time_string()
        path=self.folders[self.files['reconstructions']['folder']]
        path_modifiers={'time':time_str,'structure_name':settings.project.structure_name,'dimensions':settings.project.dimensions}     
        run= self.get_latest_run('reconstructions',path_modifiers=path_modifiers) + 1
        path_modifiers['run']=run
        run_path=path.format(**path_modifiers)

        internal_grids = data['configuration']['internal_grid']
        real_grid  = internal_grids['real_grid'].copy()
        reciprocal_grid  = internal_grids['reciprocal_grid'].copy()
        q_radial_points = reciprocal_grid.__getitem__((slice(None),)+(0,)*settings.project.dimensions)
        #log.info(reciprocal_grid[:,0,0])
        #reciprocity_coefficient = data['configuration']['reciprocity_coefficient']
        #log.info('reciprocity_coefficient in saving = {}'.format(reciprocity_coefficient))
        #reciprocal_grid[...,0]*=(np.pi/reciprocity_coefficient)
        
        log.info('run_path = {}'.format(run_path))
        if 'projection_matrices' in data:
            p_matrices = data['projection_matrices']

        ####### Data & Settings #######
        self.save('reconstructions',data,skip_custom_methods=True,path_modifiers=path_modifiers)
        self._save_settings(run_path)

        

        ####### Error Metrics #######
        try:
            if options.get('plot_error_metrics',False):
                errors={}
                for key,value in data['reconstruction_results'].items():
                    errors[key] = value['error_dict']['main'][2:]
                error_fig = plot1D.get_fig(np.array(tuple(errors.values()))[::-1],labels = tuple(errors.keys())[::-1],y_scale='log',layout={'title':'Mean of Relative projection differences in L2 norm.','x_label':'loop step','y_label':'$|| rho-rho_{projected}||^2_{l_2}/ || rho ||^2_{l_2}$'})
                error_path = run_path + 'error_metrics.matplotlib'
                self.save(error_path,error_fig)
        except Exception as e:
            log.warning(f'Faild to save error metrics with error:\n {e}')
            log.info(traceback.format_exc())

        try:
            plot_invariant_error,err_orders = option_switch(options.get('plot_invariant_error',False),[0,2,4])
            if plot_invariant_error:
                r = data['reconstruction_results']
                log.info('contents are as follows = {}'.format([d['error_dict'].keys() for d in r.values()]))
                errors = np.moveaxis(np.array([d['error_dict']['reciprocal']['deg2_invariant_l2_diff'] for d in r.values()]),2,0)
                for order in err_orders:
                    if order in settings.project.projections.reciprocal.used_order_ids:
                        fig = plot1D.get_fig(errors[order,::-1,2:],y_scale = 'log',layout={'title':'Relative B{} errors.'.format(order),'x_label':'loop step','y_label':'$\sum_{q_1,q_2}(B_l-B_l^{data})^2/(B_l^{data})^2$'})
                        error_path = run_path + 'B{}_errors.matplotlib'.format(order)
                        self.save(error_path,fig)
                mean_error = np.mean(errors[::2],axis = 0)
                fig = plot1D.get_fig(mean_error[::-1,2:],y_scale = 'log',layout={'title':'Relative Bl errors mean.','x_label':'loop step','y_label':'$\sum_{q_1,q_2}(B_l-B_l^{data})^2/(B_l^{data})^2$'})
                error_path = run_path + 'Bl_mean_error.matplotlib'.format(order)
                self.save(error_path,fig)
        except Exception as e:
            log.warning(f"Failed to save invariant error plots with error {e}")
            log.info(traceback.format_exc())

        ####### VTK Files & Pics #######
        try:  
            generate_vtk_files,vtk_slice = option_switch(options['generate_vtk_files'],slice(None))
            if generate_vtk_files:                
                log.info('start constructing/saving vtk files')
                n_reconstructions = len(data['reconstruction_results'])
                ids = np.array(tuple(data['reconstruction_results'].keys()))
                good_ids= ids[vtk_slice]
                worst_id = [ids[-1]]
                if len(ids)==1:
                    worst_id=[]
                    ids_to_plot = np.asarray(good_ids)
                else:
                    ids_to_plot = np.unique(np.concatenate((good_ids,worst_id)))
                
                for id in ids_to_plot:
                    result = data['reconstruction_results'][id]
                    real_density = result['real_density'].real
                    real_mask = result['support_mask']
                    initial_density = result['initial_density'].real
                    initial_support = result['initial_support']
                    last_real_density = result['last_real_density'].real
                    last_real_mask = result['last_support_mask']
                    reciprocal_intensity = np.abs(result['reciprocal_density']).real
                    last_reciprocal_intensity = np.abs(result['last_reciprocal_density']).real

                    _id_str = str(id)
                    if id in worst_id:
                        _id_str = f'{id}_worst_error'
                    vtk_path_modifiers = {**path_modifiers,**{'reconstruction':_id_str}}
                    real_vtk_path=self.get_path('real_vtk',path_modifiers=vtk_path_modifiers)

                    if settings.project["dimensions"] == 3:

                        self.save(real_vtk_path,[real_density,real_mask,last_real_density,last_real_mask,initial_density,initial_support],grid =real_grid,grid_type='spherical',skip_custom_methods=True,dset_names=['best_density','best_mask','last_density','last_mask','initial_density','initial_support'])
                        #save_vtk([real_density,real_mask],real_grid,real_vtk_path,grid_type='spherical')
                         
                        reciprocal_vtk_path=self.get_path('reciprocal_vtk',path_modifiers=vtk_path_modifiers)
                        self.save(reciprocal_vtk_path,[reciprocal_intensity,last_reciprocal_intensity],grid = reciprocal_grid,grid_type='spherical',skip_custom_methods=True,dset_names=['best_intensity','last_intensity'])
                    elif settings.project["dimensions"] == 2:
                        self.save(real_vtk_path,[real_density,real_mask,last_real_density,last_real_mask,initial_density,initial_support],grid =real_grid,grid_type='polar',skip_custom_methods=True,dset_names=['best_density','best_mask','last_density','last_mask','initial_density','initial_support'])
                        #save_vtk([real_density,real_mask],real_grid,real_vtk_path,grid_type='spherical')
                         
                        reciprocal_vtk_path=self.get_path('reciprocal_vtk',path_modifiers=vtk_path_modifiers)
                        self.save(reciprocal_vtk_path,[reciprocal_intensity,last_reciprocal_intensity],grid = reciprocal_grid,grid_type='polar',skip_custom_methods=True,dset_names=['best_intensity','last_intensity'])
                        #log.info(reciprocal_grid[:,0,0])
                         
        except Exception as e:
            log.warning(f"Failed to save vtk_plots with error \n{e}")
            log.info(traceback.format_exc())
            
        try:
            generate_2d_images,img_slice = option_switch(options.get('generate_2d_images',False),slice(None))
            if generate_2d_images:                
                log.info('start constructing & saving polar images')
                for id,result in tuple(data['reconstruction_results'].items())[img_slice]:
                    self._save_polar_densities(id,result,path_modifiers)
        except Exception as e:
            log.warning(f"Failed to save 2D polar images with error\n {e}")
            log.info(traceback.format_exc())
            
        ####### Deg 2 invariants & Rest #######
        try:
            if options.get('plot_last_fqc_error',False):
                log.info('\n \n \n  trying to plot errors ')
                r = data['reconstruction_results']
                errors = np.array([d['error_dict']['reciprocal']['fqc_error'] for d in r.values()])
                fig = plot1D.get_fig(1-errors[:,-1,:],y_scale = 'lin',layout={'title':'FQC','x_label':'q','y_label':'FQC'})
                error_path = run_path + 'last_fqc.matplotlib'
                self.save(error_path,fig)
        except Exception as e:
            log.warning(f"Failed to save fqc error plots with error\n {e}")
            log.info(traceback.format_exc())
                                
        try:
            plot_reconstructed_deg2_invariants,deg2_slice = option_switch(options.get('plot_reconstructed_deg2_invariants',False),slice(None))
            if plot_reconstructed_deg2_invariants:
                for key,value in tuple(data['reconstruction_results'].items())[deg2_slice]:
                    Bls = value['last_deg2_invariant']
                    log.info("last deg2 invariant shape = {}".format(Bls.shape))
                    self._save_first_invariants(Bls,q_radial_points,run_path,options,name="{}_out_".format(key))


                    #thetas = ewald_sphere_theta_pi(data['configuration']['xray_wavelength'],q_radial_points)
                    #PlmPlm = np.moveaxis(np.squeeze(ccd_associated_legendre_matrices_single_m(thetas,len(p_matrices)-1,0)),-1,0)
                    #PlmPlm[np.isnan(PlmPlm)]=0
                    #PlmPlm[0]=0
                    #C0 = np.abs(np.sum(Bls*PlmPlm,axis = 0))
                    #self._save_C0(C0,q_radial_points,run_path,options,name="{}_out_".format(key))
        except Exception as e:
            log.warning(f"Failed to save invariant plots with error \n {e}")
            log.info(traceback.format_exc())
        try:
            if options.get('plot_first_used_invariants',False):
                if settings.project["dimensions"] == 3:
                    Bls = np.array([m @ m.T.conj() for m in p_matrices])
                elif settings.project["dimensions"] == 2:
                    Bls = np.array([m[:,None] * m.T.conj()[None,:] for m in p_matrices])
                #C0 = np.abs(Bls[0]-value['last_deg2_invariant'][0])**2/np.abs(Bls[0])**2
                #log.info('B[0] errors mean = {} = {}'.format(np.mean(C0),C0))
                #self._save_C0(C0,q_radial_points,run_path,options,name="Bl0_")
                self._save_first_invariants(Bls,q_radial_points,run_path,options,name="first_")               
        except Exception as e:
            log.warning(f"Failed to save initial invariant plots with error {e}")
            log.info(traceback.print_exc())
      
            
    def load_reconstruction_settings(self,name,**kwargs):
        path = self.get_path(name,path_modifiers=kwargs['path_modifiers'])        
        settings, raw_settings = self.load('settings',direct_path=path)
        return settings
    
    def _save_first_invariants(self,bls,radial_points,base_path,options,name='',mask = True):
        bls = bls.copy()
        bls[~mask]=0
        try:
            assert not np.isnan(bls).any(),'Nan values detected during plotting setting them to -1.'
        except:
            bls[np.isnan(bls)]=-1
            
        if isinstance(mask,np.ndarray):
            max_value = max(np.max(np.abs(bls[mask].flatten())),1e-100)
            median_value = np.median(np.abs(bls[::2][mask[::2]].flatten()))
        else:
            max_value = max(np.max(np.abs(bls.flatten())),1e-100)
            median_value = np.median(np.abs(bls[::2].flatten()))
        #log.info(f'median {median_value} max = {max_value}')
        if (median_value>0) and (median_value!=np.nan) and (max_value<np.inf):
            orders = int(np.log10(max_value/median_value)*0.8)
            vmin,vmax = options.get('plot_range',[median_value*10**(-orders//2),median_value*10**orders])
        elif (max_value==np.inf) and (median_value>0) and (median_value!=np.nan):            
            orders = 12
            vmin,vmax = options.get('plot_range',[median_value*10**(-orders//2),median_value*10**orders])
        else:            
            orders = 12
            vmin,vmax = options.get('plot_range',[max_value*10**(-orders//2),max_value])
            #log.info(f'vmin vmax = {vmin},{vmax}')
        #log.info(f'vmin vmax = {vmin}, {vmax} median = {median_value} max = {max_value}')
        
        grid = GridFactory.construct_grid('uniform',[radial_points,radial_points])
        order_ids = [0,2,4,6,8]

        shape = [3,len(order_ids)]
        layouts = []
        plot_data = []
        #log.info('bls shape = {} grid shape = {}'.format(bls.shape,grid.shape))
        for i in range(shape[0]):
            layout_part = []
            plot_data_part = []
            orders = np.arange(10*i,10*i+len(order_ids)*2,2)
            for o in orders:
                layout = {'title':'$B_{'+'{}'.format(o)+'}$',
                          'x_label':'$q_1$',
                          'y_label':'$q_2$'
                          }
                layout_part.append(layout)
                try:
                    plot_data_part.append(np.abs(bls[o]))
                except IndexError:
                    plot_data_part.append(np.zeros_like(bls[0],dtype = float))
                    log.info('IndexError')
                    
            layouts.append(layout_part)
            plot_data.append(plot_data_part)
                                                    
        fig_bl_masks = heat2D_multi.get_fig(plot_data,scale = 'log',layout = layouts,grid =grid[:],shape = shape,size = (30,shape[0]*5),vmin= vmin, vmax = vmax,cmap='plasma')
        bl_path = base_path +name+'Bl.matplotlib'
        self.save(bl_path,fig_bl_masks,dpi = 300)
    def _save_C0(self,c0,radial_points,base_path,options,name=''):
        max_value = np.abs(c0).max()
        vmin,vmax = options.get('plot_range_C1',[max_value*1e-12,max_value])
        if [vmin,vmax] == [None,None]:
            vmin,vmax = [max_value*1e-12,max_value]
    
        grid = GridFactory.construct_grid('uniform',[radial_points,radial_points])
        shape = [1,1]
        layout = {'title':'$C_0(q_1,q_2)$',
                  'x_label':'$q_1$',
                  'y_label':'$q_2$'
                  }
                                                    
        fig_bl_masks = heat2D_multi.get_fig(c0,scale = 'log',layout = [[layout]],grid =grid[:],size = (10,10),vmin= vmin, vmax = vmax,cmap='plasma')
        bl_path = base_path +name+'C0.matplotlib'
        self.save(bl_path,fig_bl_masks,dpi = 300)

    def _save_polar_densities(self,id,result,path_modifiers):
        #density_path=self.get_path('result_image_density',path_modifiers={'time':time_str,'run':run,'reconstruction':id})
        #intensity_path=self.get_path('result_image_density',path_modifiers={'time':time_str,'run':run,'reconstruction':id})
        pic_path=self.get_path('result_image',path_modifiers={**path_modifiers,'reconstruction':id})
        last_real_density = result['last_real_density'].real
        last_real_mask = result['last_support_mask']
        last_reciprocal_intensity = np.abs(result['last_reciprocal_density']).real
        dpic = Polar2D.get_fig(last_real_density,transparent_backgound=True,colormap='inferno',print_colorscale=True,n_pixels=512)
        ipic = Polar2D.get_fig(np.abs(last_reciprocal_intensity),transparent_backgound=True,colormap='inferno',scale='log', print_colorscale=True,n_pixels=512)
        #log.info('ipic shape {} ipic type {}'.format(ipic.shape,ipic.dtype))
        #log.info('dpic shape {} dpic type {}'.format(dpic.shape,dpic.dtype))
        out_pic = np.concatenate((dpic,ipic),axis=1)
        #log.info(pic_path)
        self.save(pic_path,out_pic)
    def load_ccd(self,name,**kwargs):
        options = self.files['ccd'].get('options',{})        
        _type=options['type']
        input_data=self.load(name,**{'skip_custom_methods': True,**kwargs})
        if (_type == 'legacy'):
            data=self.load_ccd_legacy(input_data)
        elif _type == 'direct':
            data = self.load_ccd_direct(input_data) 
        else:
            e = AssertionError('ccd loading type {} not specifierd or known!'.format(_type))
            log.error(e)
            raise e
        return data

    def load_ccd_legacy(self,data):
        out_dict={}
        if 'intra' in data:
            cc=data['intra']['ccf_2p_q1q2'].real
        else:
            cc=data['ccf_q1q2_2p'].real
        if cc.shape[0] < cc.shape[1]:            
            qs=data['q2']#/(2*np.pi)
            step_size = int(np.round(cc.shape[1]/cc.shape[0]))
            cc = cc[:,::step_size]
            a_int=data['iaverage'][::step_size]
        elif cc.shape[0] > cc.shape[1]:            
            qs=data['q1']#/(2*np.pi)
            step_size = np.round(cc.shape[0]/cc.shape[1])
            cc = cc[::step_size,:]
            a_int=data['iaverage'][::step_size]
        else:
            qs=data['q1']#/(2*np.pi)
            a_int=data['iaverage']

        other_cc = {}
        for key,name in zip(['ccf_q1q2_3p','ccf_q1q2_4p'],['I2I1','I2I2']):
            if key in data:
                other_cc[name] = data[key].real
        #cc[0,0]=np.mean(cc[0,0]) # test purposes
        out_dict['cross_correlation'] = {**{'I1I1':cc},**other_cc}
        #log.info('iaverage = {}'.format(a_int))
        out_dict['average_intensity']=a_int

        #qs[0] = 0 #test purposes
        out_dict['radial_points']=qs
        out_dict['qs']=qs
        #log.info('\n MAX Q={}'.format(qs[-1]))
        phis = data['phi']
        out_dict['angular_points']=phis
        out_dict['phis']=phis        
        out_dict['pi_in_q']=data.get('pi_in_q',True)
        
        q_phi_grid=GridFactory.construct_grid('uniform',[qs,phis])
        #            log.info('aInt shape={}'.format(aIntGrid.shape))
        #log.info('aInt grid dimension={}'.format(q_phi_grid.total_shape[-1]))
        #            log.info('first + last qs: \n{}  | {}'.format(q[:4],q[-4:]))
        #            log.info('first + last phis: \n{}  | {}'.format(phi[:4],phi[-4:]))
        #            stepLength={'q':q[2]-q[1],'phi':phi[2]-phi[1]}
        #            domains={'q':[q[0],q[-1]],'phi':[0,2*np.pi+stepLength['phi']]}
        out_dict['average_intensity'] = SampledFunction(NestedArray(qs[:,None],1),a_int,coord_sys='cartesian')
        out_dict['xray_wavelength']=data.get('xray_wavelength',1.23984) # in angström
        if out_dict['pi_in_q']:            
            thetas = ewald_sphere_theta_pi(out_dict['xray_wavelength'],qs)
        else:
            thetas = ewald_sphere_theta(out_dict['xray_wavelength'],qs)
        out_dict['thetas'] = thetas
        out_dict['data_grid'] = {'qs':qs,'thetas':thetas,'phis':phis}
        out_dict['dimensions'] = settings.project.dimensions
        return out_dict

    def load_ccd_direct(self,data):
        qs = data['radial_points']
        phis = data['angular_points']
        grid=GridFactory.construct_grid('uniform',[qs,phis])
        a_int = data['average_intensity']
        average_intensity = SampledFunction(NestedArray(grid[:,0,0],1),a_int,coord_sys='cartesian')
        data['average_intensity'] = average_intensity
        log.info(f'data = {data.keys()}')
        thetas = ewald_sphere_theta_pi(data['xray_wavelength'],qs)
        data['data_grid'] = {'qs':qs,'thetas':thetas,'phis':phis}
        data['dimensions'] = settings.project.dimensions
        #log.info(data['cross_correlation']['I2I2'])
        return data

    def load_invariants(self,name,**kwargs):
        data = self.load_direct(name,**kwargs)
        if isinstance(data['data_projection_matrices'],np.ndarray):
            matrices = data['data_projection_matrices']
        else:
            if 'I1I1' in data['data_projection_matrices']:
                matrices = data['data_projection_matrices']['I1I1']
                data['data_projection_matrices_2'] = data['data_projection_matrices']
                
            else:
                matrices = data['data_projection_matrices']
        low_res_matrices = data.get('data_low_resolution_intensity_coefficients',False)        
        a_int = data['average_intensity']
        r_pt = data['data_radial_points']        
        data['average_intensity'] = SampledFunction(NestedArray(r_pt[:,None],1),a_int,coord_sys='cartesian')
        b_coeff = data.get('deg_2_invariant',False)
        data['b_coeff'] = b_coeff
        if isinstance(matrices,dict):
            sorted_keys = np.sort(tuple(int(key) for key in matrices.keys())).astype(str)
            matrices = tuple(matrices[k] for k in sorted_keys)
        if data['dimensions'] == 3:
            #log.info('projection_matrices 0 shape={}'.format(matrices['0'].shape))
            #matrices['0']=matrices['0'][:,None]
            #log.info('projection_matrices 0 shape={}'.format(matrices['0'].shape))
            #
            
            tmp = np.empty(len(matrices),object)
            for i,pm in enumerate(matrices):
                tmp[i]=pm
            matrices = tmp
            data['data_projection_matrices']=matrices #tuple(matrices)
            
            if not isinstance(low_res_matrices,bool):
                log.info(f'low res mat type = {type(low_res_matrices)}')
                tmp = np.empty(len(low_res_matrices),object)
                for i,pm in enumerate(low_res_matrices):
                    tmp[i]=pm
                low_res_matrices = tmp
            if len(matrices[0].shape)<2:
                matrices[0]=matrices[0][:,None] #legacy compatibility
            data['data_low_resolution_intensity_coefficients']=low_res_matrices #tuple(matrices)
        elif data['dimensions'] == 2:
            data['data_projection_matrices'] = np.array(matrices)
        return data
    
    def save_invariants(self,name,proj_class,**options):
        #log.info(options.keys())
        time_str=self.get_time_string()
        path=self.folders[self.files['invariants']['folder']]
        path_modifiers = {'dimensions':settings.project.dimensions,'structure_name':settings.project.structure_name,'date':time_str,}
        run= self.get_latest_run('invariants',path_modifiers=path_modifiers) + 1
        path_modifiers['run'] = run
        run_path=path.format(**path_modifiers)
        
        data_dict={}
        data_dict['dimensions']=proj_class.dimensions
        data_dict['xray_wavelength'] = proj_class.xray_wavelength
        data_dict['average_intensity'] = proj_class.average_intensity[:]
        data_dict['data_radial_points'] = proj_class.data_radial_points
        data_dict['data_angular_points'] = proj_class.data_angular_points
        data_dict['data_min_q'] = proj_class.data_min_q
        #matrices = proj_class.data_projection_matrices        
        if proj_class.dimensions == 3:
            data_dict['data_projection_matrices'] = proj_class.data_projection_matrices
            data_dict['data_low_resolution_intensity_coefficients'] = proj_class.data_low_resolution_intensity_coefficients
            #for key,matrices in proj_class.data_projection_matrices.items():
            #    matrices = (np.squeeze(matrices[0]),)+tuple(matrices[1:])
            #    matrices = {str(key):value for key,value in enumerate(matrices)}
            #    data_dict['data_projection_matrices'][key] = matrices
        elif proj_class.dimensions == 2:
            #log.info('n proj matrices = {} dtype = {}'.format(len(matrices),matrices.dtype))
            data_dict['data_projection_matrices'] = proj_class.data_projection_matrices
            data_dict['data_projection_matrix_error_estimates'] = proj_class.data_projection_matrix_error_estimates
        data_dict['data_projection_matrices_q_id_limits']=proj_class.data_projection_matrices_q_id_limits
        data_dict['max_order'] = proj_class.max_order
        #data_dict['pi_in_q'] = int(proj_class.pi_in_q)
        data_dict['number_of_particles'] = int(proj_class.number_of_particles)
        if options.get('save_invariant',False):
            data_dict['deg_2_invariant'] = proj_class.b_coeff
            data_dict['deg_2_invariant_masks'] = proj_class.b_coeff_masks
            data_dict['deg_2_invariant_q_id_limits'] = proj_class.b_coeff_q_id_limits

        data_folder = self.get_path('invariants_archive',path_modifiers = path_modifiers,is_file=False)
        options['path_modifiers']=path_modifiers
        self.save_direct(name,data_dict,**options)
        if options.get('create_symlink',False):
            self.create_symlink(name,'invariant_symlink',path_modifiers=path_modifiers)
        
        file_name = 'extraction_settings'
        self._save_settings(data_folder,project_name=file_name)

        try:
            if options.get('plot_first_invariants',False):
                for key,bls in proj_class.b_coeff.items():
                    bl_args = np.angle(bls)+np.pi
                    mask = proj_class.b_coeff_masks[key]
                    self._save_first_invariants(bls,proj_class.data_radial_points,data_folder,options,name="first_{}_".format(key))
                    self._save_first_invariants(bl_args,proj_class.data_radial_points,data_folder,options,name="first_arg_of_{}_".format(key))
                    self._save_first_invariants(bls,proj_class.data_radial_points,data_folder,options,name="mask_of_{}_".format(key),mask = mask)
        except Exception as e:
            log.info('Plotting first invariants failed !')
            traceback.print_exc()
        try:
            log.info(f'trying to plot errors for {proj_class.data_projection_matrix_error_estimates.keys()}')
            if options.get('plot_first_projection_matrix_error_estimates',False):                
                for key,pr_err in proj_class.data_projection_matrix_error_estimates.items():
                    #log.info(f'plotting erros for {key}')
                    mask = proj_class.b_coeff_masks[key]
                    o = {'plot_range':[1e-10,1]}
                    self._save_first_invariants(pr_err,proj_class.data_radial_points,data_folder,o,name="first_projection_matrix_errors_{}_".format(key),mask = mask)
        except Exception as e:
            log.info('Plotting first projection matrix error estimates failed !')
            traceback.print_exc()
        try:
            if options.get('plot_first_ccn',False) and (proj_class.dimensions == 3):
                for key,bls in proj_class.b_coeff.items():        
                    cns = deg2_invariant_to_cn_3d(bls,proj_class.data_radial_points,proj_class.xray_wavelength)
                    #log.info('bcoeff dtype = {}'.format(bls.dtype))
                    cns = np.moveaxis(cns,-1,0)
                    max_value = np.max(tuple(np.abs(cn).max() for cn in cns))
                    vmin,vmax = options.get('plot_range',[max_value*1e-12,max_value])            
                    if [vmin,vmax] == [None,None]:
                        vmin,vmax = [max_value*1e-12,max_value]
        
                    grid = GridFactory.construct_grid('uniform',[proj_class.data_radial_points,proj_class.data_radial_points])
                    order_ids = [0,2,4,6,8]
    
                    shape = [2,len(order_ids)]
                    layouts = []
                    for i in range(shape[0]):
                        layout_part = []
                        orders = np.arange(10*i,10*i+len(order_ids)*2,2)
                        for o in orders:
                            layout = {'title':'$B_{'+'{}'.format(o)+'}$',
                                      'x_label':'$q_1$',
                                      'y_label':'$q_2$'
                                      }
                            layout_part.append(layout)
                        layouts.append(layout_part)

            
                    fig_bl_masks = heat2D_multi.get_fig([[np.abs(cns[n]) for n in order_ids],[np.abs(cns[10+n]) for n in order_ids] ],scale = 'log',layout = layouts,grid =grid,shape = shape,size = (30,10),vmin= vmin, vmax = vmax,cmap='plasma')

                    bl_path = data_folder+'first_{}_CCn.matplotlib'.format(key)
                    self.save(bl_path,fig_bl_masks,dpi = 500)
        except Exception as e:
            log.info('Plotting first CCn failed !')
            traceback.print_exc()
            
        try:
            #log.info(f"trying to plot pr inv = {options.get('plot_first_invariants_from_proj_matrices',False)}")
            if options.get('plot_first_invariants_from_proj_matrices',False):
                #log.info('plot first invariants from projection matrices')
                bls2 = {}
                if proj_class.dimensions == 2:
                    for key,prs in proj_class.data_projection_matrices.items(): 
                        bls2[key] = harmonic_coeff_to_deg2_invariants(proj_class.dimensions,np.array(prs).T)
                elif proj_class.dimensions == 3:
                    for key,prs in proj_class.data_projection_matrices.items(): 
                        bls2[key] = harmonic_coeff_to_deg2_invariants(proj_class.dimensions,prs)
                for key,bls in bls2.items():
                    if not key == 'I2I1':
                        #log.info(f'bls shape = {bls.shape}')
                        self._save_first_invariants(bls,proj_class.data_radial_points,data_folder,options,name="first_{}_proj_matrices_to_".format(key))
        except Exception as e:
            log.info("Plotting Bl from projection matrices failed!")
            traceback.print_exc()
        try:
            if options.get('save_intensity_vtk',False):
                if proj_class.dimensions == 3:
                    V = proj_class.data_projection_matrices['I1I1']
                    cht = HarmonicTransform('complex',{'dimensions':3,'n_phis':0,'n_thetas':0,'max_order':proj_class.max_order})
                    grid = GridFactory.construct_grid('uniform',[proj_class.data_radial_points,cht.grid_param['thetas'],cht.grid_param['phis']])
                    log.info(f'len proj = {len(V)}')
                    log.info(f'shape proj 2 = {V[2].shape}')
                    log.info(f'grid shape = ')
                    I = np.abs(cht.inverse(V))

                    path = run_path + 'intensity_guess.vts'
                    log.info(f" intensity to {path}")
                    self.save(path,[I.real],grid_type='spherical',grid = grid)
                    
        except Exception as e:
            log.info('Plotting Best Intensity guess failed !')
            traceback.print_exc()


    def save_ccd(self,name,data,**options):
        log.info('custom saving of cross correaltion')
        opt=settings.project
        time_str=self.get_time_string()
        ccd_folder = self.folders['ccd_archive']
        path_modifiers =  {'structure_name':opt.structure_name,'date':time_str}
        run = self.get_latest_run(ccd_folder,path_modifiers = path_modifiers)+1
        path_modifiers['run']=run
        log.info(f'path modifiers = {path_modifiers}')
        try:
            data_path = self.get_path(name,path_modifiers=path_modifiers)
            log.info(f'data_path = {data_path},name = {name}')
            self.save_direct(name,data,path_modifiers=path_modifiers)
            log.info('saved h5')
        except Exception as e:
            log.warning(f'Failed to save cross-correlation h5 data! with error {e}')
            log.info(traceback.format_exc())
            
        try:
            log.info('saving settings')
            cc_path = self.get_path(name,path_modifiers = path_modifiers)
            cc_folder = os.path.dirname(cc_path)
            cc_name = os.path.basename(cc_path).split('.')[0]
            log.info(f'to {cc_folder}')
            self._save_settings(cc_folder,project_name='settings')
        except Exception as e:
            log.warning(f'Failed to save cross-correlation settings! with error {e}')
            log.info(traceback.format_exc())
            
        if options.get('save_model_vtk',False):
            try:
                density = options['model_density']
                grid = options['grid']
                if grid.shape[-1]==3:
                    grid_type = 'spherical'
                elif grid.shape[-1]==2:
                    grid_type = 'polar'
                self.save('model_density',[density],dset_names=['model_density'],grid=grid,grid_type=grid_type,path_modifiers = path_modifiers)
            except Exception as e:
                log.warning(f'Failed to save model density! with error {e}')
                log.info(traceback.format_exc())
        if options.get('save_symlink',False):
            try:
                self.create_symlink(name,'ccd_symlink',path_modifiers=path_modifiers)
            except Exception as e:
                log.warning(f'Failed to create cross-corelation symlink! with error {e}')
                log.info(traceback.format_exc())
