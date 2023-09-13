import logging
import sys
import os
import numpy as np
import traceback
from scipy.stats import unitary_group
file_path = os.path.realpath(__file__)
plugin_dir = os.path.dirname(file_path)
os.chdir(plugin_dir)


#from analysisLibrary.classes import ReciprocalProjectionData
from .projectLibrary import fxs_invariant_tools as i_tools
from .projectLibrary.ft_grid_pairs import max_order_from_n_angular_steps, get_grid
from .projectLibrary.fourier_transforms import generate_ft,load_fourier_transform_weights
from .projectLibrary.harmonic_transforms import HarmonicTransform
from .projectLibrary.hankel_transforms import generate_weightDict
from .projectLibrary.misk import generate_calc_center
from .projectLibrary.fxs_Projections import generate_shift_by_operator
from xframe.library.pythonLibrary import xprint
from xframe.library.gridLibrary import SampledFunction,NestedArray,GridFactory
from xframe.library.mathLibrary import nearest_positive_semidefinite_matrix
from xframe.library.mathLibrary import polar_spherical_dft_reciprocity_relation_radial_cutoffs,distance_from_line_2d
from xframe.interfaces import ProjectWorkerInterface
from xframe import database,settings
from xframe.library.mathLibrary import SampleShapeFunctions
from .projectLibrary.misk import _get_reciprocity_coefficient
from xframe import Multiprocessing
from xframe.library.mathLibrary import spherical_to_cartesian
log=logging.getLogger('root')

#class Worker(RecipeInterface):
class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        self.extractor = InvariantExtractor()
    #       global db
    #        db = database.project
    def run(self):
        db = database.project
        opt = settings.project
        
        self.extractor.extract()
            
        if self.extractor.success:
            xprint('Saving results')
            db.save('ccd', self.extractor.cc_data,model_density = self.extractor.density,grid=self.extractor.grid_pair.realGrid[:])
        return {},locals()


######### InvariantExtractor #########
class InvariantExtractor:
    def __init__(self):
        log.info('xyx')
        self.dimensions = False
        self.xray_wavelength = False
        self.enforce_cross_correlation_qq_sym = False
        self.enforce_cross_correlation_phi_sym = False    
        self.average_intensity = False
        self.integrated_intensity = False
        self.data_projection_matrices = {'I1I1':False}
        self.data_low_resolution_intensity_coefficients = False
        self.pi_in_q = None
        self.reciprocity_coefficient = np.pi
        self.max_order = settings.project.grid.max_order
        self.data_n_particles = False
        self.data_radial_points = False
        self.data_angular_points = False
        self.data_max_q = False
        self.data_min_q = False
        self.proj_min_q = {'I1I1':False}
        self.number_of_particles = 1
        self.extraction_routines = {
            'shapes': self.extract_bl_from_shapes,
            'density': self._bl_from_density
        }
        self.b_coeff={'I1I1':False}
        self.b_coeff_masks={'I1I1':False}
        self.b_coeff_q_id_limits = {'I1I1':False}
        self.data_projection_matrices_masks = {'I1I1':False}
        self.data_projection_matrices_q_id_limits = {'I1I1':False}
        self.data_projection_matrix_error_estimates={'I1I1':False}
        self.success=False

            
    def set_standard_instance_variables(self):
        opt = settings.project
        self.dimensions = opt.dimensions
        

    #################################
    ###    extraction routines    ###
    def extract_bl_from_shapes(self):        
        opt = settings.project        
        shape_opt = opt
        
        if shape_opt.GPU.use:
            settings.general.n_control_workers = shape_opt.GPU.n_gpu_workers
            Multiprocessing.comm_module.restart_control_worker()
        self.reciprocity_coefficient = _get_reciprocity_coefficient(shape_opt.fourier_transform)        
        #self.pi_in_q = shape_opt.fourier_transform.pi_in_q
        
        # create_grid
        centers = np.asarray(shape_opt['shapes']['centers'])
        log.info('centers = {}'.format(centers))
        sizes = np.asarray(shape_opt['shapes']['sizes'])
        types = np.asarray(shape_opt['shapes']['types'])
        density_values = np.asarray(shape_opt['shapes']['densities'])
        random_orientation = np.asarray(shape_opt['shapes']['random_orientation'])
        max_particle_radius = shape_opt.get('shape_size','not given')
        if not isinstance(max_particle_radius,(float,int)):
            max_particle_radius = np.max(centers[:,0]+sizes)
        else:
            max_particle_radius/=2 #since mag radius is half the size of the shape.
            
        oversampling = shape_opt.grid.oversampling        
        if isinstance(shape_opt.grid.max_q,bool):
            max_r = oversampling*max_particle_radius
            n_radial_points = shape_opt.grid.n_radial_points
            max_q = polar_spherical_dft_reciprocity_relation_radial_cutoffs(max_r,n_radial_points,reciprocity_coefficient = self.reciprocity_coefficient)
        else:
            max_q = shape_opt.grid.max_q#*(self.reciprocity_coefficient/np.pi)
            n_radial_points = shape_opt.grid.n_radial_points
            max_r = polar_spherical_dft_reciprocity_relation_radial_cutoffs(max_q,n_radial_points,self.reciprocity_coefficient)
        self.data_max_q = max_q
        log.info('maxR = {} maxQ ={} reciprocity_coefficient = {}'.format(max_r,max_q,self.reciprocity_coefficient))

        ht_opt = {'dimensions':self.dimensions,'max_order':self.max_order,**shape_opt.grid}
        log.info('max order = {}'.format(self.max_order))
        cht = HarmonicTransform('complex',ht_opt)
        self.cht = cht
        #weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points)        
        grid_opt={'dimensions':opt.dimensions,'type':shape_opt.fourier_transform.type,'max_q':max_q,'n_radial_points':n_radial_points,**cht.grid_param,'reciprocity_coefficient':self.reciprocity_coefficient}
        grid_pair = get_grid(grid_opt)
        self.grid_pair = grid_pair
        if opt.dimensions==3:
            self.data_radial_points =  grid_pair.reciprocalGrid[:,0,0,0]
            self.data_angular_points = grid_pair.reciprocalGrid[0,0,:,2]
        elif opt.dimensions==2:
            self.data_radial_points =  grid_pair.reciprocalGrid[:,0,0]
            self.data_angular_points = grid_pair.reciprocalGrid[0,:,1]
        self.data_max_q = np.max(self.data_radial_points)

        log.info('grid shape = {}'.format(grid_pair.realGrid[:].shape))
        log.info(f'grid extends: min {spherical_to_cartesian(grid_pair.realGrid[:]).min()} max {spherical_to_cartesian(grid_pair.realGrid[:]).max()}')

        xprint('Creating density:')
        density = np.zeros(grid_pair.realGrid[:].shape[:-1],dtype = float)
        for shape_type,center,size,dval,rand_rot in zip(types,centers,sizes,density_values,random_orientation):
            log.info(f'\n t {type}\n c {center}\n s {size}\n d {dval}\n r {random_orientation}')
            if shape_type == 'sphere':
                norm = 'standard'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot,coordSys='spherical')
            elif shape_type == 'tetrahedron':
                if opt.dimensions==3:
                    f = SampleShapeFunctions.get_tetrahedral_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,random_orientation=rand_rot)
                else:
                    f = SampleShapeFunctions.get_polygon_function(size,amplitude_function = lambda points: np.full(points.shape[:-1],dval),center=center,coordSys='polar',random_orientation=rand_rot)
            elif shape_type == 'cube':
                norm = 'inf'
                log.info('size = {}'.format(sizes[i]))
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            else:
                norm = 'inf'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            density += f(grid_pair.realGrid[:])
        self.density = density
        xprint('done.\n')
        log.info('densty shape = {} grid shape = {} , max_R = {}'.format(density.shape,grid_pair.realGrid[:].shape,grid_pair.realGrid[...,0].max()))
        xprint('Calculating Bl invariants:')
        self._bl_from_density(density,shape_opt,opt.structure_name,n_radial_points,grid_pair,cht)
        xprint('done.\n')

    def generate_fourier_transforms(self, grid_pair, harm_trf):
        opt = settings.project
        dimensions = opt.dimensions
        if dimensions==3:
            r_max = grid_pair.realGrid[:,0,0,0].max()
        elif dimensions ==2:
            r_max = grid_pair.realGrid[:,0,0].max()
        n_radial_points = grid_pair.reciprocalGrid.shape[0]
        
        weights_dict = load_fourier_transform_weights(opt.dimensions,opt.fourier_transform,opt.grid,database.project)

        ft_opt = opt.fourier_transform
        ft_type = ft_opt['type']
        #pi_in_q = ft_opt['pi_in_q']
        reciprocity_coefficient = _get_reciprocity_coefficient(ft_opt)
        use_gpu = opt.GPU.use
        log.info('reciprocity_coefficient = {}'.format(reciprocity_coefficient))
        fourierTransform,inverseFourierTransform=generate_ft(r_max,weights_dict,harm_trf,dimensions,reciprocity_coefficient=reciprocity_coefficient,use_gpu = use_gpu,mode = ft_type)      
        return fourierTransform,inverseFourierTransform

    def _bl_from_density(self,density,opt,name,n_radial_points,grid_pair,cht):
        self.reciprocity_coefficient = _get_reciprocity_coefficient(opt.fourier_transform)        
        db = database.project
        ft,ift = self.generate_fourier_transforms(grid_pair,cht)
        #log.info('start b_coeff calculation')
        #log.info('density shape = {}'.format(density.shape))
        #log.info('grid shape = {}'.format(grid_pair.real.shape))
        
        if settings.project.dimensions ==2:
            grid_type = 'polar'
        elif settings.project.dimensions ==3:
            grid_type = 'spherical'
        
        log.info('max_density = {}'.format(density.max()))
        n_particles=opt.n_particles

        bl = i_tools.density_to_deg2_invariants(density.astype(complex),ft,settings.project.dimensions,cht=cht)
        bl*=self.number_of_particles
        bl[0]*=self.number_of_particles
        self.b_coeff['I1I1'] = bl
        bl_shape = self.b_coeff['I1I1'].shape
        self.b_coeff_masks['I1I1'] = np.full(bl_shape,True)
        q_id_limits = np.zeros((bl_shape[0],)+(2,2),dtype=int)
        q_id_limits[...,1]=bl_shape[0]
        self.b_coeff_q_id_limits={'I1I1': q_id_limits}
        #self.b_coeff['I2I2'] = i_tools.density_to_deg2_invariants((density**2).astype(complex),ft,settings.project.dimensions,cht=cht)
        #self.b_coeff_masks['I2I2'] = np.full(self.b_coeff['I2I2'].shape,True)
        
        #self.b_coeff['I2I1'] = i_tools.density_to_deg2_invariants((density**2).astype(complex),ft,settings.project.dimensions,density2 =density.astype('complex'),cht=cht)
        #self.b_coeff_masks['I2I1'] = np.full(self.b_coeff['I2I1'].shape,True)
        #self.b_coeff[0] *= n_particles
        #log.info(self.b_coeff.shape)

        if settings.project.dimensions ==2:
            self.average_intensity = np.sqrt(np.diag(self.b_coeff['I1I1'][0].real))
        elif settings.project.dimensions ==3:
            self.average_intensity = np.sqrt(np.diag(self.b_coeff['I1I1'][0]).real/(4*np.pi))

    def from_data(self,data):
        self.dimensions = data['dimensions']
        self.xray_wavelength = data['xray_wavelength']
        self.average_intensity = data['average_intensity']
        #log.info('aint type ={}'.format(type(self.average_intensity)))
        self.data_radial_points = data['data_radial_points'][:]
        self.data_angular_points = data['data_angular_points'][:]
        self.data_max_q = np.max(self.data_radial_points)
        self.data_min_q = data.get('data_min_q',False)
        #self.pi_in_q = data.get('pi_in_q',False)
        self.pi_in_q = data.get('pi_in_q',True)
        self.max_order = data['max_order']
        self.number_of_particles=data.get('number_of_particles',1)
        self.data_projection_matrices = data['data_projection_matrices']
        log.info('data keys = {}'.format(data.keys()))
        self.b_coeff = data['b_coeff']
        if isinstance(self.b_coeff,bool):
            self.b_coeff = i_tools.projection_matrices_to_deg2_invariant_3d(self.data_projection_matrices)
        self.success =True

        
    ##################################
    ###   calc cross-correlation   ###

    def calc_cross_correlation(self):
        opt = settings.project
        mode = opt.cross_correlation.method
        data_grid = {'qs':self.data_radial_points,'phis':self.data_angular_points}
        wavelength = opt.cross_correlation.xray_wavelength
        bl = self.b_coeff['I1I1']
        if opt.dimensions ==3:
            cc = i_tools.deg2_invariant_to_cc_3d(bl,wavelength,data_grid,mode= mode,n_processes = opt.n_processes)
        elif opt.dimensions ==2:
            cc = i_tools.deg2_invariant_to_cc_2d(bl,self.cht)
        self.cross_correlation = cc
        

    ##################
    ## main routine ##
    def extract(self):
        opt = settings.project
        db = database.project
        self.set_standard_instance_variables()        
        self.extract_bl_from_shapes()

        xprint('Calculating cross-correlation:')
        self.calc_cross_correlation()
        xprint('done.\n')
        if opt.dimensions==3:
            self.integrated_intensity = np.trapz(self.average_intensity.data * self.data_radial_points**2 , x = self.data_radial_points,axis = 0)*4*np.pi
        elif opt.dimensions==2:
            self.integrated_intensity = np.trapz(self.average_intensity.data * self.data_radial_points , x = self.data_radial_points,axis = 0)*4*np.pi

        cc_data = {}
        cc_data['radial_points']=self.data_radial_points
        cc_data['angular_points']=self.data_angular_points
        cc_data['xray_wavelength']=opt.cross_correlation.xray_wavelength
        cc_data['cross_correlation']= {'I1I1':self.cross_correlation}
        cc_data['average_intensity']=self.average_intensity 
        cc_data['deg_2_invariant']= {"I1I1":self.b_coeff['I1I1']}
        cc_data['number_of_particles']=self.number_of_particles
        
        self.cc_data = cc_data

        self.success =True
        #log.info(self.data_projection_matrices['I1I1'])
        log.info('Extraction completed! \n')
        
