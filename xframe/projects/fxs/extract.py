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
from xframe.library.gridLibrary import SampledFunction,NestedArray,GridFactory
from xframe.library.mathLibrary import nearest_positive_semidefinite_matrix
from xframe.library.pythonLibrary import xprint
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
            xprint('Saving Results \n')
            db.save('invariants', self.extractor)
        return {},locals()


######### InvariantExtractor #########
class InvariantExtractor:
    def __init__(self):
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
        self.max_order = False
        self.data_n_particles = False
        self.data_radial_points = False
        self.data_angular_points = False
        self.data_max_q = False
        self.data_min_q = False
        self.proj_min_q = {'I1I1':False}
        self.number_of_particles = False
        self.extraction_routines = {
            'cross_correlation': self.extract_bl_from_cc,
            #'pdb': self.pre_init_from_pdb,
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
        self.max_order = opt.max_order
        

    #################################
    ###    extraction routines    ###
    def extract_bl_from_cc(self):
        log.info('Computing degree 2 invariants from cross-correlation data:')
        db = database.project
        ccd = db.load('ccd')
        log.info('loaded ccd')                
        log.info(f'ccd shape = {ccd["cross_correlation"]["I1I1"].shape}')
        opt = settings.project
        average_intensity = ccd.get('average_intensity',False)

        self.average_intensity = average_intensity
        self.data_average_intensity = np.array(average_intensity.data)
        self.xray_wavelength = ccd['xray_wavelength']
        self.data_radial_points = ccd['radial_points']
        self.data_max_q = np.max(self.data_radial_points)
        self.data_min_q = np.min(self.data_radial_points)
        self.data_angular_points = ccd['angular_points']
        
        max_theoretical_order = len(self.data_angular_points)//2 #max_order_from_n_angular_steps(self.dimensions,len(self.data_angular_points),anti_aliazing_degree = 1)
        max_order = opt.max_order
        try:
            assert max_theoretical_order >= max_order , 'Violation of:  max specified harmonic order ({})<{} for the given number of angular points {}. Continue with max harmonic order = '.format(max_order,max_theoretical_order,len(self.data_angular_points),max_theoretical_order)
        except AssertionError as e:
            log.warning(e)
            max_order = max_theoretical_order
        self.max_order = max_order

        cc_arrays = ccd.pop('cross_correlation')
        for dset_name,dopt in opt.cross_correlation.datasets.items():
            if isinstance(opt.cross_correlation.datasets_to_process,(list,tuple)):
                if not (dset_name in opt.cross_correlation.datasets_to_process):
                    continue
            if dset_name in cc_arrays:
                if opt.dimensions==3:
                    extraction_method = dopt.bl_extraction_method
                elif opt.dimensions == 2:
                    extraction_method = '2d Fourier series'
                xprint(f'\tProcessing cross-correlation {dset_name} (extraction method = {extraction_method})...')
                
                cc = cc_arrays.pop(dset_name)
                metadata = {**ccd,**dopt,'orders':np.arange(self.max_order+1),'mode':dopt.bl_extraction_method,'average_intensity':self.data_average_intensity}
                #log.info('max eig 3 = {}'.format(np.linalg.eigh(np.fft.rfft(cc,axis=-1)[...,0])[0].max()))
                b_coeff,q_mask = i_tools.cross_correlation_to_deg2_invariant(cc,self.dimensions,**metadata)
                del(cc)
                #log.info(f"b coeff shape  =  {b_coeff.shape}")
                #log.info(dopt)
                mask,q_id_limits = self.calc_deg_2_invariant_masks(dopt,b_coeff.shape,q_mask)
                self.b_coeff_masks[dset_name] = mask
                self.b_coeff_q_id_limits[dset_name] = q_id_limits
                self.b_coeff[dset_name] = self.apply_invariant_constraints(dopt,b_coeff,q_id_limits)
                
        if ('I2I1' in self.b_coeff) and ('I1I1' in self.b_coeff) and ('I2I2' in self.b_coeff) :
            n_q1 = self.b_coeff_masks["I1I1"].shape[0]
            min_q2 = self.b_coeff_masks["I1I1"].any(axis = -1)
            combined_mask = self.b_coeff_masks["I1I1"].any(axis = 1)[:,None,:]*self.b_coeff_masks["I2I2"].any(axis = 2)[:,:,None]
            q_id_mins = np.concatenate((self.b_coeff_q_id_limits['I2I2'][:,0,None,:],self.b_coeff_q_id_limits['I1I1'][:,0,None,:]),axis = 1)
            self.b_coeff_masks['I2I1'] = combined_mask
            self.b_coeff_q_id_limits['I2I1'] = q_id_mins
            b3 = self.b_coeff['I2I1']
            for o,b in enumerate(b3):
                #enforce that b3 is of the form A*B^dagger where A and B are of shape (Nq,2*o+1)
                u,s,vh = np.linalg.svd(b)
                num_o = 2*o+1
                #eig,_ = np.linalg.eig(b)
                #log.info(np.sort(eig)[::-1])
                b[:] = (u[:,:num_o]*s[:num_o])@vh[:num_o,:]
        if opt.cross_correlation.datasets.I1I1.modify_cc.get('subtract_average_intensity',False):
            if self.dimensions==3:
                # since the spherical harmonic of degree 0 is  Y^0_0 = 1/(2*np.sqrt(np.pi))
                # Another way to look at it is that 4*pi is the total angular surface of a unit sphere 
                factor = 4*np.pi
            elif self.dimensions ==2:
                factor = 1 # 2*np.pi
            self.b_coeff['I1I1'][0]=average_intensity[:,None]*average_intensity[None,:]*factor
        #log.info(f'max bls = \n {[np.abs(b).max() for b in self.b_coeff["I1I1"]]}\n')
        
    def extract_bl_from_shapes(self):        
        opt = settings.project        
        shape_opt = opt.shapes

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
            n_radial_points = shape_opt.n_radial_points
            max_r = polar_spherical_dft_reciprocity_relation_radial_cutoffs(max_q,n_radial_points,self.reciprocity_coefficient)
        self.data_max_q = max_q
        log.info('maxR = {} maxQ ={} reciprocity_coefficient = {}'.format(max_r,max_q,self.reciprocity_coefficient))

        ht_opt = {'dimensions':self.dimensions,'max_order':self.max_order,**shape_opt.grid}
        log.info('max order = {}'.format(self.max_order))
        cht = HarmonicTransform('complex',ht_opt)
        #weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points)        
        grid_opt={'dimensions':opt.dimensions,'type':shape_opt.fourier_transform.type,'max_q':max_q,'n_radial_points':n_radial_points,**cht.grid_param,'reciprocity_coefficient':self.reciprocity_coefficient}
        grid_pair = get_grid(grid_opt)
        if opt.dimensions==3:
            self.data_radial_points =  grid_pair.reciprocalGrid[:,0,0,0]
            self.data_angular_points = grid_pair.reciprocalGrid[0,0,:,2]
        elif opt.dimensions==2:
            self.data_radial_points =  grid_pair.reciprocalGrid[:,0,0]
            self.data_angular_points = grid_pair.reciprocalGrid[0,:,1]
        self.data_max_q = np.max(self.data_radial_points)

        log.info('grid shape = {}'.format(grid_pair.realGrid[:].shape))
        log.info(f'grid extends: min {spherical_to_cartesian(grid_pair.realGrid[:]).min()} max {spherical_to_cartesian(grid_pair.realGrid[:]).max()}')

        density = np.zeros(grid_pair.realGrid[:].shape[:-1],dtype = float)
        for shape_type,center,size,dval,rand_rot in zip(types,centers,sizes,density_values,random_orientation):
            log.info(f'\n t {type}\n c {center}\n s {size}\n d {dval}\n r {random_orientation}')
            if shape_type == 'sphere':
                norm = 'standard'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot,coordSys='spherical')
            elif shape_type == 'tetrahedron':
                f = SampleShapeFunctions.get_tetrahedral_function(size,lambda points: np.full(points.shape[:-1],dval),center=center)
            elif shape_type == 'cube':
                norm = 'inf'
                log.info('size = {}'.format(sizes[i]))
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            else:
                norm = 'inf'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            density += f(grid_pair.realGrid[:])
        log.info(f'radial density = {density.mean(axis=(1,2))} shape = {density.shape}')
            
        log.info('densty shape = {} grid shape = {} , max_R = {}'.format(density.shape,grid_pair.realGrid[:].shape,grid_pair.realGrid[...,0].max()))

        self._bl_from_density(density,shape_opt,opt.structure_name,n_radial_points,grid_pair,cht)


    def generate_fourier_transforms(self, grid_pair, harm_trf):
        opt = settings.project
        dimensions = opt.dimensions
        if dimensions==3:
            r_max = grid_pair.realGrid[:,0,0,0].max()
        elif dimensions ==2:
            r_max = grid_pair.realGrid[:,0,0].max()
        n_radial_points = grid_pair.reciprocalGrid.shape[0]
        
        weights_dict = load_fourier_transform_weights(opt.dimensions,opt.shapes.fourier_transform,opt.shapes.grid,database.project)

        ft_opt = opt.shapes.fourier_transform
        ft_type = ft_opt['type']
        #pi_in_q = ft_opt['pi_in_q']
        reciprocity_coefficient = _get_reciprocity_coefficient(ft_opt)
        use_gpu = opt.shapes.GPU.use
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
        
        if opt.save_vtk_density:
            log.info(f'saving density {type(density)}')
            db.save('model_density', [density], grid = grid_pair.realGrid[:], grid_type=grid_type,path_modifiers={'name':name})
            log.info(f'alive')
        if opt.save_vtk_intensity:
            db.save('model_intensity', [np.abs(ft(density.astype(complex))).real], grid = grid_pair.reciprocalGrid[:], grid_type=grid_type,path_modifiers={'name':name})
        
        log.info('max_density = {}'.format(density.max()))
        n_particles=opt.n_particles
        
        self.b_coeff['I1I1'] = i_tools.density_to_deg2_invariants(density.astype(complex),ft,settings.project.dimensions,cht=cht)
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
            self.average_intensity = SampledFunction(NestedArray(self.data_radial_points,1),np.sqrt(np.diag(self.b_coeff[0])),coord_sys='cartesian')
        elif settings.project.dimensions ==3:
            average_data = np.sqrt(np.diag(self.b_coeff['I1I1'][0]).real/(4*np.pi))
            self.average_intensity = SampledFunction(NestedArray(self.data_radial_points,1),average_data,coord_sys='cartesian')

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

        
    ###########################################################
    ###    PSD constraint + Projection matrix extraction    ###

    def calc_deg_2_invariant_masks(self,dopt,bl_shape,q_mask):
        min_type =  dopt.bl_q_limits.min.type
        max_type =  dopt.bl_q_limits.max.type

        empty_mask = np.ones(bl_shape,dtype = bool)
        #log.info(dopt.bl_q_limits)
            
        # min_limit        
        q_id_limits = np.zeros((bl_shape[0],)+(2,2),dtype=int)
        q_id_limits[...,1]=len(q_mask)
        if min_type == 'line':
            min_mask,q_id_mins = self.calc_deg_2_invariant_line_mask(dopt.bl_q_limits.min[min_type])
            q_id_limits[:,:,0]=q_id_mins
        else:
            min_mask = empty_mask.copy()
            
        # max_limit        
        if max_type == 'line':
            max_mask,q_id_maxs = self.calc_deg_2_invariant_line_mask(dopt.bl_q_limits.max[max_type],invert = True)
            q_id_limits[:,:,1]=q_id_maxs
        else:
            max_mask = empty_mask.copy()


        q_id_min = np.argmax(q_mask)
        q_id_max = np.argmax(q_mask[::-1])
        q_id_max = len(q_mask)-q_id_max
        mask = min_mask & max_mask
        mask[:,~q_mask]=False
        q_id_limits[q_id_limits[:,:,0]<q_id_min]=q_id_min
        q_id_limits[q_id_limits[:,:,1]>q_id_max]=q_id_max
        #log.info(f'computed q_ids shape = {q_id_limits.shape}')
        return mask,q_id_limits
        
        
        
    def calc_deg_2_invariant_line_mask(self,line_specifier,invert=False):
        qs = self.data_radial_points
        n_qs = len(qs)
        orders = np.arange(self.max_order+1)
        data_grid = GridFactory.construct_grid('uniform',[orders,qs])
        #log.info(f'line specifyer type = {type(line_specifier)}')
        if isinstance(line_specifier,tuple):
            mask1 = (-1*distance_from_line_2d(np.array(line_specifier[0]),data_grid[:]))>=0
            #log.info(f'mask1 shape = {mask1.shape}')
            mask2 = (-1*distance_from_line_2d(np.array(line_specifier[1]),data_grid[:]))>=0
            if not invert:
                q1_id = np.argmax(mask1,axis = 1)
                q2_id = np.argmax(mask2,axis = 1)
                if not mask1.any():
                    q1_id = n_qs-1
                if not mask2.any():
                    q2_id = n_qs-1
            else:
                mask1 = ~mask1
                mask2 = ~mask2
                q1_id = np.argmin(mask1,axis = 1)                
                q2_id = np.argmin(mask2,axis = 1)
                if mask1.all():
                    q1_id = n_qs
                if mask1.all():
                    q2_id = n_qs

            #log.info(f'mask2 shape = {mask2.shape}')
            q_id_limits = np.stack((q1_id,q2_id),axis = -1) 
            mask = mask1[:,:,None] * mask2[:,None,:]
        else:
            mask = (-1*distance_from_line_2d(np.array(line_specifier),data_grid[:]))>=0
            #log.info(f'mask shape = {mask.shape}')
            if not invert:
                q_id = np.argmax(mask,axis = 1)
                if not mask.any():
                    q_id = n_qs -1
            else:
                mask = ~mask
                q_id = np.argmin(mask,axis = 1)
                if mask.all():
                    q_id = n_qs
                #q_id = n_qs -q_id
            mask = mask[:,:,None] * mask[:,None,:]
            q_id_limits = np.stack((q_id,q_id),axis = -1) 
        #log.info(f'mask any = {mask.any()} mask all = {mask.all()}')
        return mask,q_id_limits


    def apply_invariant_constraints(self,dopt,b_coeff,q_id_limits):
        b_coeff_out = b_coeff.copy()
        if dopt.bl_enforce_psd:
            try:
                assert (q_id_limits[:,0,:]==q_id_limits[:,1,:]).all(),'Trying to compute projection matrices from non-square submatrix of deg2 invariants specified. That makes no sense since projection matrices are theoretically given by eig values of a positive semidefinite matrix. Continue using the q1_id_limits also for q2 to make the selection square.'
            except AssertionError as e:
                log.info(e)
                q_id_limits[:,1]=q_id_limits[:,0]
            for o in range(len(b_coeff)):
                q_slice = slice(*q_id_limits[o,0])
                b_coeff_out[o,q_slice,q_slice] = nearest_positive_semidefinite_matrix(b_coeff[o,q_slice,q_slice],low_positive_eigenvalues_to_zero=False)
            #eig,_=np.linalg.eigh(b_coeff_out[2])
            #log.info(f'min eigenvalue after corrections = {np.min(eig)}')
        return b_coeff_out
    

    def calc_projection_matrices(self):
        log.info('Extracting projection matrices:')
        log.info('Processing I1I1 ...')
        if settings.project.bl_eig_sort_mode == 'median_of_scaled_eigenvector':            
            sort_mode = 1
        else:
            sort_mode = 0
        
        temp = i_tools.deg2_invariant_to_projection_matrices(self.dimensions,self.b_coeff['I1I1'],q_id_limits=self.b_coeff_q_id_limits['I1I1'],sort_mode = sort_mode)
        self.data_projection_matrices['I1I1'],eig_I1I1 = temp
        self.data_projection_matrices_masks['I1I1'] = self.b_coeff_masks['I1I1'].sum(axis = 2)
        self.data_projection_matrix_error_estimates['I1I1'] = i_tools.calc_projection_matrix_error_estimate(self.b_coeff["I1I1"],self.data_projection_matrices['I1I1'])
        
        if settings.project.optimize_projection_matrices.use:
            log.info('Enforcing SHT constraints on projection matrices:')
            niter = settings.project.optimize_projection_matrices.n_iterations
            err_change_limit = settings.project.optimize_projection_matrices.error_change_limit
            new_proj_matrices = self.prephase_projection_matrices(niter,err_change_limit)
            self.data_projection_matrices['I1I1'] = new_proj_matrices
            
        if 'I2I2' in self.b_coeff:
            log.info('Processing I2I2 ...')
            temp = i_tools.deg2_invariant_to_projection_matrices(self.dimensions,self.b_coeff['I2I2'],q_id_limits = self.b_coeff_q_id_limits['I2I2'],sort_mode = sort_mode)
            self.data_projection_matrices['I2I2'],eig_I2I2 = temp
            self.data_projection_matrices_masks['I2I2'] = self.b_coeff_masks['I2I2'].sum(axis = 2)
            self.data_projection_matrix_error_estimates['I2I2'] = i_tools.calc_projection_matrix_error_estimate(self.b_coeff["I2I2"],self.data_projection_matrices['I2I2'])
            if 'I2I1' in self.b_coeff:
                log.info('Processing I2I1 ...')
                proj_I1I1 = self.data_projection_matrices['I1I1']
                proj_I2I2 = self.data_projection_matrices['I2I2']
                enforce_unitarity = settings.project.unitary_transform.enforce_unitarity
                if self.dimensions == 3:
                    self.data_projection_matrices['I2I1'],errors = i_tools.calc_unknown_unitary_transform(proj_I1I1,eig_I1I1,proj_I2I2,eig_I2I2,self.b_coeff['I2I1'],self.data_radial_points,q_id_limits = self.b_coeff_q_id_limits["I2I1"],method = settings.project.I2I1_unknown_tranrform_extraction_method)
                    self.data_projection_matrix_error_estimates['I2I1']=errors
    def calc_low_resolution_intensity_coefficients(self):
        max_order = settings.project.low_resolution_intensity_approximation.max_order
        if settings.project.optimize_projection_matrices.use:
            niter = settings.project.low_resolution_intensity_approximation.n_iterations
            err_change_limit = settings.project.low_resolution_intensity_approximation.error_change_limit
            new_projection_matrices = self.data_projection_matrices['I1I1'][:max_order+1]
            cht = HarmonicTransform('complex',{'dimensions':3,'max_order':max_order,'n_phi':0,'n_theta':0})
            new_proj_matrices,error_target_reached = i_tools.enforce_spherical_harmonic_transform_constraint(new_projection_matrices,niter,cht,rel_err_limit = err_change_limit)            
        else:
            new_proj_matrices = self.data_projection_matrices['I1I1'][:max_order+1]
        return new_proj_matrices
    
    def prephase_projection_matrices(self,n_iterations,err_change_limit):
        new_projection_matrices = [self.data_projection_matrices['I1I1'][0].copy()]
        for i in range(2,self.max_order,2):
            log.info(f'Processing orders <= {i}')
            new_projection_matrices.append(self.data_projection_matrices['I1I1'][i-1].copy())#@unitary_group.rvs(2*(i-1)+1))
            new_projection_matrices.append(self.data_projection_matrices['I1I1'][i].copy())#@unitary_group.rvs(2*i+1))
            #max_order = min(self.max_order,settings.project.low_resolution_intensity_approximation.max_order)
            cht = HarmonicTransform('complex',{'dimensions':3,'max_order':i,'n_phi':0,'n_theta':0})
            new_projection_matrices,error_target_reached = i_tools.enforce_spherical_harmonic_transform_constraint(new_projection_matrices,n_iterations,cht,rel_err_limit = err_change_limit)
        new_projection_matrices += self.data_projection_matrices['I1I1'][self.max_order:]
        cht = HarmonicTransform('complex',{'dimensions':3,'max_order':self.max_order,'n_phi':0,'n_theta':0})
        new_projection_matrices,error_target_reached = i_tools.enforce_spherical_harmonic_transform_constraint(new_projection_matrices,n_iterations,cht,rel_err_limit = err_change_limit)
        #cht = HarmonicTransform('complex',{'dimensions':3,'max_order':self.max_order,'n_phi':0,'n_theta':0})
        #new_proj_matrices,error_target_reached = i_tools.enforce_spherical_harmonic_transform_constraint(self.data_projection_matrices['I1I1'][:self.max_order+1],n_iterations,cht,rel_err_limit = 1e-6)
        return new_projection_matrices
    ##################
    ## main routine ##
    def extract(self):
        opt = settings.project
        db = database.project
        self.set_standard_instance_variables()        
        mode = settings.project.extraction_mode
        
        try:
            #this extracts the invariants
            xprint('Extracting invariants')
            self.extraction_routines[mode].__func__(self)
            xprint('done.\n')
        except KeyError as e:
            traceback.print_exc()
            log.error(' Extraction mode {} not known. Known modes are "{}"'.format(mode,self.extraction_routines.keys()))
            raise e

        xprint('\nComputing projection matrices Vl|vn')
        self.calc_projection_matrices()
        xprint('done.\n')

        xprint('Computing total intensity.')
        if opt.dimensions==3:
            self.integrated_intensity = np.trapz(self.average_intensity.data * self.data_radial_points**2 , x = self.data_radial_points,axis = 0)*4*np.pi
        elif opt.dimensions==2:
            self.integrated_intensity = np.trapz(self.average_intensity.data * self.data_radial_points , x = self.data_radial_points,axis = 0)*4*np.pi
        xprint('done.\n')
        
        for key,value in self.b_coeff_q_id_limits.items():
            self.data_projection_matrices_q_id_limits[key]=value[:,0]

        if self.dimensions==3:            
            low_res_Ilm = self.calc_low_resolution_intensity_coefficients()    
            self.data_low_resolution_intensity_coefficients = low_res_Ilm
            
        self.success =True
        #log.info(self.data_projection_matrices['I1I1'])
        xprint('Extraction completed! \n')
        
