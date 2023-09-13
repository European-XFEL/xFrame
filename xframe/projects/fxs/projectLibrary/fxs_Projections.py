import numpy as np
import time
from scipy.optimize import minimize_scalar
import logging


from xframe.library.pythonLibrary import DictNamespace
from xframe.library.pythonLibrary import create_threshold_projection
from xframe.library.physicsLibrary import spherical_formfactor
from xframe.library.gridLibrary import NestedArray,GridFactory
from xframe.library.gridLibrary import ReGrider,SampledFunction
from xframe.library.mathLibrary import PolarIntegrator,SphericalIntegrator,distance_from_line_2d,midpoint_rule
from xframe.library.mathLibrary import gaussian_fourier_transformed_spherical
from xframe.library.mathLibrary import spherical_to_cartesian

from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_2d
from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_3d
from . import fxs_invariant_tools as i_tools

from xframe import settings
from xframe import database
from xframe import Multiprocessing

log=logging.getLogger('root')

class RealProjection:
    # collection of possible real constraints
    # generation routine of a real_constraint function has to be named generate_<name>_projection
    # where <name> is the same string that is used in the settings file
    # each real support function takes as input an electron density (complex array)
    # and outputs a list [new_density,mask] where new_density is the changed density and mask is true at points in which the input denisity was modified.
    def __init__(self,opt,metadata):
        #self.initial_mask = False
        self.enforce_initial_support = True
        self.opt = opt
        self.real_grid = metadata['real_grid']
        self.auto_correlation = metadata.get('auto_correlation',False)
        self.metadata = metadata
        self._initial_mask = ~self.generate_initial_support_mask()
        self._mask = [self._initial_mask.copy()]
        self._random_mask = (np.random.rand(*self._initial_mask.shape)>0.9)
        self.projection = self.assemble_projection()
            
    
    @property
    def initial_support(self):
        return ~self._initial_mask.copy()


    @property
    def support(self):
        return ~self._mask[0]
    @support.setter
    def support(self,support):
        if self.enforce_initial_support:
            self._mask[0] = self._initial_mask | (~support)
        else:
            self._mask[0] = ~support        
    
    @property
    def mask(self):
        return self._mask[0]
    
    @mask.setter
    def mask(self,mask):
        if self.enforce_initial_support:
            self._mask[0] = self._initial_mask | mask
        else:
            self._mask[0] = mask
            
            
    def generate_support_projection(self):
        mask = self._mask
        def support_projection(density):
            #log.info('fu')
            #log.info('\n initial mask type = {} \n'.format(type(self.initial_mask)))
            #log.info(mask[0][:10,0])
            m= mask[0]
            density[m]=0
            return [density,m]
        return support_projection

    def generate_value_threshold_projection(self):
        threshold = self.opt['value_threshold'].get('threshold',0.0)
        threshold_projection = create_threshold_projection(threshold)
        return threshold_projection

    def generate_limit_imag_projection(self):        
        thresh = self.opt['limit_imag'].get('threshold',0.0)        
        def projection(density):
            imag_part = density.imag
            is_invalid_mask = np.abs(imag_part) >= thresh
            imag_part[is_invalid_mask] = 0
            return [density,is_invalid_mask]
        return projection
    def generate_average_center_projection(self):
        thresh = int(self.opt['average_center'].get('max_radial_id',1))
        dimension = self.real_grid.array.shape[-1]
        log.info('dim = {}'.format(dimension))
        axes = tuple(range(1,dimension))
        if dimension==2:
            def projection(density):
                density[:thresh]=np.mean(density[:thresh],axis=axes)[:,None]
                return [density,False]
        elif dimension==3:
            def projection(density):
                density[:thresh]=np.mean(density[:thresh],axis=axes)[:,None,None]
                return [density,False]
        return projection
    def assemble_projection(self):
        projections = []
        mask_dict = {}
        for key in self.opt.apply:
            try:
                projections.append([key,getattr(self,'generate_'+key+'_projection')()])
                mask_dict[key]=False
            except AttributeError as e:
                log.error('projection {} not known. Ignoring it.'.format(key))
        mask_dict['all']=False
        def real_projection(data):
            mask = False
            for name,projection in projections:
                #log.info(name)
                data,p_mask=projection(data)
                # individual masks are supposed to be True where projection changes the data
                mask_dict[name]=p_mask
                mask |= p_mask
                mask_dict['all']=mask
            return [data,mask_dict]
        return real_projection


    def generate_initial_support_mask(self):
        opt = self.opt.support.initial_support
        realGrid = self.real_grid
        support_type=opt['type']
        if support_type=='max_radius':
            maxR=opt['max_radius']
            #log.info(f'given_maximal radius is {maxR}')
            support_mask=np.where(realGrid[...,0]<maxR,True,False)
        elif support_type=='auto_correlation':
            assert isinstance(self.auto_correlation,np.ndarray), 'auto correlation not specified.'
            threshold=opt.auto_correlation.threshold
            max_value=np.max(self.auto_correlation)
            support_mask=self.auto_correlation >= threshold*max_value
            support_mask[realGrid[...,0]>settings.project.particle_radius]=0
        else:
            e=AssertionError('Initial support type "{}" is not known.'.format(support_type))
            log.error(e)
            raise e
        #    log.info('support mask[:]={}'.format(supportMask))
        #pres=heatPolar2D()
        #pres.present(auto_correlation.data.array,grid=realGrid)
        #pres.present(supportMask,grid=realGrid)
        return support_mask


def generate_initial_support_mask(opt,realGrid,projection):
    support_type=opt['type']
    if support_type=='max_radius':
        maxR=opt['max_r']
        supportMask=np.where(realGrid[...,0]<maxR,True,False)
    elif support_type=='auto_correlation':
        raise NotImplementedError()
        threshold=initial_support_specifier['threshold']        
        maxValue=np.max(auto_correlation.data)
        supportMask=auto_correlation.data>=(1-threshold)*maxValue
    else:
        e=AssertionError('Initial support type "{}" is not known.'.format(support_type))
        log.error(e)
        raise e
    #    log.info('support mask[:]={}'.format(supportMask))
    #pres.present(auto_correlation.data.array,grid=realGrid)
    #pres.present(supportMask,grid=realGrid)
    return ~supportMask


class ShrinkWrapParts():
    def __init__(self,real_grid,reciprocal_grid,initial_support,options = {}):# threshold = 0.04,gaussian_sigma = 2,mode = 'threshold',mode_options = {}):
        self.mode_routines = {'threshold':self.generate_get_new_mask_threshold,'fixed_volume':self.generate_get_new_mask_fixed_volume}

        mode = options.get('mode','threshold')
        mode_options = options.get(mode,{})
        
        dimension = real_grid[:].shape[-1]
        
        threshold = options.get('thresholds',[0.06])[0]
        #log.info(f'threshold = {threshold}')
        if dimension == 2:
            self.default_sigma = np.pi/(reciprocal_grid[:,0,0].max())
        elif dimension == 3:
            self.default_sigma = np.pi/(reciprocal_grid[:,0,0,0].max())
        gaussian_sigma = self.default_sigma
        #log.info(f'default sigma = {self.default_sigma}')
  
        self.mode = mode
        self.mode_options = mode_options
        self.real_grid = real_grid[:]
        self.reciprocal_grid = reciprocal_grid[:]
        dimension = self.real_grid.shape[-1]
        if dimension ==2:
            self.integrator = PolarIntegrator(self.real_grid)
        elif dimension == 3:
            self.integrator = SphericalIntegrator(self.real_grid)
        self.initial_support = initial_support
        self.initial_volume = self.integrator.integrate_normed(initial_support.astype(float))
        
        self._threshold = [threshold]
        self._gaussian_sigma = [gaussian_sigma]
        self.gaussian_values = gaussian_fourier_transformed_spherical(self.reciprocal_grid,self._gaussian_sigma[0])
        
        self.get_new_mask = self.mode_routines[self.mode]()
        self.multiply_with_ft_gaussian = self.generate_multiply_by_ft_gaussian()
        #log.info(f'\n Shrink wrap mode = {mode} , get new mask routine = {self.get_new_mask} \n')
    @property
    def threshold(self):
        return self._threshold[0]
    @threshold.setter
    def threshold(self,value):
        if value<0:
            self._threshold[0]=0
            log.warning('Shrikwrap threshold has to lie in [0,1] but given value is {}. Projecting threshold to {}.'.format(value,self._threshold[0]))
        elif value>=1:
            self._threshold[0]=1
            log.warning('Shrikwrap threshold has to lie in [0,1] but given value is {}. Projecting threshold to {}.'.format(value,self._threshold[0]))
        else:
            self._threshold[0]=value


    @property
    def gaussian_sigma(self):
        return self._gaussian_sigma[0]
    @gaussian_sigma.setter
    def gaussian_sigma(self,value):
        value_valid_type = not ( (not np.issubdtype(np.array(value).dtype,np.number)) or isinstance(value,bool))
        value_valid = False
        if value_valid_type:
            value_valid = value>0
        if value_valid:
            self._gaussian_sigma[0]=value
        else:
            self._gaussian_sigma[0]=self.default_sigma
        self.gaussian_values[:] = gaussian_fourier_transformed_spherical(self.reciprocal_grid,self._gaussian_sigma[0])
        
    def generate_get_new_mask_threshold(self):
        threshold = self._threshold
        def get_new_mask(convolution_data):
            #log.info(f'\n SW threshold = {threshold} and sigma = {self._gaussian_sigma}\n')
            convolution_data=convolution_data.real #np.abs(convolution_data).real
            #convolution_data=np.abs(convolution_data).real
            #convolution_data=np.abs(convolution_data).real #np.abs(convolution_data).real
            convolution_data[convolution_data<0]=0
            max_value= convolution_data.real.max()
            min_value= convolution_data.real.min()
            diff = max_value-min_value
            new_mask = convolution_data >= min_value + threshold[0]*diff
            return new_mask
        return get_new_mask
    
    def generate_get_new_mask_fixed_volume(self):
        try:
            target_volume = self.initial_volume*self.mode_options['volume'] # number in [0,1] indication volume fraction relative to self.initial_volume
        except KeyError as e:
            log.error('selected ShrinkWrap mode "fixed_volume" but no volume was specified in mode options {}'.format(self.mode_options))
            raise e
        integrate = self.integrator.integrate_normed
        old_volume = [self.initial_volume]
        d_vol_thresh = 0.2
        def get_new_mask(convolution_data):
            convolution_data=convolution_data.real
            max_value= convolution_data.real.max()
            min_value= convolution_data.real.min()
            diff = max_value-min_value
            def new_volume(threshold):
                new_mask = (convolution_data >= min_value + threshold*diff) & self.initial_support
                _vol = integrate(new_mask.astype(float))
                metric = abs(_vol-target_volume)
                rate_of_change = abs(old_volume[0] - _vol)/old_volume[0]
                #log.info('rate_of_change = {} threshold = {}'.format(rate_of_change,threshold))
                if rate_of_change>d_vol_thresh:
                    metric = np.inf
                return metric
            opti_result = minimize_scalar(new_volume,bounds=(0.0,1.0),method='golden')
            threshold = opti_result.x
            log.info('Optimization result = {}'.format(opti_result))
            new_mask = (convolution_data >= min_value + threshold*diff) & self.initial_support
            new_volume = integrate(new_mask.astype(float))
            log.info(f'old volume = {old_volume[0]} new volume = {new_volume}, target_volume={target_volume}')
            old_volume[0] = new_volume
            return new_mask
        return get_new_mask

    
    def generate_multiply_by_ft_gaussian(self):
        gaussian_values = self.gaussian_values
        def multiply_ft_gaussian(data):
            return data*gaussian_values
        return multiply_ft_gaussian

    
### FXS Projections
def generate_fxs_projection(specifier):
    dimension=specifier['fxs_data'].dimension
    if dimension==2:
        FXS_projection=construct_fxs_projection_parts_2D(specifier)
    else:
        raise NotImplementedError
    return FXS_projection

def calculate_fxs_projection_vector_old(fxs_data,positive_harmonic_orders):
    bCoefficients=fxs_data.bCoeff
    
    def firstEigenvectorAndValue(expansionCoefficient):
        matrix=bCoefficients.data.array[:,:,expansionCoefficient]
#        log.info('matrix of shape {} to take eigen values from = \n{}'.format(matrix.shape,matrix))
        eigValues,eigVectors=np.linalg.eigh(matrix)
        max_arg=eigValues.argmax()
        return [eigVectors[:,max_arg],eigValues[max_arg]]
    listOfEigenvectorsAndValues=np.array(list(map(firstEigenvectorAndValue,positive_harmonic_orders)),dtype=np.object)    
    eigenValues=listOfEigenvectorsAndValues[:,1].astype(complex)
    eigenVectors=np.array(tuple(listOfEigenvectorsAndValues[:,0]),dtype=complex)
    projection_vector=np.swapaxes(eigenVectors,0,1)*np.sqrt(eigenValues.astype(complex))
    fxs_data.projection_vector=projection_vector
    return fxs_data

def calculate_fxs_projection_vector(fxs_data,reciprocal_projection_specifier):
    positive_harmonic_orders=reciprocal_projection_specifier['positive_harmonic_orders']
    #log.info('projection orders={}'.format(positive_harmonic_orders))
    bCoefficients=fxs_data.bCoeff
    
    def firstEigenvectorAndValue(expansionCoefficient):
        matrix=bCoefficients[:,:,expansionCoefficient]
#        log.info('matrix of shape {} to take eigen values from = \n{}'.format(matrix.shape,matrix))
        eigValues,eigVectors=np.linalg.eigh(matrix)
        max_arg=eigValues.argmax()
        
        return [eigVectors[:,max_arg],eigValues[max_arg]]
    listOfEigenvectorsAndValues=np.array(list(map(firstEigenvectorAndValue,positive_harmonic_orders)),dtype=np.object)    
    eigenValues=listOfEigenvectorsAndValues[:,1].astype(complex)
    #log.info('lambdas={}'.format(eigenValues))
    eigenVectors=np.array(tuple(listOfEigenvectorsAndValues[:,0]),dtype=complex)
    projection_vector=np.swapaxes(eigenVectors,0,1)*np.sqrt(eigenValues.astype(complex))    
    fxs_data.projection_vector=projection_vector
#    log.info('shape ={} projection_vector={}'.format(projection_vector.shape[1],np.abs(np.sum(projection_vector,axis=0))))
    return fxs_data


def construct_fxs_projection_parts_2D(specifier):
    fxs_data=specifier['fxs_data']
    specifier['projection_vector']=modify_projection_vector_2D(fxs_data.projection_vector,specifier)
    mask=specifier.get('mask',False)
    if isinstance(mask,bool):
        specifier['mask']=True
    
    use_SO_freedom=specifier['SO_freedom']['use']
    
    fxs_projection=generate_coefficient_projection_2D(specifier)
    approximate_unknowns_without_SO=generate_approximate_unknowns_2D(specifier)

    if use_SO_freedom:
        apply_SO_freedom=generate_apply_SO_freedom_2D(specifier)
        def approximate_unknowns(intensity_harmonic_coefficients):
            return apply_SO_freedom(approximate_unknowns_without_SO(intensity_harmonic_coefficients))
    else:
        approximate_unknowns=approximate_unknowns_without_SO
            
    projection_dict={'projection':fxs_projection,'approx_unknowns':approximate_unknowns,'proj_mask':mask}
    return projection_dict
        
def modify_projection_vector_2D(projection_vector,specifier):
    rescale_projection=specifier.get('rescale_projection_to_1',False)
    average_intensity=specifier['fxs_data'].aInt.data.astype(complex)
    positive_harmonic_orders=specifier['positive_harmonic_orders']
    odd_order_mask=positive_harmonic_orders%2==1
    vector=projection_vector.copy()
    vector[:,odd_order_mask]=0
    vector[:,0]=average_intensity
    if rescale_projection:        
        max_vector_value=np.abs(np.max(vector))
        vector/=max_vector_value
#    log.info('projection vector for SO={}'.format(np.abs(np.sum(projection_vector,axis=0))))
    return vector

def generate_approximate_unknowns_2D(specifier):
    positive_harmonic_orders=specifier['positive_harmonic_orders']
    projection_vector=specifier['projection_vector']
    reciprocal_grid=specifier['reciprocal_grid']
 #   log.info(np.conjugate(np.sum(projection_vector,axis=0)))
    def approximate_unknowns(intensity_harmonic_coefficients):
        scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,positive_harmonic_orders]*np.conjugate(projection_vector)*reciprocal_grid[:,positive_harmonic_orders,0]
        scalar_prod_Im_vm=np.sum(scalar_prod_Im_vm_summands,axis=0)
#        log.info(scalar_prod_Im_vm)
        unknowns=scalar_prod_Im_vm/np.where(np.abs(scalar_prod_Im_vm)==0,1,np.abs(scalar_prod_Im_vm))
        unknowns[0]=1
        #log.info('unknowns={}'.format(unknowns))
        return unknowns
    return approximate_unknowns

def generate_coefficient_projection_2D(specifier):
    positive_harmonic_orders=specifier['positive_harmonic_orders']
    projection_vector=specifier['projection_vector']
    mask=specifier['mask']
    copy = np.array
    def fxs_projection(intensity_harmonicCoefficients,unknowns):
        projected_intensity_coefficients_array=copy(intensity_harmonicCoefficients)
        projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_vector*unknowns
        #apply mask
        intensity_harmonic_coefficients=np.where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
        return intensity_harmonic_coefficients
    return fxs_projection


def generate_orientation_matching_2D(n_positive_harmonic_orders):
    previous_phases=[False]
    previous_mask=[True]
    def orientation_matching(intensity_harmonic_coefficients,unknowns):
        non_zero_mask=unknowns!=0
        mask=previous_mask[0]*non_zero_mask
        phases=np.zeros(unknowns.shape)
        phases[non_zero_mask]=(-1.j*np.log(unknowns[non_zero_mask])).real
        if not isinstance(previous_phases[0],bool):
            non_zero_phases=phases[mask]
            optimal_phase=np.sum(non_zero_phases-previous_phases[0][mask])/len(non_zero_phases)
            shift=np.exp(-1.j*np.arange(n_positive_harmonic_orders)*optimal_phase)
#            log.info(shift)
            intensity_harmonic_coefficients*=shift
        previous_phases[0]=phases
        previous_mask[0]=non_zero_mask
        return intensity_harmonic_coefficients
    return orientation_matching

def generate_orientation_matching(dimensions,n_positive_harmonic_orders):
    if dimensions == 2:
        orientation_matching=generate_orientation_matching_2D(n_positive_harmonic_orders)
    elif dimensions == 3:
        def orientation_matching(intensity_harmonic_coefficients,unknowns):
            raise NotImplementedError()
    else:
        log.error('dimensions must be 2 or 3 but {} was given'.format(dimensions))
    return orientation_matching


class ReciprocalProjection:
    def load_data(self,data):
        opt = settings.project
        self.dimensions = data['dimensions']
        self.xray_wavelength = data['xray_wavelength']
        self.average_intensity = data['average_intensity']
        #log.info('aint type ={}'.format(type(self.average_intensity)))
        self.data_radial_points = data['data_radial_points'][:]
        self.data_angular_points = data['data_angular_points'][:]
        self.data_max_q = np.max(self.data_radial_points)
        self.data_min_q = np.min(self.data_radial_points)
        #self.pi_in_q = data.get('pi_in_q',False)
        #self.pi_in_q = data['pi_in_q']
        #log.info(f'data q_min = {self.data_min_q} max_q ={self.data_max_q}')
        
        self.data_max_order=data['max_order']
        self.data_projection_matrices = data['data_projection_matrices']
        self.data_low_resolution_intensity_coefficients = data.get('data_low_resolution_intensity_coefficients',self.data_projection_matrices)
        #log.info(f' pr matrix type {type(self.data_projection_matrices)}')
        self.data_projection_matrices_q_id_limits = data.get('data_projection_matrices_q_id_limits',False)
        #log.info(f'data proj matrices len = {len(self.data_projection_matrices)}')
        #if not isinstance(self.data_projection_matrices,np.ndarray):
        #    tmp = np.empty(len(self.data_projection_matrices),object)
        #    for i,pm in enumerate(self.data_projection_matrices):
        #        tmp[i]=pm
        #    self.data_projection_matrices = tmp

                
    def __init__(self,grid,data,max_order):
        self.load_data(data)
        if self.dimensions==2:
            self.integrated_intensity = midpoint_rule(self.average_intensity.data * self.data_radial_points , self.data_radial_points,axis = 0)*2*np.sqrt(np.pi)
        else:
            self.integrated_intensity = midpoint_rule(self.average_intensity.data * self.data_radial_points**2 , self.data_radial_points,axis = 0)*2*np.sqrt(np.pi)
        opt = settings.project.projections.reciprocal        
        self.opt = opt
        self.grid=grid        
        self.radial_points=grid.__getitem__((slice(None),)+(0,)*self.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        self.max_q = np.max(self.radial_points)
        #self.input_is_complex=opt.input_is_halfcomplex
        
        self.positive_orders = np.arange(max_order+1)        
        
        self.projection_orders = self.read_projection_orders(opt.get('used_orders',opt.used_order_ids),max_order)
        #log.info('projection_orders = {}'.format(self.projection_orders))
        self.projection_order_ids_local = np.arange(len(self.projection_orders))
        self.projection_order_ids_global = np.isin(self.positive_orders,self.projection_orders).nonzero()[0]
                    
        self.used_order_ids = opt.used_order_ids
        self.used_orders = {order:id for (order,id) in zip(self.positive_orders,self.used_order_ids)}
        self.use_SO_freedom=opt.SO_freedom.use
        
        # Won't work without the list around opt.number_of_particles.initial .
        # This way number_of_particles[0] automatically contains updated versions of the particle number.  
        self.number_of_particles = [opt.number_of_particles.initial] 
        self.number_of_particles_dict = {'number_of_particles':self.number_of_particles,'negative_fraction':[],'gradient':[]}

        #log.info('data_projection_matrices shape = {}'.format(tuple(d.shape for d in self.data_projection_matrices)))
        pm,low_res = self._regrid_data()        
        #log.info(f'len regridded proj mat = {len(pm)}')
        
        self.projection_matrices = pm
        nq = self.grid.shape[0]
        if self.dimensions ==3:
            self.full_projection_matrices = [np.zeros((nq,min(nq,2*o+1)),dtype=complex) for o in range(max_order+1)]
        elif self.dimensions == 2:
            self.full_projection_matrices = np.zeros((max_order+1,nq),dtype=complex)
        for oid,pm in zip(self.used_order_ids,self.projection_matrices):
            self.full_projection_matrices[oid]=pm
            
        self.low_resolution_intensity_coefficients = low_res
        #log.info('projection_matrices after regrid = {}'.format(tuple(d.shape for d in self.projection_matrices)))
        self.projection_matrices=self.modify_projection_matrices(opt)
        #log.info('projection_matrices after modify = {}'.format(tuple(d.shape for d in self.projection_matrices)))
        self.assert_projection_matrices_have_right_shape()        
        self.radial_mask=self.generate_radial_mask(opt.q_mask)        
        self._fixed_intensity = [np.ones(grid.shape[:-1],dtype=complex)]        
        mtip_projection_base=self.generate_coeff_projection_base()        
        self.mtip_projection_base  = mtip_projection_base        
        self.mtip_projection = self.generate_coeff_projection(mtip_projection_base)            
        self.approximate_unknowns=self.generate_approximate_unknowns()
            
        #if opt.number_of_particles.GPU:            
        #    self.particle_number_projection = self.generate_number_of_particles_porjection_gpu()
        #else:
        #    self.particle_number_projection = self.generate_number_of_particles_porjection()
        
        self.project_to_modified_intensity = self.generate_project_to_modified_intensity()
        self.project_to_fixed_intensity = self.generate_project_to_fixed_intensity()
        self.deg2_invariants = self.calc_deg2_invariants()

        self.remaining_SO_projection=False
        if self.use_SO_freedom:
            radial_high_pass=opt.SO_freedom.get('radial_high_pass',0.2)
            self.remaining_SO_projection = self.generate_remaining_SO_projection(radial_high_pass=radial_high_pass)

    

    @property
    def fixed_intensity(self):
        return self._fixed_intinsity[0]
    @fixed_intensity.setter
    def fixed_intensity(self,value):
        self._fixed_intensity[0]=value.real
        
    def read_projection_orders(self,projection_orders,max_harmonic_order):
        try:
            assert projection_orders.max()<=self.data_max_order, 'Max available harmonic order from dataset is {} but the maximal projection order is {}. Restricting ordes to the ones present in the dataset.'.format(self.data_max_order,projection_orders.max())
        except AssertionError as e:
            log.warning(e)
            projection_orders = projection_orders[projection_orders<=self.data_max_order]
        try:
            assert projection_orders.max()<=max_harmonic_order, 'The maximal harmonic oder given by the input harmonic analysis {} is smaller than the specified maximal projection order {}.Limiting projection orders and continue.'.format(max_harmonic_order,projection_orders.max())
        except AssertionError as e :
            log.warning(e)
            projection_orders = projection_orders[projection_orders<=max_harmonic_order]
        return projection_orders
        
    def assert_projection_matrices_have_right_shape(self):
        try:
            n_radial_points_grid = len(self.radial_points)
            n_radial_points_projection_matrix = self.projection_matrices[0].shape[0]
            assert n_radial_points_grid == n_radial_points_projection_matrix,'Mismatch between th number of radial sampling points in reconstruction grid {}  and projection data {}. Abort reconstruction !'.format(n_radial_points_grid,n_radial_points_projection_matrix)
        except AssertionError as e:
            log.error(e)
            raise e
            
    def extract_used_options(self):
        fourier_opt = settings.project.fourier_transform
        rp_opt = settings.project.projections.reciprocal
        opt = DictNamespace(
            pos_orders = fourier_opt.pos_orders,
            **rp_opt
        )
        return opt
    def generate_radial_mask(self,mask_opt):
        '''
        Constructs the radial mask (mask of momentum transfair values). Masked values are False and will not be used 
        in the reciprocal coefficient projection defined in self.generate_coeff_projection.
        '''
        radial_points = self.radial_points
        mask = np.full((len(self.positive_orders),len(radial_points)),False)        
        if ~isinstance(self.data_min_q,bool):
            data_mask =  mask | (radial_points>=self.data_min_q) & (radial_points<=self.data_max_q)
        
        if isinstance(mask_opt,(dict,DictNamespace)):
            mtype=mask_opt['type']
            #log.info('mask type = {}'.format(mtype))
            if mtype=='none':
                mask = True
            elif mtype == 'from_projection_matrices':
                data_limit_ids = self.data_projection_matrices_q_id_limits['I1I1']
                for mask_part,lim in zip(mask,data_limit_ids):
                    min_q = self.data_radial_points[lim[0]]
                    max_q = self.data_radial_points[lim[1]-1]
                    mask_part[:] = (radial_points>min_q) & (radial_points<max_q)
            elif mtype == 'manual':
                manual_opt = mask_opt['manual']
                manual_type = manual_opt['type']
                #log.info('manual type = {}'.format(manual_type))
                if manual_type == 'region':
                    region=manual_opt['region']
                    radial_points=self.radial_points
                    #log.info(f'region = {region}')
                    if (region[0] == False) and (region[1] != False):
                        #log.info(f'radial_points = {radial_points}')
                        mask[:]=(radial_points<region[1])[None,:]
                    elif (region[0] != False) and (region[1] == False):                    
                        mask[:]=(radial_points>=region[0])[None,:]
                        log.info('radial mask non zero count {} of {}'.format(np.sum(mask),np.prod(mask.shape)))
                    elif (region[0] != False) and (region[1] != False):
                        mask[:]=((radial_points>=region[0])  & (radial_points<region[1]))[None,:]
                    else:
                        mask[:]=True
                        log.info('nothing masked  {}'.format(mask.all()))
                elif manual_type == 'order_dependent_line':
                    points = manual_opt['order_dependent_line']
                    log.info('points = {}'.format(points))
                    orders = self.positive_orders
                    data_grid = GridFactory.construct_grid('uniform',[orders,radial_points])                 
                    mask = (-1*distance_from_line_2d(np.array(points),data_grid[:]))>=0
        else:
            log.warning('Could not parse projections.reciprocal.q_mask option. Proceeding without custom q_mask')
            mask = True
                
        mask = mask & data_mask
        return mask

    def calc_deg2_invariants(self):
        if self.dimensions == 2:
            invariants = harmonic_coeff_to_deg2_invariants_2d(self.projection_matrices.T)
        elif self.dimensions == 3:
            invariants = harmonic_coeff_to_deg2_invariants_3d(self.projection_matrices)
        #log.info('rp deg 2 invariant shape = {}'.format(invariants.shape))
        return invariants

    def _regrid_data(self):
        dim = self.dimensions
        order_ids=list(self.used_orders.values())
        have_same_shapes = self.radial_points.shape == self.data_radial_points.shape
        projection_matrices = self.data_projection_matrices[order_ids]
        needs_regridding = True
        interpolation_type = self.opt.regrid.interpolation
        if have_same_shapes:
            if (self.data_radial_points == self.radial_points).all():
                needs_regridding = False
        #log.info('needs regridding = {}'.format(needs_regridding))
        #log.info('initial projection matrix shape = {}'.format(projection_matrices[-1].shape))   
        if needs_regridding:
            r_pt=NestedArray(self.radial_points[:,None],1)
            #log.info('r_pt = {}'.format(r_pt[:10]))
            data_r_pt=NestedArray(self.data_radial_points[:,None],1)
            #log.info('n new points={} n old points ={}'.format(len(r_pt[:]),len(data_r_pt[:])))
            low_res = False
            if dim == 2:                
                self.average_intensity.regrid(r_pt,options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                #log.info('proj data shape = {}'.format(self.data_projection_matrices.shape))
                #log.info(f'order ids = {order_ids}')
                projection_matrices = ReGrider.regrid(np.array(self.data_projection_matrices)[order_ids,...],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                #log.info("data_r_min = {} grid r min = {}".format(data_r_pt.array.min(),r_pt.array.min()))
                #log.info('regrided projection matrices complete shape = {}'.format(projection_matrices.shape))
                #log.info('regrided projection matrices shape = {}'.format(projection_matrices.shape))
                #self.average_intensity.regrid(r_pt)
                #projection_matrices = ReGrider.regrid(self.data_projection_matrices[...,order_ids],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type})
            elif dim == 3:
                self.average_intensity.regrid(r_pt,options={'apply_over_axis':0,'fill_value': 0.0,'interpolation':interpolation_type})
                data_projection_matrices=self.data_projection_matrices
                projection_matrices = tuple(ReGrider.regrid(data_projection_matrices[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type}) for o_id in order_ids)
                if isinstance(self.data_low_resolution_intensity_coefficients,np.ndarray):
                    data_low_res = self.data_low_resolution_intensity_coefficients
                    low_res = tuple(ReGrider.regrid(data_low_res[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type}) for o_id in np.arange(len(data_low_res)))
                    
        #log.info('regrided projection matrices shape = {}'.format(projection_matrices[-1].shape))
        return projection_matrices,low_res
    

    def modify_projection_matrices(self,opt):
        dim = self.dimensions
        use_averaged_intensity=opt.get('use_averaged_intensity',False)
        use_odd_orders_to_0=opt.get('odd_orders_to_0',False)
        
        average_intensity=self.average_intensity.data.astype(complex)
        used_orders=self.used_orders
        odd_order_mask=np.array(tuple(used_orders))%2==1
        if dim == 2:
            proj_matrices=self.projection_matrices.copy()
        elif dim == 3:
            proj_matrices=[matrix.copy() for matrix in self.projection_matrices]
        #log.info(f'use average intensity = {use_averaged_intensity}')
        
        if use_odd_orders_to_0:
            if dim == 2:        
                proj_matrices[odd_order_mask,:]=0
            elif dim == 3:
                for odd_order in np.array(tuple(used_orders))[odd_order_mask]:
                    proj_matrices[used_orders[odd_order]][:]=0

        if use_averaged_intensity:
            zero_id = used_orders[0]            
            if dim == 2:
                #log.info('proj dtype = {} average dtype = {}'.format(proj_matrices.dtype,average_intensity.dtype))
                proj_matrices[zero_id] = average_intensity
                #proj_matrices[:,used_orders[0]]=average_intensity
            elif dim == 3:
                # assumes schmidt seminormalized definition of spherical harmonics -> factor of 2*np.sqrt(pi) 
                proj_matrices[zero_id]= average_intensity[:,None].real*2*np.sqrt(np.pi)
                
        # internally orthonarmalized spherical harmonics are used but data is ussually supplied for schmidt seminormalized spherical harmonics
        if dim == 3:
            for pm in proj_matrices:
                pm[:]*=2  
        return proj_matrices


    def generate_approximate_unknowns(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            proj_matrices = self.projection_matrices.T
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            unknowns = np.zeros(len(order_ids),dtype = complex)
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info(proj_matrices.shape)
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(proj_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                non_zero_mask = scalar_prod_Im_vm !=0
                unknowns[non_zero_mask] = scalar_prod_Im_vm[non_zero_mask]/abs(scalar_prod_Im_vm[non_zero_mask])
                #        log.info(scalar_prod_Im_vm)
                unknowns[~non_zero_mask]=1
                #tmp = np.concatenate((unknowns,unknowns[:0:-1].conj()))
                #unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns#np.zeros_like(unknowns)
            
            if self.use_SO_freedom:
                SO_order_id =self.get_SO_application_order()
                def function(intensity_harmonic_coefficients):
                    unknowns = approximate_unknowns(intensity_harmonic_coefficients)
                    unknowns[SO_order_id]=1
                    return unknowns
            else:
                function = approximate_unknowns
            
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            determinant = np.linalg.det
            loge = np.log
            n_orders = len(PDs)
            unknowns = tuple(np.zeros((len(PD),2*o+1),dtype = complex) for PD,o in zip(PDs,used_orders))
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                for unknown,PD,oid in zip(unknowns,PDs,order_ids):
                    I = intensity_harmonic_coefficients[oid]
                    matmul(*svd(PD @ I,full_matrices=False)[::2],out = unknown)  # PD @ Intensity is  B^\dagger A in a Procrustres Problem min|A-BR|
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                return unknowns
            if self.use_SO_freedom:
                radial_high_pass = self.opt.SO_freedom.radial_high_pass
                ranked_ids,ranked_orders,_ = i_tools.rank_projection_matrices(self.dimensions,proj_matrices,self.positive_orders,radial_points,radial_high_pass = radial_high_pass)
                SO_order_id = ranked_ids[0]
                SO_order = self.positive_orders[SO_order_id]
                max_order = self.positive_orders.max()
                ms_per_order = []
                for order in np.arange(max_order+1):
                    ms_per_order.append(np.concatenate((np.arange(order+1),np.arange(-order,0))))
                
                def function(intensity_harmonic_coefficients):
                    for ms,unknown,PD,I in zip(ms_per_order,unknowns,PDs,intensity_harmonic_coefficients):
                        u,s,vh = svd(PD @ I,full_matrices=False)
                        matmul(u,vh,out = unknown)  # PD @ Intensity is  B^\dagger A in a Procrustres Problem min|A-BR|
                    
                    u_SO = unknowns[SO_order_id]
                    u_SO[4,2] = u_SO[4,2].real                    #u_SO[4,2] = u_SO[4,2].real
                    #log.info(f'unknown SO_order imag part = {unknowns[SO_order_id][4,2].imag}')
                    return unknowns
            else:                
                function = approximate_unknowns
            
        return function
    
    def generate_coeff_projection_base(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id  = used_orders.get(0,False)
        zero_pos = np.argmax(np.array(tuple(used_orders.keys())) == 0)
        
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        #average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        #mask=self.mask_2d
        radial_mask = self.radial_mask
        #projection_matrices[zero_id]= np.abs(projection_matrices[zero_id])
        if dim == 2:
            where=np.where
            order_array=np.array(list(used_orders.values()))
            #order_array=np.concatenate((order_array,order_array[:0:-1]))
            new_intensity_coefficients=np.zeros((self.grid.shape[0],(self.grid.shape[1]+1)//2),dtype=complex)
            mult=np.multiply
            mask = np.zeros((self.grid.shape[0],(self.grid.shape[1]+1)//2),dtype=bool)
            for o_id in order_array:
                mask[:,o_id]=radial_mask[o_id]
            radial_mask2 = radial_mask[order_array].T
            #log.info('2d mask .shape = {}, 2d_radial_mask shape = {}, grid shape = {} len orders = {} \n\n\n'.format(mask.shape,radial_mask2.shape,self.grid.shape,order_array.shape))
            #projection_matrices=np.concatenate((projection_matrices,projection_matrices[:0:-1].conj()),axis=0)
            projection_matrices=projection_matrices.T
            
            qs = self.grid[:,0,0]
            sphere = spherical_formfactor(qs,radius = 10)
            if not isinstance(zero_id,bool):
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    new_intensity_coefficients[:]=intensity_harmonic_coefficients
                    new_intensity_coefficients[mask] = (projection_matrices*unknowns[None,:])[radial_mask2]
                    new_intensity_coefficients[radial_mask[zero_id],zero_id] = projection_matrices[radial_mask[zero_id],zero_pos]
                    return new_intensity_coefficients                
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    new_intensity_coefficients[:]=intensity_harmonic_coefficients
                    new_intensity_coefficients[mask] = (projection_matrices*unknowns[None,:])[radial_mask2]
                    return new_intensity_coefficients
        elif dim == 3:
            copy=np.array #array is faster than copy
            if not isinstance(zero_id,bool):
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[copy(coeff) for coeff in intensity_harmonic_coefficients]
                    for o_id in self.used_orders.values():
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask[o_id],...] = tmp_coeff[radial_mask[o_id],...]
                    projected_intensity_coefficients[zero_id][radial_mask[zero_id],...] = projection_matrices[zero_id][radial_mask[zero_id],...]
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[copy(coeff) for coeff in intensity_harmonic_coefficients]
                    for o_id in self.used_orders.values():
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask[o_id],...] = tmp_coeff[radial_mask[o_id],...]
                    return projected_intensity_coefficients
        return mtip_projection
    
    def generate_coeff_projection(self,coeff_projection):
        dim = self.dimensions
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        radial_mask = self.radial_mask
        number_of_particles = self.number_of_particles
        if dim == 2:
            def fixed_projection(intensity_harmonic_coefficients,unknowns):
                projected_intensity_coefficients = coeff_projection(intensity_harmonic_coefficients,unknowns)
                #projected_intensity_coefficients[zero_id][radial_mask[zero_id]]/=np.sqrt(number_of_particles[0])
                #log.info(f'number of particles scaling factor = {1/np.sqrt(number_of_particles[0])}')
                #log.info(f'out_array = {projected_intensity_coefficients[:,zero_id].shape} in array = {self.projection_matrices[zero_id].shape}')
                projected_intensity_coefficients[:,zero_id]/=np.sqrt(number_of_particles[0])
                return projected_intensity_coefficients
        elif dim == 3:
            def fixed_projection(intensity_harmonic_coefficients,unknowns):
                projected_intensity_coefficients = coeff_projection(intensity_harmonic_coefficients,unknowns)
                #projected_intensity_coefficients[zero_id][radial_mask[zero_id]]/=np.sqrt(number_of_particles[0])
                #log.info(f'number of particles scaling factor = {1/np.sqrt(number_of_particles[0])}')
                projected_intensity_coefficients[zero_id][:]/=np.sqrt(number_of_particles[0])
                return projected_intensity_coefficients
        return fixed_projection

    def generate_project_to_modified_intensity(self):
        if settings.general.cache_aware:
            L2_cache = settings.general.L2_cache
            func = self._generate_project_to_modified_intensity_cache_aware(L2_cache)
        else:
            func = self._generate_project_to_modified_intensity_default()
        return func
    
    def generate_project_to_fixed_intensity(self):
        if settings.general.cache_aware:
            L2_cache = settings.general.L2_cache
            func = self._generate_project_to_modified_intensity_cache_aware(L2_cache,use_fixed_intensity=True)
        else:
            func = self._generate_project_to_modified_intensity_default(use_fixed_intensity=True)
        return func
    def _generate_project_to_modified_intensity_default(self,use_fixed_intensity=False):
        zeros = np.zeros
        density_shape = self.grid[:].shape[:-1]
        sqrt = np.sqrt
        mult = np.multiply
        nabs = np.abs
        new_reciprocal_density = np.zeros(density_shape,dtype =complex)
        intensity_multipliers = np.zeros(new_reciprocal_density.shape , dtype= complex)
        temp = np.zeros(new_reciprocal_density.shape , dtype= float)
        if not use_fixed_intensity:
            def project_to_modified_intensity(reciprocal_density,square,new_intensity):            
                non_zero_mask = (square>=0)  & (new_intensity.real>=0)            
                #log.info('square dtype = {}'.format(square.dtype))
                temp[non_zero_mask] = new_intensity.real[non_zero_mask]/square[non_zero_mask].real
                np.sqrt(temp,out = intensity_multipliers)
                intensity_multipliers[~non_zero_mask] = 0            
                mult(reciprocal_density ,intensity_multipliers,out = new_reciprocal_density)
                #log.info("old intesity sum = {} new intensity sum = {}".format(np.sum(square),np.sum(new_intensity)))
                #log.info('nans = {} infs = {}'.format(np.isnan(new_reciprocal_density),np.isinf(new_reciprocal_density)))
                #new_reciprocal_density = reciprocal_density * sqrt(new_intensity.real/(reciprocal_density*reciprocal_density.conj()).real)            
                return new_reciprocal_density
        else:
            new_intensity2 = self._fixed_intensity
            def project_to_modified_intensity(reciprocal_density,square):
                #log.info(new_intensity2[0][:10,0,0])
                non_zero_mask = (square>=0) & (new_intensity2[0]>=0)            
                #log.info('square dtype = {}'.format(square.dtype))
                temp.real[non_zero_mask] = new_intensity2[0][non_zero_mask]/square[non_zero_mask].real
                np.sqrt(temp,out = intensity_multipliers)
                intensity_multipliers[~non_zero_mask] = 0            
                mult(reciprocal_density ,intensity_multipliers,out = new_reciprocal_density)
                #log.info("old intesity sum = {} new intensity sum = {}".format(np.sum(square),np.sum(new_intensity)))
                #log.info('nans = {} infs = {}'.format(np.isnan(new_reciprocal_density),np.isinf(new_reciprocal_density)))
                #new_reciprocal_density = reciprocal_density * sqrt(new_intensity.real/(reciprocal_density*reciprocal_density.conj()).real)            
                return new_reciprocal_density
        
        return project_to_modified_intensity
    def _generate_project_to_modified_intensity_cache_aware(self,L2_cache,use_fixed_intensity=False):
        data_shape = self.grid[:].shape[:-1]
        #split_id,step = get_L2_cache_split_parameters(data_shape,np.dtype(complex),L2_cache)
        return self._generate_project_to_modified_intensity_default(use_fixed_intensity = use_fixed_intensity)

    #scaled version to donatellies n particle determination

    def rank_projection_matrix_orders_2d(self):
        radial_high_pass=self.opt.SO_freedom.get('radial_high_pass',0.2)
        radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
        radial_high_pass=0
        if not isinstance(radial_high_pass_index,bool):
            radial_high_pass=radial_high_pass_index
        radial_points=self.radial_points[radial_high_pass:]


        orders = self.projection_orders
        #log.info(orders)
        order_ids= self.projection_order_ids_local
        even_order_mask= orders%2 == 0
        non_zero_mask= orders != 0
        order_mask=even_order_mask*non_zero_mask
        relevant_orders=orders[order_mask]
        #relevant_order_ids = order_ids[order_mask]

        projection_vector=self.projection_matrices[order_mask,radial_high_pass_index:].T        
        

        weighted_vect=projection_vector*radial_points[:,None]#/(np.concatenate(([100.0,100.0,6,8],harmonic_orders[4:])))[None,:]
        #log.info('weighted_vect shape =\n {}'.format(weighted_vect.shape))
        metric = np.mean(np.abs(weighted_vect),axis=0)#/orders[order_mask]
        #log.info(metric)
        sorted_indices = np.argsort(metric)[::-1]
        #log.info(sorted_indice)
        SO_order_indices=order_mask.nonzero()[0][sorted_indices]
        SO_orders=orders[SO_order_indices]
        return SO_order_indices,SO_orders,sorted_indices

    def get_SO_application_order(self):
        dim=self.dimensions
        if dim ==2 :
            SO_order_indices,SO_orders,sorted_indices=self.rank_projection_matrix_orders_2d()            
            #log.info('Selected SO_order_ids = {}'.format(SO_order_indices))
        elif dim == 3:
            raise NotImplementedError
        return SO_order_indices[0]
            
    def generate_apply_SO_freedom_2D(self,opt):
        dim=self.dimensions
        if dim ==2 :
            ordes_dict=self.used_orders
            orders=np.array(tuple(orders_dict))
            order_ids=tuple(orders_dict.values())
            
            radial_high_pass=opt['SO_freedom'].get('radial_high_pass',0.2)
            radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
            radial_points=self.radial_points

            even_order_mask= orders%2 == 0
            if use_averaged_intensity:
                non_zero_mask= orders != 0
            else:
                non_zero_mask=True
            order_mask=even_order_mask*non_zero_mask
            relevant_orders=orders[order_mask]
            max_order=np.max(relevant_orders)
        
            projection_vector=self.projection_matrices[radial_high_pass_index:,order_mask]
            
            #pres_cart.present(np.swapaxes(np.abs(projection_vector),0,1),scale='log')
            #pres_1d.present(np.sum(np.abs(projection_vector),axis=0))
            sorted_projection_indices=self.rank_orders(projection_vector,radial_points[radial_high_pass_index:],relevant_orders)
            #log.info('sorted_projection_indices={}'.format(sorted_projection_indices))
            SO_order_index=order_mask.nonzero()[0][sorted_projection_indices[0]]
            #log.info('nonzero = {}'.format(order_mask.nonzero()[0]))
            log.info('SO order Index={}'.format(SO_order_index))
            #   log.info('first_order_index={}'.format(first_order_index))
            #   log.info('first_order={}'.format(first_order))
            
            def apply_SO_freedom(unknowns):
                #log.info('unknowns shape={}'.format(unknowns.shape))
                unknowns[SO_order_index]=1
                return unknowns
        elif dim == 3:
            raise NotImplementedError
        return apply_SO_freedom

    def generate_remaining_SO_projection(self,radial_high_pass=0.2):
        dim=self.dimensions
        if dim == 2:
            remaining_SO_projection=self.generate_remaining_SO_projection_2D(radial_high_pass=radial_high_pass)
        elif dim == 3:
            def remaining_SO_projection(harmonic_coefficients,fxs_unknowns):            
                raise NotImplementedError()        
        return remaining_SO_projection
    
    def generate_remaining_SO_projection_2D(self,radial_high_pass=0.2):
        radial_high_pass_index = int((len(self.radial_points)-1)*radial_high_pass)
        projection_vectors = self.projection_matrices
        n_angular_points=self.grid.shape[1]
        projection_orders=np.concatenate((np.arange(int(n_angular_points/2)+1),-1*np.arange(int(n_angular_points/2)+n_angular_points%2)[:0:-1]))
        
        pos_orders = self.positive_orders
        positive_harmonic_orders=np.array(tuple(self.used_orders.keys()))
        order_ids= np.array(tuple(self.used_orders.values()))
        even_order_mask=positive_harmonic_orders%2==0
        non_zero_mask=positive_harmonic_orders!=0
        order_mask=even_order_mask*non_zero_mask
        harmonic_orders=positive_harmonic_orders[order_mask]
        harmonic_orders_ids=order_ids[order_mask]
        max_order=np.max(harmonic_orders)
            
        SO_order_indices,SO_orders,sorted_order_indices=self.rank_projection_matrix_orders_2d() 
        #sorted_order_indices=self.rank_projection_matrix_orders_2d(projection_vector,self.radial_points[radial_high_pass_index:])
        
        #log.info('sorted_order_indices={}'.format(SO_order_indices))
        first_order_index=SO_order_indices[0]
        first_order=SO_orders[0]#harmonic_orders[first_order_index]
        #log.info('ranked_orders={}'.format(harmonic_orders[sorted_order_indices]))
    
        remaining_rotations=first_order
        current_order=first_order    
        free_orders_mask=True
        angle_coeffs=()
        angles=()
        order_indices=()
        gcds=()
        while remaining_rotations>2:
            #find non invariant orders
            order_multiples=np.arange(current_order,max_order+1,current_order)
            multiple_indices=np.where(np.isin(harmonic_orders,order_multiples))
            free_orders_mask*=~np.isin(sorted_order_indices,multiple_indices)
            #log.info('remaining_orders={}'.format(harmonic_orders[sorted_order_indices[free_orders_mask]]))
            if free_orders_mask.any()==False:
                break
            else:
                #select next order
                current_order_index=sorted_order_indices[free_orders_mask][0]
                current_order=harmonic_orders[current_order_index]
                #log.info('current order ={}'.format(current_order))
                #calculate remaining rotations
                gcd=np.gcd(remaining_rotations,current_order)
                #log.info('gcd={}'.format(gcd))
                #use rotational freedom on current_order
                n_independent_rotations=remaining_rotations/gcd
                #log.info('n independent rotations={}'.format(n_independent_rotations))
                smallest_angle=2*np.pi/n_independent_rotations
                smallest_angle_coeff=np.argmin((np.arange(1,n_independent_rotations)*current_order/gcd)%n_independent_rotations)+1
    
                order_indices+=(current_order_index,)
                angle_coeffs+=(smallest_angle_coeff,)
                angles+=(smallest_angle,)
                gcds+=(gcd,)
                remaining_rotations=gcd
        
        def apply_SO_freedom(harmonic_coefficients,fxs_unknowns):
            #log.info('fxs unknowns shape ={}'.format(fxs_unknowns.shape))
            phases=(-1.j*np.log(fxs_unknowns[order_mask])).real
            #log.info('complete phases ={}'.format(-1.j*np.log(fxs_unknowns[::2])))
            #log.info('phases ={}'.format(phases))
            rotation_phase=0
            for order_index,angle,angle_coeff,gcd in zip(order_indices,angles,angle_coeffs,gcds):
                rotation_phase-=(phases[order_index]//angle)*angle_coeff*angle/gcd
                #log.info('order={} phase={}'.format(harmonic_orders[order_index],phases[order_index]))
                #log.info('min angle*gcd={}'.format(angle))
                #log.info('rotated phase={}'.format((phases[order_index]+harmonic_orders[order_index]*rotation_phase)%(2*np.pi)))
            
            harmonic_coefficients*=np.exp(1.j*projection_orders*rotation_phase)
            return harmonic_coefficients
        return apply_SO_freedom


    # Number of particle estimation experimental not working jet
    def change_n_particles(self,N):
        '''Setter for n_particles attribute'''
        self.n_particles = N

    def get_number_of_particles(self):
        return self.number_of_particles
    
    def apply_particle_number_scaling(self,projection_matrices,zero_id,average_intensity):
        '''
        Let N be the nunmber of particles. this method scales the projection_matrices $V_l$ by $1/\sqrt(N)$ for $l\neq 0$ and $1/N$ for $l = 0$.
        The averaged intensity is scaled by $1/N$, since it is proportional to $V_0$
        '''
        N = settings.project.projections.reciprocal.number_of_particles
        projection_matrices = [matrix/np.sqrt(N) for matrix in projection_matrices]
        projection_matrices[zero_id] = projection_matrices[zero_id]/np.sqrt(N)
        average_intensity = average_intensity/N
        return projection_matrices,average_intensity
            
    def generate_number_of_particles_porjection(self):
        p_opt = settings.project.projections.reciprocal
        zero_id = self.used_orders.get(0,'')
        radial_mask=self.radial_mask[zero_id]
        
        proj_matrices = self.projection_matrices
        if p_opt.use_averaged_intensity:
            I00 = np.abs(self.average_intensity.data.real)
        else:
            I00 = np.abs(proj_matrices[0].flatten().real)
        I00y00=I00/(2*np.sqrt(np.pi))
        N_space = p_opt.number_of_particles.scan_space
        #Ns = np.linspace(*N_space)
        Ns_sqrt = np.linspace(*np.sqrt(N_space[:-1]),N_space[-1])
        Ns = Ns_sqrt**2
        #Ns_lin = np.linspace(*N_space)
        #Ns_lin_sqrt = np.sqrt(Ns_lin)
        #log.info("N_max = {} Ns_sqrt max = {}".format(Ns.max(),Ns_sqrt.max()))
        #log.info("N_min = {} Ns_sqrt min = {}".format(Ns.min(),Ns_sqrt.min()))
        summands = (1/Ns_sqrt-1)[:,None]*I00y00[None,radial_mask]
        #summands = (1/Ns_sqrt-1)[:,None]*I00y00[None,radial_mask]
        nsum=np.sum
        n_pixels=np.prod(self.grid[radial_mask].shape[:-1])
        estimate_n_particles = i_tools.estimate_number_of_particles
        initial_n_particles = p_opt.number_of_particles.initial
        if p_opt.number_of_particles.project:
            def particle_number_projection(I):
                #log.info('I shape = {}'.format(I.shape))
                scaled_I = I[None,radial_mask,...] + summands[:,:,None,None]
                neg_fractions = np.sum(scaled_I<0,axis=(1,2,3))/n_pixels
                grad = np.gradient(neg_fractions,Ns_sqrt)            
                inflection_id = np.argmax(grad)
                self.number_of_particles_dict = {'number_of_particles':Ns[inflection_id],'negative_fraction':neg_fractions,'gradient':grad}
                self.number_of_particles[0] = Ns[inflection_id]
                #log.info('number of particles = {}'.format(self.number_of_particles))
                I[radial_mask] = scaled_I[inflection_id]
                I[I<0]=0
                del(scaled_I)            
                return I
        else:
            def particle_number_projection(I):
                #log.info('I shape = {}'.format(I.shape))
                scaled_I = I[None,radial_mask,...] + summands[:,:,None,None]
                neg_fractions = np.sum(scaled_I<0,axis=(1,2,3))/n_pixels
                grad = np.gradient(neg_fractions,Ns)            
                inflection_id = np.argmax(grad)
                self.number_of_particles_dict = {'number_of_particles':Ns[inflection_id],'negative_fraction':neg_fractions,'gradient':grad}
                self.number_of_particles[0] = Ns[inflection_id]
                #log.info('number of particles = {}'.format(self.number_of_particles))
                #I[radial_mask] += (1/np.sqrt(initial_n_particles)-1)*I00y00[radial_mask]
                #I[I<0]=0
                del(scaled_I)            
                return I
            
        return particle_number_projection
    
    def generate_number_of_particles_porjection_gpu(self):
        p_opt = settings.project.projections.reciprocal
        zero_id = self.used_orders.get(0,'')
        radial_mask=self.radial_mask[zero_id]
        proj_matrices = self.projection_matrices
        if p_opt.use_averaged_intensity:
            I00 = np.abs(self.average_intensity.data.real)
        else:
            I00 = np.abs(proj_matrices[0].flatten().real)
        I00y00=I00/(2*np.sqrt(np.pi))
        N_space = p_opt.number_of_particles.scan_space
        Ns = np.linspace(*N_space)
        summands = (1/np.sqrt(Ns)-1)[:,None]*I00y00[None,radial_mask]
        nsum=np.sum
        n_pixels=np.prod(self.grid[radial_mask].shape)
        estimate_n_particles = i_tools.estimate_number_of_particles


        cld = Multiprocessing.load_openCL_dict()
        cl = cld['cl']
        ctx = cld['context']
        queue = cld['queue']

        kernel = cl.Program(ctx, """
        __kernel void
        count_negative(__global double* I, 
        __global double* summands, 
        __global double* neg_counts, 
        long nN,long nq,long ntheta, long nphi)
        {
        
        long N = get_global_id(0); 
        long q = get_global_id(1); 
        long theta = get_global_id(2); 
        long phi = get_global_id(3); 
        
        // value stores the element that is 
        // computed by the thread
        double neg_count = 0;
        for (int phi = 0; phi < nphi; ++phi)
        {
        double sum = I[q*ntheta*nphi+theta*nphi+phi] + summands[N*nq+q]; 
        neg_count += (1.0-sum/fabs(sum)); // 2 if sum is negative, 0 else
        }
        // Write the matrix to device memory each 
        // thread writes one element
        neg_counts[N*nq*ntheta + q*ntheta + theta] = neg_count/2;
        }    

        __kernel void 
        floatSum(__global float* inVector, __global float* outVector, const int inVectorSize,const long nN, __local float* resultScratch){
        int N = get_global_id(0);
        int gid = get_global_id(1);
        int wid = get_local_id(0);
        int wsize = get_local_size(0);
        int grid = get_group_id(0);
        int grcount = get_num_groups(0);
    
        int i;
        int workAmount = inVectorSize/grcount;
        int startOffest = workAmount * grid + wid;
        int maxOffest = workAmount * (grid + 1);
        if(maxOffset > inVectorSize){
            maxOffset = inVectorSize;
        }
        resultScratch[nN*wsize + wid] = 0.0;
        for(i=startOffest;i<maxOffest;i+=N*wsize){
                resultScratch[N*wsize + wid] += inVector[N*wsize + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    
        if(gid == 0){
        for(i=1;i<wsize;i++){
        resultScratch[0] += resultScratch[i];
        }
        outVector[grid] = resultScratch[0];
        }

        __kernel
        void reduce(__global float* buffer,
        __local float* scratch,
        __const int length,
        __global float* result) {
        int global_index = get_global_id(0);
        float accumulator = INFINITY;
        // Loop sequentially over chunks of input vector
        while (global_index < length) {
        float element = buffer[global_index];
        accumulator = (accumulator < element) ?
        accumulator : element;
        global_index += get_global_size(0);
        }
        // Perform parallel reduction
        int local_id = get_local_id(0)
        scratch[local_id]=accumulator;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int offset = get_local_size(0)/2;
        offset > 0;
        offset = offset / 2) {
        if (local_id<offset){
        double other = scratch[local_id+offset];
        double mine = scratch[local_id];
        scratch[local_id]= (mine<other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        }
        }
    
        sum_space(__global double* neg_count_phi, 
        __global double* neg_count, 
        long nN,long nq,long ntheta, long nphi)
        {
         // Get the index of the current element to be processed
        int N = get_global_id(0);
        int i = get_global_id(1)*2;
        int locali = get_local_id(0);
        int2 va = vload2(i, numbers);
        int2 vb = vload2(i+1, numbers);
        int2 vc = va + vb;
        numReduce[locali] = vc[0] + vc[1];


        long N = get_global_id(0); 
        double sum = 0;
        for (int q = 0; q < nq; ++q)
        {
        for (int theta = 0; theta < ntheta; ++theta)
        {
        sum +=neg_count_phi[N*nq*ntheta + q*ntheta + theta]
        }
        }
        // value stores the element that is 
        // computed by the thread
        double neg_count = 0;
        
        double sum = I[q*ntheta*nphi+theta*nphi+phi] + summands[N*nq+q]; 
        neg_counts[N] += (1.0-sum/fabs(sum)); // 2 if sum is negative, 0 else
        
        // Write the matrix to device memory each 
        // thread writes one element
         = neg_count/2;
        }    
        """).build()
        count_negative = kernel.count_negative
        sum_space = kernel.sum_space
        
        count_negative.set_scalar_arg_dtypes([None,None,None,np.int64,np.int64,np.int64,np.int64])
        
        local_range = None
        nN = len(Ns)
        nq, ntheta, nphi = self.grid[:].shape[:-1]
        nq = np.sum(radial_mask)
        global_range_count = (nN,nq,ntheta)
        global_range_sum = (nN,)
        neg_counts = np.zeros(nN,dtype = float)
        I_mock = np.zeros((nq,ntheta,nphi),dtype = float)

        mf = cl.mem_flags
        summands_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(summands))
        I_buff = cl.Buffer(ctx , mf.READ_WRITE, size=I_mock.nbytes)
        out_buff = cl.Buffer(ctx , mf.READ_WRITE, size=neg_counts.nbytes)

        def particle_number_projection(I):
            #log.info('I shape = {} I dtype = {}'.format(I[radial_mask].shape,I.dtype))
            start = time.time()
            cl.enqueue_copy(queue,I_buff,np.ascontiguousarray(I[radial_mask].real))
            count_negative(queue,global_range,local_range,I_buff,summands_buff,out_buff,nN,nq,ntheta,nphi)
            cl.enqueue_copy(queue, out_buff,neg_counts)
            log.info('GPU took {} seconds'.format(time.time()-start))
            grad = np.gradient(neg_counts/n_pixels,Ns)        
            inflection_id = np.argmax(grad)
            self.number_of_particles = Ns[inflection_id]
            #log.info('number of particles = {}'.format(self.number_of_particles))
            I[radial_mask] = I[radial_mask]+summands[inflection_id,:,None,None]
            I[I<0]=0
            return I
        return particle_number_projection




    

# particle number approximation and projection as described in
# K. Pande et.al. PNAS 2018 'Ab initio structure determination from experimental fluctuation X-ray scattering data'
def gnerate_estimate_particle_number(radial_grid,projection_matrices,used_orders):
    '''
    Particle number approximation
    K. Pande et.al. PNAS 2018 'Ab initio structure determination from experimental fluctuation X-ray scattering data'
    :param radial_grid: (N_q) shaped array of radial sampling points.
    :type numpy.ndarray: 
    :param projection_matrices: list of length (L_max+1) containing the $(N_q,min(2*l+1,N_q))$ shaped projection matrices $V_l$
    :type list: of numpy.ndarrays
    :param used_orders: dictionary linking the harmonic orders(as key) to their ids (as value) 
    :type dict:  
    :return N: estimated Number of particles for which the B_l coefficients where calculated.
    :rtype float: 
    '''
    B_l = i_tools.projection_matrices_to_deg2_invariant_3d(projection_matrices)[1:]
    B_l_diag = np.diagonal(B_l,axis1=-2,axis2=-1)
    G = np.sum(B_l_diag**2*(radial_grid**2)[None,:])
    B_l_diag_q2 = np.diagonal(B_l,axis1=-2,axis2=-1)*(radial_grid**2)[None,:]
    calc_Bl=i_tools.spherical_harmonic_coefficients_to_deg2_invariant
    order_ids = list(used_orders.values())
    def estimate_particle_number(I_coeff):
        B_l_guess = calc_Bl(I_coeff)[order_ids][1:]
        B_l_guess_diag = np.diagonal(B_l_guess,axis1=-2,axis2=-1)                
        N = G/np.sum(B_l_guess_diag*B_l_diag_q2)
        log.info('Estimated number of particles = {}'.format(N))
        return 1.0
    return estimate_particle_number

    
    
#positions and point inversion projections
def generate_fix_point_inversion(radial_low_pass = 0.1):
    
    def fix_point_inversion(scattering_amplitude):
        'Routine that corrects point inversion if given scattering amplitudes which only differ by point inversion (i.e. complex conjugation) from each other.'
        phases = scattering_amplitude/np.abs(scattering_amplitude)
        shape = scattering_amplitude.shape
        inversion_indicator = np.sum(phases.imag[:int(shape[0]*radial_low_pass),:shape[1]//2])
        log.info('inversion indicator ={} radial_low_pass = {}'.format(inversion_indicator,radial_low_pass))
        if inversion_indicator<0:
            scattering_amplitude=scattering_amplitude.conjugate()
        return scattering_amplitude
    return fix_point_inversion
def generate_negative_shift_operator(reciprocal_grid,fourier_type):
    dim=reciprocal_grid.n_shape[0]
    pre_exponent=2*np.pi
    pi=np.pi
    if (fourier_type=='Zernike') or (fourier_type=='trapz'):
        pre_exponent=1

    cart_grid = spherical_to_cartesian(reciprocal_grid)
    def negative_shift(reciprocal_density,vector):
        cart_vect = spherical_to_cartesian(vector)
        log.info('cart_vector = {}'.format(cart_vect))
        
        phases = np.exp(-1.j*pre_exponent*(cart_grid*cart_vect).sum(axis=-1))
        log.info('phases shape = {}'.format(phases.shape))
        reciprocal_density*=phases
        return reciprocal_density        
    return negative_shift

def generate_shift_by_operator(grid,opposite_direction=False):
    dim=grid[:].shape[-1]
    pi = np.pi
    if opposite_direction:
        prefactor = -1
    else:
        prefactor = 1
    if dim == 2:
        cart_grid = spherical_to_cartesian(grid)
        def shift_by(reciprocal_density,vector):
            #log.info('vector = {} /n'.format(vector))
            cart_vect = spherical_to_cartesian(vector)
            #log.info('cart shift vect = {}'.format(-cart_vect*prefactor))
            phases = np.exp(-1.j*prefactor*(cart_grid*cart_vect).sum(axis=-1))
            reciprocal_density*=phases
            return reciprocal_density   
    elif dim == 3:
        cart_grid = spherical_to_cartesian(grid)
        def shift_by(reciprocal_density,vector):
            #log.info('vector = {}'.format(vector))
            cart_vect = spherical_to_cartesian(vector)
            #log.info('cart shift vect = {}'.format(-cart_vect*prefactor))
            phases = np.exp(-1.j*prefactor*(cart_grid*cart_vect).sum(axis=-1))
            reciprocal_density*=phases
            return reciprocal_density        
    return shift_by


