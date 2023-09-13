import time
import numpy as np
import numpy.ma as mp
import scipy.integrate as spIntegrate
from scipy.linalg import pinv2
from scipy.linalg import qr
from itertools import repeat
import logging


import xframe.library.pythonLibrary as pyLib
from xframe.library.pythonLibrary import DictNamespace,measureTime
import xframe.library.physicsLibrary as pLib
from xframe.library.gridLibrary import NestedArray
from xframe.library.gridLibrary import ReGrider,SampledFunction
from xframe.library.mathLibrary import SphericalIntegrator
import xframe.library.mathLibrary as mLib
from xframe.library.mathLibrary import polar_spherical_dft_reciprocity_relation_radial_cutoffs
from xframe.library.physicsLibrary import energy_to_wavelength
from xframe.library import units
from xframe.analysis.interfaces import DatabaseInterface,PresenterInterface
from xframe.presenters.matplolibPresenter import heatPolar2D
from xframe.presenters.matplolibPresenter import heat2D
from xframe.presenters.matplolibPresenter import plot1D
from .ft_grid_pairs import radial_grid_func_zernike

from .ft_grid_pairs import get_grid
from .fourier_transforms import generate_ft
from .harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary import fxs_invariant_tools as i_tools
from xframe import settings
from xframe import database
import xframe.Multiprocessing
from xframe import Multiprocessing

pres=heatPolar2D()
pres_cart=heat2D()
pres_1d=plot1D()
log=logging.getLogger('root')



def generate_other_real_projections(projections_specifier,metadata):
    projections=[]
    generators={
        'positivity':generatePositivityConstraint,
        'upper_bound':generate_upper_bound_consraint,
        'assert_real':generate_complex_part_projection,
        'fix_total_density':generate_total_desity_projection
    }
    for key in projections_specifier.apply:
        projections.append(generators[key](projections_specifier[key],metadata))
    #positivity_specifier=projections_specifier.get('positivity',False)
    #upper_bound_specifier=projections_specifier.get('upper_bound',False)
    #complex_part_specifier=projections_specifier.get('complex_part',False)
    #if not isinstance(complex_part_specifier,bool):
    #    complex_part_projection=generate_complex_part_projection(complex_part_specifier)
    #    projections.append(complex_part_projection)
    #if not isinstance(positivity_specifier,bool):        
    #    positivityProjection=generatePositivityConstraint(positivity_specifier)
    #    projections.append(positivityProjection)
    #if not isinstance(upper_bound_specifier,bool):
    #    upper_bound_projection=generate_upper_bound_consraint(upper_bound_specifier)
    #    projections.append(upper_bound_projection)
    
    def other_real_projections(data):
        for projection in projections:
            data=projection(data)
        return data
    return other_real_projections

def generate_upper_bound_consraint(options,metadata):
    bound=options['bound_value']
    def upper_bound_projection(data):
        data.real=np.minimum(data.real,bound)
#        log.info('old_real_values={} \n new real values={}'.format(data.array.real.flatten()[:10],new_array.flatten()[:10]))
        return data
    return upper_bound_projection
    
        
def generatePositivityConstraint(options,metadata):
    threshold_relToMax=options['threshold_rel_to_max']
    def positivityProjection(dataGrid):
#        log.info('data before pos proj = {}'.format(dataGrid[:]))
        data=dataGrid.real
        if threshold_relToMax==0:
#            pres.present(data<0,layout={'title':'positivity Mask'})
            data=np.where(data<0,0,data)
            dataGrid.real=data

        else:
            maxValue=np.max(data)
            if maxValue>=0:
                absThreshold=-1*maxValue*threshold_relToMax
                data=np.where(data<absThreshold,0,data)
                dataGrid.real=data
            else:
                log.error('Error: Estimate for electron density is entirely negative! setting all values to 0.')
                dataGrid[:]=0
#        log.info('data after pos proj = {}'.format(dataGrid[:]))
        return dataGrid
    return positivityProjection

def generate_complex_part_projection(options,metadata):
    threshold_to_real_max=options['threshold_rel_to_real_max']
    def complex_part_projection_relative(data_grid):
        data=data_grid
        real_max=np.max(data.real)
        if real_max>=0:
            complex_part=np.clip(data.imag,-real_max*threshold_to_real_max,real_max*threshold_to_real_max)
            data.imag=complex_part
        else:
            log.error('Error: Estimate for electron density is entirely negative! setting all complex values to 0.')
            data.imag=0
        return data
            
    def complex_part_projection_absolut(data_grid):
        data_grid.imag=0
        return data_grid

    if threshold_to_real_max==0:
        selected_proj=complex_part_projection_absolut
    else:
        selected_proj=complex_part_projection_relative
    return selected_proj

        
def project_to_real_component(density_grid):
    density_grid=density_grid.real
    return density_grid

def generate_real_support_projection(support_mask):    
    #True if datapoint is in support
#    log.info('dataGrid non zero= {}'.format((dataGrid.array!=0).any()))
#    log.info('support mask not False= {}'.format((supportMask==True).any()))
#    pres.present(~supportMask,layout={'title':'1 values are set to zero'})
#    log.info('supportMask=\n{}'.format(support_mask))
    def real_support_projection(data):
        data[~support_mask]=0    
        return data
    return real_support_projection

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
    pres=heatPolar2D()
    #pres.present(auto_correlation.data.array,grid=realGrid)
    #pres.present(supportMask,grid=realGrid)
    return supportMask

def generate_total_desity_projection(opt,metadata):
    integrated_intensity = metadata['integrated_intensity']
    real_grid = metadata['real_grid']
    integrator=mLib.SphericalIntegrator(real_grid.array)
    def project_density(density):
        total_squared_density = integrator.integrate(np.abs(density)**2)
        log.info('total_squared_density = {}, total_intensity = {}'.format(total_squared_density,integrated_intensity))
        #log.info('total_intensity = {}'.format(integrated_intensity))
        #density*= np.sqrt(integrated_intensity/total_squared_density)
        return density
    return project_density

def generate_RealspaceProjection(supportMask,non):
    realspaceProjection_WithoutSupport=generateFXS_RealspaceProjection_WithoutSupport(paramDict)
    def FXS_RealspaceProjection(dataGrid):
        newDataGrid=Grid.copy(dataGrid)
        newDataGrid=realspaceProjection_WithoutSupport(newDataGrid)
        newDataGrid=supportProjection(supportMask,newDataGrid)
#        log.info('realDataGrid={}'.format(newDataGrid[:]))
        return newDataGrid
    return FXS_RealspaceProjection



def projectToModifiedIntensities(reciprocalDensity,newIntensity):
#    log.info('reciprocal Density ={}'.format(reciprocalDensity.array))
    #        log.info('reciprocalDensity dimension={}'.format(reciprocalDensity.dimension))
    positiveAmplitude=np.sqrt(np.where(newIntensity>=0,newIntensity,0))
    #log.info('pos_amplitude is positive={}'.format((positiveAmplitude.real>=0).all() ))
    #log.info('pos_amplitude does not contain imag values={}'.format((positiveAmplitude.imag==0).all() ))
    shape=reciprocalDensity.shape
    #        normOfDensity=reciprocalDensity.apply(np.linalg.norm).array
    normOfDensity=np.abs(reciprocalDensity)
    normOfDensity=np.where(normOfDensity==0,1,normOfDensity)
    #        log.info('norm of Density={}'.format(normOfDensity))
    #        pres.present(np.abs(reciprocalDensity.array),layout={'title':'original reciprocal density'},scale='log')
    #        pres.present(np.abs(reciprocalDensity.array/normOfDensity),layout={'title':'normed original reciprocal density'},scale='log')
    new_reciprocal_density=reciprocalDensity/normOfDensity*positiveAmplitude
    #        log.info('new Array={}'.format(newArray))
    return new_reciprocal_density
    

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
    eigenValues=listOfEigenvectorsAndValues[:,1].astype(np.complex)
    eigenVectors=np.array(tuple(listOfEigenvectorsAndValues[:,0]),dtype=np.complex)
    projection_vector=np.swapaxes(eigenVectors,0,1)*np.sqrt(eigenValues.astype(np.complex))
    fxs_data.projection_vector=projection_vector
    return fxs_data

def calculate_fxs_projection_vector(fxs_data,reciprocal_projection_specifier):
    positive_harmonic_orders=reciprocal_projection_specifier['positive_harmonic_orders']
    log.info('projection orders={}'.format(positive_harmonic_orders))
    bCoefficients=fxs_data.bCoeff
    
    def firstEigenvectorAndValue(expansionCoefficient):
        matrix=bCoefficients[:,:,expansionCoefficient]
#        log.info('matrix of shape {} to take eigen values from = \n{}'.format(matrix.shape,matrix))
        eigValues,eigVectors=np.linalg.eigh(matrix)
        max_arg=eigValues.argmax()
        
        return [eigVectors[:,max_arg],eigValues[max_arg]]
    listOfEigenvectorsAndValues=np.array(list(map(firstEigenvectorAndValue,positive_harmonic_orders)),dtype=np.object)    
    eigenValues=listOfEigenvectorsAndValues[:,1].astype(np.complex)
    log.info('lambdas={}'.format(eigenValues))
    eigenVectors=np.array(tuple(listOfEigenvectorsAndValues[:,0]),dtype=np.complex)
    projection_vector=np.swapaxes(eigenVectors,0,1)*np.sqrt(eigenValues.astype(np.complex))    
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
    average_intensity=specifier['fxs_data'].aInt.data.astype(np.complex)
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

    def fxs_projection(intensity_harmonicCoefficients,unknowns):
        projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()
        projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_vector*unknowns
        #apply mask
        intensity_harmonic_coefficients=np.where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(np.complex)
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
        opt = settings.analysis
        self.dimensions = data['dimensions']
        self.xray_wavelength = data['xray_wavelength']
        self.average_intensity = data['average_intensity']
        #log.info('aint type ={}'.format(type(self.average_intensity)))
        self.data_radial_points = data['data_radial_points'][:]
        self.data_angular_points = data['data_angular_points'][:]
        self.data_max_q = np.max(self.data_radial_points)
        self.data_min_q = np.min(self.data_radial_points)
        #self.pi_in_q = data.get('pi_in_q',False)
        self.pi_in_q = data['pi_in_q']
        self.max_order=data['max_order']
        self.data_projection_matrices = data['data_projection_matrices']
        
    def __init__(self,grid,data):
        self.load_data(data)
        self.integrated_intensity = np.trapz(self.average_intensity.data * self.data_radial_points**2 , x = self.data_radial_points,axis = 0)*2*np.sqrt(np.pi)
        opt = settings.analysis.projections.reciprocal        
        self.opt = opt
        self.grid=grid        
        self.radial_points=grid.__getitem__((slice(None),)+(0,)*self.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        self.max_q = np.max(self.radial_points)
        self.input_is_halfcomplex=opt.input_is_halfcomplex
        self.positive_orders = np.arange(self.max_order+1)
        self.used_order_ids = opt.used_order_ids
        self.used_orders = {order:id for (order,id) in zip(self.positive_orders,self.used_order_ids)}

        use_SO_freedom=opt.SO_freedom.use
        
        self.projection_matrices=self._regrid_data()
        self.projection_matrices=self.modify_projection_matrices(opt)
        self.assert_projection_matrices_have_right_shape()
        
        self.radial_mask=self.generate_radial_mask(opt.q_mask)
        self.order_mask=self.generate_order_mask()
        self.mask_2d=self.radial_mask[:,None]*self.order_mask[None,:]
        mtip_projection=self.generate_coeff_projection()
        self.mtip_projection = mtip_projection
        self.mtip_projection_fixed = self.generate_coeff_projection_fixed(mtip_projection)
            
        self.approximate_unknowns=self.generate_approximate_unknowns()

        if use_SO_freedom:
            self.apply_SO_freedom=generate_apply_SO_freedom(opt)
            def calc_unknowns(intensity_harmonic_coefficients):
                return apply_SO_freedom(approximate_unknowns(intensity_harmonic_coefficients))
            self.calc_unknowns=calc_unknowns
        else:
            self.calc_unknowns=self.approximate_unknowns
            
        if opt.number_of_particles.GPU:            
            self.particle_number_projection = self.generate_number_of_particles_porjection_gpu()
        else:
            self.particle_number_projection = self.generate_number_of_particles_porjection()
        
        #log.info('init end projection matrices shape = {}'.format(self.projection_matrices[-1].shape))
        self.number_of_particles = opt.number_of_particles.initial
        self.number_of_particles_dict = {'number_of_particles':self.number_of_particles,'negative_fraction':[],'gradient':[]}
        self.project_to_modified_intensity = self.generate_project_to_modified_intensity()
        
    def assert_projection_matrices_have_right_shape(self):
        try:
            n_radial_points_grid = len(self.radial_points)
            n_radial_points_projection_matrix = self.projection_matrices[0].shape[0]
            assert n_radial_points_grid == n_radial_points_projection_matrix,'Mismatch between th number of radial sampling points in reconstruction grid {}  and projection data {}. Abort reconstruction !'.format(n_radial_points_grid,n_radial_points_projection_matrix)
        except AssertionError as e:
            log.error(e)
            raise e
            
    def extract_used_options(self):
        fourier_opt = settings.analysis.fourier_transform
        rp_opt = settings.analysis.projections.reciprocal
        opt = DictNamespace(
            pos_orders = fourier_opt.pos_orders,
            **rp_opt
        )
        return opt
    def _regrid_data(self):
        dim = self.dimensions
        order_ids=list(self.used_orders.values())
        have_same_shapes = self.radial_points.shape == self.data_radial_points.shape
        projection_matrices = self.data_projection_matrices
        needs_regridding = True
        interpolation_type = self.opt.regrid.interpolation
        if have_same_shapes:
            if (self.data_radial_points == self.radial_points).all():
                needs_regridding = False
        log.info('needs regridding = {}'.format(needs_regridding))
        log.info('initial projection matrix shape = {}'.format(projection_matrices[-1].shape))   
        if needs_regridding:
            r_pt=NestedArray(self.radial_points[:,None],1)
            data_r_pt=NestedArray(self.data_radial_points[:,None],1)
            #log.info('n new points={} n old points ={}'.format(len(r_pt[:]),len(data_r_pt[:])))
            if dim == 2:                
                self.average_intensity.regrid(r_pt)
                projection_matrices = ReGrider.regrid(self.data_projection_matrices[...,order_ids],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type})
            elif dim == 3:
                self.average_intensity.regrid(r_pt)
                data_projection_matrices=self.data_projection_matrices
                projection_matrices = tuple(ReGrider.regrid(data_projection_matrices[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':interpolation_type}) for o_id in order_ids)
        log.info('regrided projection matrices shape = {}'.format(projection_matrices[-1].shape))
        return projection_matrices

    
    def generate_order_mask(self):
        dim=self.dimensions
        positive_orders=self.positive_orders
        used_orders=self.used_orders
        
        mask=np.zeros(len(positive_orders),dtype=bool)
        #log.info('mask shape={}'.format(mask.shape))
        mask[list(used_orders.values())]=True
        
        if dim == 2:
            if self.positive_orders[0] == 0:
                mask = np.concatenate((mask,mask[:0:-1]))
            else:
                mask = np.concatenate((mask,mask[::-1]))

        return mask
        
    def generate_radial_mask(self,mask_opt):
        '''
        Constructs the radial mask (mask of momentum transfair values). Masked values are False and will not be used 
        in the reciprocal coefficient projection defined in self.generate_coeff_projection.
        '''
        radial_points = self.radial_points
        data_mask = True
        if ~isinstance(self.data_min_q,bool):
            data_mask =  (radial_points>=self.data_min_q) & (radial_points<=self.data_max_q)
        
        if isinstance(mask_opt,(dict,DictNamespace)):
            mtype=mask_opt['type']
            if mtype == 'region':
                region=mask_opt['region']
                radial_points=self.radial_points
                if (region[0] == False) and (region[1] != False):
                    mask=radial_points<region[1]
                elif (region[0] != False) and (region[1] == False):                    
                    mask=radial_points>=region[0]
                    log.info('radial mask non zero count {} of {}'.format(np.sum(mask),np.prod(mask.shape)))
                elif (region[0] != False) and (region[1] != False):
                    mask=(radial_points>=region[0])  & (radial_points<region[1])
                else:
                    mask=True
        mask = mask & data_mask
        return mask


    def modify_projection_matrices(self,opt):
        dim = self.dimensions
        rescale_projection=opt.get('rescale_projection_to_1',False)
        use_averaged_intensity=opt.get('use_averaged_intensity',False)
        use_odd_orders_to_0=opt.get('odd_orders_to_0',False)
        use_q_window=opt.get('use_q_window',False)
        
        average_intensity=self.average_intensity.data.astype(np.complex)
        used_orders=self.used_orders
        odd_order_mask=np.array(tuple(used_orders))%2==1

        if dim == 2:
            proj_matrices=self.projection_matrices.copy()
            if use_odd_orders_to_0:
                proj_matrices[:,odd_order_mask]=0
        elif dim == 3:
            proj_matrices=[matrix.copy() for matrix in self.projection_matrices]
            if use_odd_orders_to_0:
                for odd_order in np.array(tuple(used_orders))[odd_order_mask]:
                    #log.info('odd order={}'.format(odd_order))
                    proj_matrices[used_orders[odd_order]][:]=0                                    
        #if use_averaged_intensity:
        #    if dim == 2:
        #        proj_matrices[:,used_orders[0]]=average_intensity
        #    elif dim == 3:
        #        raise NotImplementedError
    
        if rescale_projection:
            max_value=max( (np.max(np.abs(matrix)) for matrix in proj_matrices))
            if max_value==0:
                max_value=1e-12                
            for matrix in proj_matrices:
                matrix/=max_vector_value
        average_intensity[:]/=10
        proj_mmatrices = [m/np.sqrt(10) for m in proj_matrices]
        proj_mmatrices[0] /= np.sqrt(10)
        return proj_matrices

    def generate_coeff_projection(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        mask=self.mask_2d
        radial_mask = self.radial_mask
        projection_matrices[zero_id]= np.abs(projection_matrices[zero_id])
        if dim == 2:
            where=np.where
            if use_averaged_intensity:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    intensity_harmonic_coefficients[radial_mask,zero_id]=average_intensity[radial_mask]
                    return intensity_harmonic_coefficients
            else:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    return intensity_harmonic_coefficients
        elif dim == 3:
            dot=np.dot
            copy=np.array #array is faster than copy
            a_factor = 2*np.sqrt(np.pi)            
            if use_averaged_intensity and isinstance(zero_id,int):
                scaled_average_int = np.copy(average_intensity)*a_factor
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #log.info("projection_matrix_shape = {}").format()
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #apply mask
                        #log.info('I0 shape = {}'.format(projected_intensity_coefficients[zero_id].shape))
                    projected_intensity_coefficients[zero_id][radial_mask,...] = scaled_average_int[radial_mask,...]
                    #projected_intensity_coefficients[zero_id] = (a_factor*average_intensity
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #projected_intensity_coefficients[o_id][radial_mask,...] = intensity_harmonic_coefficients[o_id][radial_mask,...]
                        #apply mask
                    projected_intensity_coefficients[zero_id][radial_mask,...] = projection_matrices[zero_id][radial_mask,...]
                    return projected_intensity_coefficients                
        return mtip_projection
    
    def generate_coeff_projection_fixed(self,coeff_projection):
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        radial_mask = self.radial_mask
        def fixed_projection(intensity_harmonic_coefficients,unknowns):
            projected_intensity_coefficients = coeff_projection(intensity_harmonic_coefficients,unknowns)
            projected_intensity_coefficients[zero_id][radial_mask]/=np.sqrt(self.number_of_particles)
            return projected_intensity_coefficients
        return fixed_projection
    
    def generate_approximate_unknowns(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            def approximate_unknowns(intensity_harmonic_coefficients):
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(projection_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                #        log.info(scalar_prod_Im_vm)
                unknowns=scalar_prod_Im_vm/where(abs(scalar_prod_Im_vm)==0,1,abs(scalar_prod_Im_vm))
                unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                BAs=tuple(PD @ I for PD,I in zip(PDs,intensity_harmonic_coefficients))  #each element is B^\dagger A in a Procrustres Problem min|A-BR|
                unknowns=tuple(matmul(*svd(BA,full_matrices=False)[::2]) for BA in BAs)
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                #unknowns = [np.zeros(u.shape) for u in unknowns]
                #for u in unknowns:                    
                #    u[:,:len(u)]=np.eye(len(u))
                return unknowns             
        return approximate_unknowns
    #scaled version to donatellies n particle determination
    def generate_coeff_projection_n_particle_scaled(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        mask=self.mask_2d
        radial_mask = self.radial_mask
        if dim == 2:
            where=np.where
            if use_averaged_intensity:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    intensity_harmonic_coefficients[radial_mask,zero_id]=average_intensity[radial_mask]
                    return intensity_harmonic_coefficients
            else:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    return intensity_harmonic_coefficients
        elif dim == 3:
            dot=np.dot
            copy=np.array #array is faster than copy
            a_factor = 2*np.sqrt(np.pi)            
            if use_averaged_intensity and isinstance(zero_id,int):
                scaled_average_int = np.copy(average_intensity)*a_factor
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    scale = np.sqrt(self.n_particles)
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #log.info("projection_matrix_shape = {}").format()
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id]/scale @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #apply mask
                        #log.info('I0 shape = {}'.format(projected_intensity_coefficients[zero_id].shape))
                    projected_intensity_coefficients[zero_id][radial_mask,...] = scaled_average_int[radial_mask,...]/self.n_particles
                    #projected_intensity_coefficients[zero_id] = (a_factor*average_intensity
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    scale = np.sqrt(self.n_particles)
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id]/scale @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #projected_intensity_coefficients[o_id][radial_mask,...] = intensity_harmonic_coefficients[o_id][radial_mask,...]
                        #apply mask
                    projected_intensity_coefficients[zero_id]/=scale
                    return projected_intensity_coefficients                
        return mtip_projection
    def generate_approximate_unknowns_n_particle_scaled(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            def approximate_unknowns(intensity_harmonic_coefficients):
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(projection_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                #        log.info(scalar_prod_Im_vm)
                unknowns=scalar_prod_Im_vm/where(abs(scalar_prod_Im_vm)==0,1,abs(scalar_prod_Im_vm))
                unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                scale = np.sqrt(self.n_particles)
                BAs=list(PD/scale @ I for PD,I in zip(PDs,intensity_harmonic_coefficients))  #each element is B^\dagger A in a Procrustres Problem min|A-BR|
                BAs[0]/=scale
                unknowns = tuple(matmul(*svd(BA,full_matrices=False)[::2]) for BA in BAs)
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                return unknowns             
        return approximate_unknowns
    def generate_apply_SO_freedom_2D(self,opt):
        dim=self.dimensions
        if dim ==2 :
            ordes_dict=self.used_orders
            orders=np.array(tuple(orders_dict))
            order_ids=tuple(orders_dict.values())
            
            radial_high_pass_index=opt['SO_freedom'].get('radial_high_pass',np.nan)
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

    def change_n_particles(self,N):
        '''Setter for n_particles attribute'''
        self.n_particles = N

    def apply_particle_number_scaling(self,projection_matrices,zero_id,average_intensity):
        '''
        Let N be the nunmber of particles. this method scales the projection_matrices $V_l$ by $1/\sqrt(N)$ for $l\neq 0$ and $1/N$ for $l = 0$.
        The averaged intensity is scaled by $1/N$, since it is proportional to $V_0$
        '''
        N = settings.analysis.projections.reciprocal.number_of_particles
        projection_matrices = [matrix/np.sqrt(N) for matrix in projection_matrices]
        projection_matrices[zero_id] = projection_matrices[zero_id]/np.sqrt(N)
        average_intensity = average_intensity/N
        return projection_matrices,average_intensity
            
    def generate_number_of_particles_porjection(self):
        p_opt = settings.analysis.projections.reciprocal
        radial_mask=self.radial_mask
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
                self.number_of_particles = Ns[inflection_id]
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
                self.number_of_particles = Ns[inflection_id]
                #log.info('number of particles = {}'.format(self.number_of_particles))
                #I[radial_mask] += (1/np.sqrt(initial_n_particles)-1)*I00y00[radial_mask]
                #I[I<0]=0
                del(scaled_I)            
                return I
            
        return particle_number_projection
    
    def generate_number_of_particles_porjection_gpu(self):
        p_opt = settings.analysis.projections.reciprocal
        radial_mask=self.radial_mask
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

    def get_number_of_particles(self):
        return self.number_of_particles
    def generate_project_to_modified_intensity(self):
        zeros = np.zeros
        density_shape = self.grid[:].shape[:-1]
        sqrt = np.sqrt
        nabs = np.abs
        
        def project_to_modified_intensity(reciprocal_density,new_intensity):
            bigger_than_zero_I = new_intensity>0
            nonzero_d = reciprocal_density!=0
            mask = bigger_than_zero_I & nonzero_d

            #d_part = reciprocal_density[mask]
            #new_reciprocal_density = zeros(density_shape,dtype = complex)
            #new_reciprocal_density[mask] = d_part/nabs(d_part).real*sqrt(new_intensity[mask].real)
            new_intensity[~mask]=1
            reciprocal_density[~mask]=1
            new_reciprocal_density = reciprocal_density/np.abs(reciprocal_density).real *sqrt(new_intensity.real)
            return new_reciprocal_density
        return project_to_modified_intensity
class ReciprocalProjection_with_preinit:
    dimensions = False
    xray_wavelength = False
    enforce_cross_correlation_qq_sym = False
    enforce_cross_correlation_phi_sym = False    
    average_intensity = False
    integrated_intensity = False
    data_projection_matrices = False
    pi_in_q = None    
    max_order = False
    data_n_particles = False
    data_radial_points = False
    data_angular_points = False
    data_max_q = False
    data_min_q = False
    @classmethod
    def set_standard_class_variables(cls):
        opt = settings.analysis
        prj_opt = opt.projections.reciprocal.pre_init
        cls.dimensions = opt.dimensions
        cls.max_order = prj_opt.max_order
        
    @classmethod
    def pre_init_from_ccd(cls):
        ccd=db.load('ccd')
        pre_init_opt = settings.analysis.projections.reciprocal.pre_init
        proj_opt = pre_init_opt.cross_correlation

        cls.dimensions = ccd['dimensions']
        cls.data_n_particles = proj_opt.n_particles
        average_intensity = ccd.get('average_intensity',False)
        if cls.data_n_particles>1:
            average_intensity.data/=cls.data_n_particles
        cls.average_intensity = average_intensity
        cls.data_average_intensity = np.array(average_intensity.data)
        cls.xray_wavelength = ccd['xray_wavelength']
        cls.data_radial_points = ccd['radial_points']
        cls.data_max_q = np.max(cls.data_radial_points)
        cls.data_min_q = np.min(cls.data_radial_points)
        cls.data_angular_points = ccd['angular_points']
        cls.pi_in_q = ccd['pi_in_q']
        max_theoretical_order = max_order_from_n_angular_steps(cls.dimensions,len(cls.data_angular_points))
        max_order = settings.analysis.projections.reciprocal.pre_init.max_order
        try:
            assert max_theoretical_order >= max_order , 'max order needs to be smaller than {} for the given number of angular points {} but {} was specified'.format(max_theoretical_order,len(cls.angular_points),max_order)
        except AssertionError as e:
            log.warning(e)
            max_order = max_theoretical_order
        cls.max_order = max_order
        cc=ccd.pop('cross_correlation')
        #cc = i_tools.modify_cross_correlation(cc,cls.max_order,**proj_opt.modify)
        extract_odd_orders = settings.analysis.projections.reciprocal.pre_init.extract_odd_orders
        metadata = {**ccd,'orders':np.arange(cls.max_order+1),**proj_opt,'extract_odd_orders':extract_odd_orders,'bl_enforce_psd':pre_init_opt.bl_enforce_psd,'psd_q_min_id':pre_init_opt.psd_q_min_id,'mode':pre_init_opt.bl_extraction_method,'modify':proj_opt.modify}
        log.info('options = {}'.format(proj_opt.modify))
        cls.b_coeff,q_mask = i_tools.cross_correlation_to_deg2_invariant(cc,cls.dimensions,**metadata)
        cls.data_min_q = np.min(cls.data_radial_points[q_mask])
        del(cc)
        cls.data_projection_matrices=i_tools.deg2_invariant_to_projection_matrices(cls.dimensions,cls.b_coeff)

    @classmethod
    def resize_density(cls,density,n_radial_points):
        density_shape = density.shape
        n_data_radial_points = density_shape[0]
        if n_data_radial_points>=n_radial_points:
            density=density[:n_radial_points,...]
        else:
            new_shape = (n_radial_points,) + density_shape[1:]
            temp = np.zeros(new_shape,dtype=float)
            temp[:n_data_radial_points,...] = density
            density = temp
        return density
    @classmethod
    def pre_init_from_density(cls,density_dict,comm_module,opt):
        density = density_dict['density']
        cls.dimensions=len(density.shape)
        cls.xray_wavelength = density_dict['xray_wavelength']
        cls.pi_in_q = ccd['pi_in_q']       
        max_order = settings.analysis.projections.reciprocal.pre_init.max_order
        oversampling = settings.analysis.grid.oversampling
        anti_aliazing_degree = settings.grid.anti_aliazing_degree
        particle_size = settings.particle_size
        
        n_data_radial_points = density.shape[0]
        
        data_max_r = density_dict['max_r']
        r_step = density_dict['r_step']        
        n_radial_points = (oversampling*particle_size)//r_step+1
        density = cls.resize_density(density,n_radial_points)

    @classmethod
    def pre_init_from_pdb(cls):
        raise NotImplementedError("changes to pdb access necessary. It's not usableright now.")
        opt = settings.analysis
        prj_opt = opt.projections.reciprocal.pre_init
        pdb_opt = opt.projections.reciprocal.pre_init.from_pdb
        
        cls.pi_in_q = opt.fourier_transform.pi_in_q

        db = database.analysis
        
        extractor = db.load('pdb://'+pdb_opt.pdb_id)

        # create_grid
        n_radial_points = pdb_opt.n_radial_points
        particle_radius = extractor.max_r
        oversampling = opt.grid.oversampling
        max_r = oversampling*particle_radius
        max_q = polar_spherical_dft_reciprocity_relation_radial_cutoffs(max_r,n_radial_points,pi_in_q = cls.pi_in_q)
        cls.data_max_q = max_q
        if opt.grid.max_order < cls.max_order:
            log.warning('Maximal invariant extraction order(given as {}) must be larger than maximal order of internal grid (given as {}). I will continue by setting them equal.'.format(cls.max_order,opt.grid.max_order))
            cls.max_order = opt.grid.max_order
        ht_opt={'dimensions':cls.dimensions,'max_order':cls.max_order,**opt.grid}
        log.info('max order = {}'.format(cls.max_order))
        cht=HarmonicTransform('complex',ht_opt)
        #weight_dict = generate_weightDict_zernike_spherical(max_order,n_radial_points)        
        grid_opt={'dimensions':3,'type':'Zernike_dict','max_q':max_q,'n_radial_points':n_radial_points,**cht.grid_param,'pi_in_q':True}
        grid_pair = get_grid(grid_opt)
        cls.data_radial_points =  grid_pair.reciprocal[:,0,0,0]
        cls.data_angular_points = grid_pair.reciprocal[0,0,:,2]
        cls.data_max_q = np.max(cls.data_radial_points)

        log.info('radial data = {}'.format(cls.data_radial_points))

        
        density = extractor.extract_density(
            pdb_opt.density_type,
            grid_pair.real,
            gaussian_sigma=pdb_opt.gaussian_sigma,
            atom_neighbour_shells = pdb_opt.atom_neighbour_shells
        )

        log.info('densty shape = {} grid shape = {}'.format(density.shape,grid_pair.real[:].shape))
        if pdb_opt.save_vtk_density:
            db.save('pdb_density', [density], grid = grid_pair.real[:], grid_type='spherical',path_modifiers={'name':opt.name})
        ft_opt = {
            'dimensions':cls.dimensions,
            'type': opt.fourier_transform.type,
            'data_type': opt.fourier_transform.data_type,
            'n_radial_points': n_radial_points,
            'pos_orders': np.arange(cls.max_order+1),
            'pi_in_q':cls.pi_in_q,
            'use_GPU':opt.GPU.use,
            'allow_weight_calculation':opt.fourier_transform.allow_weight_calculation,        
            'allow_weight_saving': opt.fourier_transform.allow_weight_saving
        }
        ft,ift = False,False
        raise NotImplementedError("changes to pdb access necessary. It's not usableright now.")
        log.info('start b_coeff calculation')
        log.info('density shape = {}'.format(density.shape))
        log.info('grid shape = {}'.format(grid_pair.real.shape))
        
        cls.b_coeff = i_tools.density_to_deg2_invariants_3d(density,cht,ft)
        cls.average_intensity = SampledFunction(NestedArray(cls.data_radial_points,1),np.sqrt(np.diag(cls.b_coeff[0]*4*np.pi)),coord_sys='cartesian')
        log.info('start projection calculation')
        log.info('bcoeff shapes = {}'.format(cls.b_coeff.shape))
        cls.data_projection_matrices=i_tools.deg2_invariant_to_projection_matrices(cls.dimensions,cls.b_coeff)
        log.info('Finished')
        #raise AssertionError
        
    @classmethod
    def pre_init_from_file(cls):
        opt = settings.analysis
        data = db.load('reciprocal_proj_data',path_modifiers={'name':opt.name,'max_order':cls.max_order})        
        cls.dimensions = data['dimensions']
        cls.xray_wavelength = data['xray_wavelength']
        cls.average_intensity = data['average_intensity']
        #log.info('aint type ={}'.format(type(cls.average_intensity)))
        cls.data_radial_points = data['data_radial_points'][:]
        cls.data_angular_points = data['data_angular_points'][:]
        cls.data_max_q = np.max(cls.data_radial_points)
        cls.data_min_q = data.get('data_min_q',False)
        #cls.pi_in_q = data.get('pi_in_q',False)
        cls.pi_in_q = data['pi_in_q']
        cls.max_order=data['max_order']
        mode = settings.analysis.projections.reciprocal.pre_init.from_file.mode
        if mode== 'proj_matrices':
            cls.data_projection_matrices = data['data_projection_matrices']
        elif mode == 'b_coeff':
            cls.b_coeff = data['b_coeff']
            cls.data_projection_matrices=i_tools.deg2_invariant_to_projection_matrices(cls.dimensions,cls.b_coeff)
        
        
    @classmethod
    def pre_init(cls):
        print('call to preinit.')
        global db
        db = database.analysis
        cls.set_standard_class_variables()
        mode = settings.analysis.projections.reciprocal.pre_init.get_mode
        try:
            cls.pre_init_routines[mode].__func__(cls)
            cls.integrated_intensity = np.trapz(cls.average_intensity.data * cls.data_radial_points**2 , x = cls.data_radial_points,axis = 0)*4*np.pi
        except KeyError as e:
            log.error(e)
            log.error('Error during pre initialization of Reciprocal Projection in mode "{}". Known modes are "{}"'.format(mode,cls.pre_init_routines.keys()))
            raise e
        save_projection = settings.analysis.projections.reciprocal.pre_init.save        
        if mode != 'from_file' and save_projection:
            name = settings.analysis.name            
            db.save('reciprocal_proj_data', ReciprocalProjection, path_modifiers={'name':name,'max_order':cls.max_order})


    pre_init_routines = {
        'from_ccd': pre_init_from_ccd,
        'from_file': pre_init_from_file,
        'from_pdb': pre_init_from_pdb
    }    

    def __init__(self,grid,opt=False):
        if not isinstance(opt,(dict,DictNamespace)):
            opt = self.extract_used_options()
        self.opt=opt
        self.grid=grid        
        self.radial_points=grid.__getitem__((slice(None),)+(0,)*self.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
        self.max_q = np.max(self.radial_points)
        self.input_is_halfcomplex=opt.input_is_halfcomplex
        self.positive_orders=opt.pos_orders
        self.used_order_ids=opt.used_order_ids
        self.used_orders={order:id for (order,id) in zip(self.positive_orders,self.used_order_ids)}

        use_SO_freedom=opt.SO_freedom.use
        
        self.projection_matrices=self._regrid_data()
        self.projection_matrices=self.modify_projection_matrices(opt)
        self.assert_projection_matrices_have_right_shape()
        
        self.radial_mask=self.generate_radial_mask(opt.q_mask)
        self.order_mask=self.generate_order_mask()
        self.mask_2d=self.radial_mask[:,None]*self.order_mask[None,:]
        self.mtip_projection=self.generate_coeff_projection()
        self.approximate_unknowns=self.generate_approximate_unknowns()

        if use_SO_freedom:
            self.apply_SO_freedom=generate_apply_SO_freedom(opt)
            def calc_unknowns(intensity_harmonic_coefficients):
                return apply_SO_freedom(approximate_unknowns(intensity_harmonic_coefficients))
            self.calc_unknowns=calc_unknowns
        else:
            self.calc_unknowns=self.approximate_unknowns            
        log.info('init end projection matrices shape = {}'.format(self.projection_matrices[-1].shape))
        self.n_particles = 1
        
    def assert_projection_matrices_have_right_shape(self):
        try:
            n_radial_points_grid = len(self.radial_points)
            n_radial_points_projection_matrix = self.projection_matrices[0].shape[0]
            assert n_radial_points_grid == n_radial_points_projection_matrix,'Mismatch between th number of radial sampling points in reconstruction grid {}  and projection data {}. Abort reconstruction !'.format(n_radial_points_grid,n_radial_points_projection_matrix)
        except AssertionError as e:
            log.error(e)
            raise e
            
    def extract_used_options(self):
        fourier_opt = settings.analysis.fourier_transform
        rp_opt = settings.analysis.projections.reciprocal
        opt = DictNamespace(
            pos_orders = fourier_opt.pos_orders,
            **rp_opt
        )
        return opt
    def _regrid_data(self):
        dim = self.dimensions
        order_ids=list(self.used_orders.values())
        have_same_shapes = self.radial_points.shape == self.data_radial_points.shape
        projection_matrices = self.data_projection_matrices
        needs_regridding = True
        if have_same_shapes:
            if (self.data_radial_points == self.radial_points).all():
                needs_regridding = False
        log.info('needs regridding = {}'.format(needs_regridding))   
        if needs_regridding:
            r_pt=NestedArray(self.radial_points[:,None],1)
            data_r_pt=NestedArray(self.data_radial_points[:,None],1)
            #log.info('n new points={} n old points ={}'.format(len(r_pt[:]),len(data_r_pt[:])))
            if dim == 2:                
                self.average_intensity.regrid(r_pt)
                projection_matrices = ReGrider.regrid(self.data_projection_matrices[...,order_ids],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':self.opt.regrid.interpolation})
            elif dim == 3:
                self.average_intensity.regrid(r_pt)
                data_projection_matrices=self.data_projection_matrices
                projection_matrices = tuple(ReGrider.regrid(data_projection_matrices[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1,'fill_value': 0.0,'interpolation':self.opt.regrid.interpolation}) for o_id in order_ids)
        log.info('regrided projection matrices shape = {}'.format(projection_matrices[-1].shape))
        return projection_matrices

    
    def generate_order_mask(self):
        dim=self.dimensions
        positive_orders=self.positive_orders
        used_orders=self.used_orders
        
        mask=np.zeros(len(positive_orders),dtype=bool)
        #log.info('mask shape={}'.format(mask.shape))
        mask[list(used_orders.values())]=True
        
        if dim == 2:
            if self.positive_orders[0] == 0:
                mask = np.concatenate((mask,mask[:0:-1]))
            else:
                mask = np.concatenate((mask,mask[::-1]))

        return mask
        
    def generate_radial_mask(self,mask_opt):
        '''
        Constructs the radial mask (mask of momentum transfair values). Masked values are False and will not be used 
        in the reciprocal coefficient projection defined in self.generate_coeff_projection.
        '''
        radial_points = self.radial_points
        data_mask = True
        if ~isinstance(self.data_min_q,bool):
            data_mask =  radial_points >= self.data_min_q
        
        if isinstance(mask_opt,(dict,DictNamespace)):
            mtype=mask_opt['type']
            if mtype == 'region':
                region=mask_opt['region']
                radial_points=self.radial_points
                if (region[0] == False) and (region[1] != False):
                    mask=radial_points<region[1]
                elif (region[0] != False) and (region[1] == False):                    
                    mask=radial_points>=region[0]
                    log.info('radial mask non zero count {} of {}'.format(np.sum(mask),np.prod(mask.shape)))
                elif (region[0] != False) and (region[1] != False):
                    mask=(radial_points>=region[0]) & (radial_points<region[1])
                else:
                    mask=True
        mask = mask & data_mask
        return mask


    def modify_projection_matrices(self,opt):
        dim = self.dimensions
        rescale_projection=opt.get('rescale_projection_to_1',False)
        use_averaged_intensity=opt.get('use_averaged_intensity',False)
        use_odd_orders_to_0=opt.get('odd_orders_to_0',False)
        use_q_window=opt.get('use_q_window',False)
        
        average_intensity=self.average_intensity.data.astype(np.complex)
        used_orders=self.used_orders
        odd_order_mask=np.array(tuple(used_orders))%2==1

        if dim == 2:
            proj_matrices=self.projection_matrices.copy()
            if use_odd_orders_to_0:
                proj_matrices[:,odd_order_mask]=0
        elif dim == 3:
            proj_matrices=[matrix.copy() for matrix in self.projection_matrices]
            if use_odd_orders_to_0:
                for odd_order in np.array(tuple(used_orders))[odd_order_mask]:
                    #log.info('odd order={}'.format(odd_order))
                    proj_matrices[used_orders[odd_order]][:]=0                                    
        #if use_averaged_intensity:
        #    if dim == 2:
        #        proj_matrices[:,used_orders[0]]=average_intensity
        #    elif dim == 3:
        #        raise NotImplementedError
    
        if rescale_projection:
            max_value=max( (np.max(np.abs(matrix)) for matrix in proj_matrices))
            if max_value==0:
                max_value=1e-12                
            for matrix in proj_matrices:
                matrix/=max_vector_value                
        return proj_matrices

    def generate_coeff_projection(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        mask=self.mask_2d
        radial_mask = self.radial_mask
        if dim == 2:
            where=np.where
            if use_averaged_intensity:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    intensity_harmonic_coefficients[radial_mask,zero_id]=average_intensity[radial_mask]
                    return intensity_harmonic_coefficients
            else:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    return intensity_harmonic_coefficients
        elif dim == 3:
            dot=np.dot
            copy=np.array #array is faster than copy
            a_factor = 2*np.sqrt(np.pi)            
            if use_averaged_intensity and isinstance(zero_id,int):
                scaled_average_int = np.copy(average_intensity)*a_factor
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #log.info("projection_matrix_shape = {}").format()
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #apply mask
                        #log.info('I0 shape = {}'.format(projected_intensity_coefficients[zero_id].shape))
                    projected_intensity_coefficients[zero_id][radial_mask,...] = scaled_average_int[radial_mask,...]
                    #projected_intensity_coefficients[zero_id] = (a_factor*average_intensity
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id] @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #projected_intensity_coefficients[o_id][radial_mask,...] = intensity_harmonic_coefficients[o_id][radial_mask,...]
                        #apply mask
                    return projected_intensity_coefficients                
        return mtip_projection

    def generate_approximate_unknowns(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            def approximate_unknowns(intensity_harmonic_coefficients):
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(projection_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                #        log.info(scalar_prod_Im_vm)
                unknowns=scalar_prod_Im_vm/where(abs(scalar_prod_Im_vm)==0,1,abs(scalar_prod_Im_vm))
                unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                BAs=tuple(PD @ I for PD,I in zip(PDs,intensity_harmonic_coefficients))  #each element is B^\dagger A in a Procrustres Problem min|A-BR|
                unknowns=tuple(matmul(*svd(BA,full_matrices=False)[::2]) for BA in BAs)
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                return unknowns             
        return approximate_unknowns
    #scaled version to donatellies n particle determination
    def generate_coeff_projection_n_particle_scaled(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices
        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        mask=self.mask_2d
        radial_mask = self.radial_mask
        if dim == 2:
            where=np.where
            if use_averaged_intensity:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    intensity_harmonic_coefficients[radial_mask,zero_id]=average_intensity[radial_mask]
                    return intensity_harmonic_coefficients
            else:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply q_mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    return intensity_harmonic_coefficients
        elif dim == 3:
            dot=np.dot
            copy=np.array #array is faster than copy
            a_factor = 2*np.sqrt(np.pi)            
            if use_averaged_intensity and isinstance(zero_id,int):
                scaled_average_int = np.copy(average_intensity)*a_factor
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    scale = np.sqrt(self.n_particles)
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #log.info("projection_matrix_shape = {}").format()
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id]/scale @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #apply mask
                        #log.info('I0 shape = {}'.format(projected_intensity_coefficients[zero_id].shape))
                    projected_intensity_coefficients[zero_id][radial_mask,...] = scaled_average_int[radial_mask,...]/self.n_particles
                    #projected_intensity_coefficients[zero_id] = (a_factor*average_intensity
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    scale = np.sqrt(self.n_particles)
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        #projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]
                        tmp_coeff = projection_matrices[o_id]/scale @ unknowns[o_id]
                        projected_intensity_coefficients[o_id][radial_mask,...] = tmp_coeff[radial_mask,...]
                        #projected_intensity_coefficients[o_id][radial_mask,...] = intensity_harmonic_coefficients[o_id][radial_mask,...]
                        #apply mask
                    projected_intensity_coefficients[zero_id]/=scale
                    return projected_intensity_coefficients                
        return mtip_projection

    def generate_approximate_unknowns_n_particle_scaled(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            def approximate_unknowns(intensity_harmonic_coefficients):
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(projection_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                #        log.info(scalar_prod_Im_vm)
                unknowns=scalar_prod_Im_vm/where(abs(scalar_prod_Im_vm)==0,1,abs(scalar_prod_Im_vm))
                unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                scale = np.sqrt(self.n_particles)
                BAs=list(PD/scale @ I for PD,I in zip(PDs,intensity_harmonic_coefficients))  #each element is B^\dagger A in a Procrustres Problem min|A-BR|
                BAs[0]/=scale
                unknowns=tuple(matmul(*svd(BA,full_matrices=False)[::2]) for BA in BAs)
                #log.info('unknowns shape ={}'.format(unknowns[-1].shape))
                return unknowns             
        return approximate_unknowns
    def generate_apply_SO_freedom_2D(self,opt):
        dim=self.dimensions
        if dim ==2 :
            ordes_dict=self.used_orders
            orders=np.array(tuple(orders_dict))
            order_ids=tuple(orders_dict.values())
            
            radial_high_pass_index=opt['SO_freedom'].get('radial_high_pass',np.nan)
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

    def change_n_particles(self,N):
        '''Setter for n_particles attribute'''
        self.n_particles = N                
class ReciprocalProjection_old2:
    dimensions = False
    xray_wavelength = False
    average_intensity = False
    data_radial_points = False
    data_angular_points = False
    data_projection_matrices = False
    pi_in_q = None
    comm_module = False
    max_order = False
    b_abs_tolerance = False
    
    @classmethod
    def pre_init_from_ccd(cls,ccd,comm_module,opt):
        cls.dimensions=opt['dimensions']
        cls.b_abs_tolerance=float(opt.get('b_abs_tolerance',1e-15))
        cls.average_intensity=ccd.get('average_intensity',False)
        cls.xray_wavelength = ccd['xray_wavelength']
        cls.data_radial_points = ccd['radial_points']
        cls.data_angular_points = ccd['angular_points']
        cls.pi_in_q = ccd['pi_in_q']
        cls.comm_module=comm_module
        max_order=opt['max_order']
        theoretical_max_order=(len(cls.data_angular_points)-1)//2
        if max_order>theoretical_max_order:
            log.warning('Maximal order cannot exeed (angular_points -1)//2={} but {} was given. Proceed with {}'.format(theoretical_max_order,max_order,theoretical_max_order))
            max_order=theoretical_max_order
        cls.max_order=max_order
        log.info('stg max_order = {}'.format(max_order))
               
        cc=ccd['cross_correlation']
        bl_extraction_method = opt.get('bl_extraction_method','svd')
        b_coeff=cls.extract_b_coeff(cc,bl_extraction_method)
        log.info('b_coeff shape = {}'.format(b_coeff.shape))
        cls.b_coeff = b_coeff.copy()
        del(cc)
        cls.data_projection_matrices=cls.calc_projection_matrices(b_coeff)

    @classmethod
    def pre_init_from_file(cls,data,comm_module, mode = 'proj_matrices'):
        cls.dimensions = data['dimensions']
        cls.xray_wavelength = data['xray_wavelength']
        cls.average_intensity = data['average_intensity']
        log.info('aint type ={}'.format(type(cls.average_intensity)))
        cls.data_radial_points = data['data_radial_points'][:]
        cls.data_angular_points = data['data_angular_points'][:]
        cls.pi_in_q = data.get('pi_in_q',False)
        cls.max_order=data['max_order']
        cls.comm_module=comm_module
        if mode== 'proj_matrices':
            cls.data_projection_matrices = data['data_projection_matrices']
        elif mode == 'b_coeff':
            log.info('test mode started from b_coeff')
            cls.b_coeff = data['b_coeff']

            #cls.b_coeff[1::2] = 0
            #for b in cls.b_coeff[::2]:            
                #b[b< 1e-25*np.abs(b).max()]=0
                
            cls.data_projection_matrices = cls.calc_projection_matrices(cls.b_coeff)
        
    @classmethod
    def extract_b_coeff_3d(cls,cc,mode):
        #l_shape= l,q,qq,phi
        qs=cls.data_radial_points
        q_ids=np.arange(len(qs),dtype=int)
        phis=cls.data_angular_points
        cc = cc[...,phis < np.pi] # due to fridel symmetry
        phis = phis[phis < np.pi] # fridel symmetry
        n_phis = len(phis)
        # log.info('\n \n len phis = {} \n \n'.format(len(phis)))
        # log.info('original cc shape = {}'.format(cc.shape))
        # log.info('new cc shape = {}'.format(cc.shape))
        orders=np.arange(cls.max_order+1)
        if cls.pi_in_q:
            thetas=pLib.ewald_sphere_theta(cls.xray_wavelength,qs/(2*np.pi))
        else:
            thetas=pLib.ewald_sphere_theta(cls.xray_wavelength,qs)
        n_orders=len(orders)
        b_coeff = np.zeros((len(qs),len(qs),len(orders)))
        #b_coeff[...,0::2]=cls.comm_module.request_mp_evaluation(cls.b_coeff_worker_3d,argArrays=[q_ids,q_ids], const_args=[cc,phis,thetas,n_orders,mode], callWithMultipleArguments=True)[...,:(len(orders)+1)//2]
        b_coeff=cls.comm_module.request_mp_evaluation(cls.b_coeff_worker_3d,argArrays=[q_ids,q_ids], const_args=[cc,phis,thetas,n_orders,mode], callWithMultipleArguments=True)
        #worker = bl_3d_pseudo_inverse_worker
        #b_coeff=cls.comm_module.request_mp_evaluation(worker,argArrays=[q_ids,q_ids], const_args=[cc,phis,thetas,orders], callWithMultipleArguments=True)
        #b_coeff=cls.comm_module.request_mp_evaluation(cls.b_coeff_worker_3d,argArrays=[q_ids[1:],q_ids[1:]], const_args=[cc,phis,orders[::2],thetas,matrix_size], callWithMultipleArguments=True)#[...,:(len(orders)+1)//2]
        #because for q=0 I(q,theta,phi) is a constant and thus only the coefficient I^0_0 is non-zero and F_0=1/(4*\pi).
        b_coeff[0,0,0]=4*np.pi*cc[0,0,0]
        b_coeff =np.moveaxis(b_coeff,2,0)
        #%b_mask = (np.abs(b_coeff) < np.max(np.abs(b_coeff),axis = 0)[None,...]*cls.b_abs_tolerance)
        #b_coeff[b_mask]=cls.b_abs_tolerance
        return b_coeff

    
    @staticmethod
    def b_coeff_sub_worker_3d_LU(leg_args,cc,n_orders):
        #log.info('leg_args shae = {}'.format(leg_args.shape))
        Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,cc.shape[-1]*2,2)[:,None,None],leg_args),0,-1) #Q,phi,l
        #log.info('legendre det = {}'.format(np.linalg.det(tuple(Legendre_matrices))))
        #Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(orders[:,None,None],leg_args),0,-1) #Q,phi,l
        log.info('leg matrix shape = {}'.format(Legendre_matrices.shape))
        log.info('cc shape = {}'.format(cc.shape))
        log.info('start solving.')
        b_coeffs = np.linalg.solve(Legendre_matrices,cc)
        #l_inv=np.linalg.pinv(tuple(Legendre_matrices))
        #l_inv=np.linalg.inv(tuple(Legendre_matrices))
        log.info(' \n\n b_coeff shape = {} \n\n'.format(b_coeffs.shape))
        #b_coeffs=np.sum(l_inv*cc[:,None,:],axis = -1)
        log.info('finished part of bcoeff calc.')
        return b_coeffs

    @staticmethod        
    def b_coeff_sub_worker_3d_svd(leg_args,cc,n_orders):
        # log.info('leg_args shae = {}'.format(leg_args.shape))
        Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,n_orders,2)[:,None,None],leg_args),0,-1) #Q,phi,l
        #log.info('legendre det = {}'.format(np.linalg.det(tuple(Legendre_matrices))))
        #Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(orders[:,None,None],leg_args),0,-1) #Q,phi,l
        # log.info('leg matrix shape = {}'.format(Legendre_matrices.shape))
        # log.info('cc shape = {}'.format(cc.shape))
        # log.info('start solving.')
        l_inv = np.linalg.pinv(Legendre_matrices)       
        b_coeffs=np.sum(l_inv*cc[:,None,:],axis = -1)
        # log.info('finished part of bcoeff calc.')
        return b_coeffs
    
    @staticmethod        
    def b_coeff_sub_worker_3d_QR(leg_args,cc,n_orders):
        #log.info('leg_args shae = {}'.format(leg_args.shape))
        Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,n_orders,2)[:,None,None],leg_args),0,-1) #Q,phi,l
        #log.info('legendre det = {}'.format(np.linalg.det(tuple(Legendre_matrices))))
        #Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(orders[:,None,None],leg_args),0,-1) #Q,phi,l
        
        # log.info('leg matrix shape = {}'.format(Legendre_matrices.shape))
        # log.info('cc shape = {}'.format(cc.shape))
        # log.info('start solving.')
        qrs = tuple(qr(matrix,mode = 'economic') for matrix in Legendre_matrices)        
        Q_inv = np.concatenate(tuple(qr[0].T[None,...] for qr in qrs),axis = 0)
        R = np.concatenate(tuple(qr[1][None,...] for qr in qrs),axis = 0)
        # log.info('Q_T shape = {} R shape = {}'.format(Q_inv.shape,R.shape))
        #l_inv=np.linalg.pinv(tuple(Legendre_matrices))
        #l_inv=np.linalg.inv(tuple(Legendre_matrices))
        b_coeffs =  np.linalg.solve(R,np.sum(Q_inv*cc[:,None,:],axis = -1))
        # log.info(' \n\n b_coeff shape = {} \n\n'.format(b_coeffs.shape))
        #b_coeffs=np.sum(l_inv*cc[:,None,:],axis = -1)
        #log.info('finished part of bcoeff calc.')
        return b_coeffs
    
    @staticmethod
    def b_coeff_worker_3d_old(q_ids,qq_ids,cc,phis,thetas,n_orders,mode,**kwargs):
        #gen leg_alg and cc_part
        matrix_size = n_orders*len(phis)
        q_ids = q_ids.astype(int)
        qq_ids = qq_ids.astype(int)
        legendre_args=np.cos(thetas)[q_ids,None]*np.cos(thetas)[qq_ids,None]+np.sin(thetas)[q_ids,None]*np.sin(thetas)[qq_ids,None]*np.cos(phis)[None,:] #Q,phi
        cc=cc[q_ids,qq_ids] # Q,phi
        # log.info('q_ids = {} qq_ids = {} '.format(q_ids.dtype,qq_ids.dtype))

        #gen chunks
        free_mem=kwargs['available_mem']
#        chunck_size= int(0.4*free_mem/(cc.itemsize * matrix_size*len(phis)))
        chunck_size= int(0.4*free_mem/(cc.itemsize * matrix_size))
        log.info('chunck_size = {}'.format(0.4*free_mem/(cc.itemsize * matrix_size*len(phis)) ))
        n_chunks=max(1,len(q_ids)//chunck_size)
        
        leg_split=np.array_split(legendre_args,n_chunks)
       # log.info('cc_shape={}'.format(cc[q_ids,qq_ids].shape))
        cc_split = np.array_split(cc,n_chunks)        
        # log.info('n parts = {}'.format(len(cc_split)))

        # log.info('bl extraction method = {}'.format(mode))
        # call subworker
        if mode == 'LU':
            sub_worker=ReciprocalProjection.b_coeff_sub_worker_3d_LU
        elif mode == 'svd':
            sub_worker=ReciprocalProjection.b_coeff_sub_worker_3d_svd
        elif mode == 'QR':
            sub_worker=ReciprocalProjection.b_coeff_sub_worker_3d_QR
        b_coeff_tuple=tuple(sub_worker(l_part,cc_part,n_orders) for l_part,cc_part in zip(leg_split,cc_split) )
        b_coeffs=np.concatenate(b_coeff_tuple)
        log.info('part finished')
        #b_coeffs[:,1::2]=0
        return b_coeffs

    @staticmethod
    def b_coeff_worker_3d(q_ids,qq_ids,cc,phis,thetas,n_orders,mode,**kwargs):
        q_ids = q_ids.astype(int)
        qq_ids = qq_ids.astype(int)
        leg_args=np.cos(thetas)[q_ids,None]*np.cos(thetas)[qq_ids,None]+np.sin(thetas)[q_ids,None]*np.sin(thetas)[qq_ids,None]*np.cos(phis)[None,:] #Q,phi
        cc=cc[q_ids,qq_ids] # Q,phi

        if mode == 'LU':
            Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,cc.shape[-1]*2,2)[:,None,None],leg_args),0,-1) #Q,phi,l
            b_coeffs = np.linalg.solve(Legendre_matrices,cc)
        elif mode == 'svd':
            #Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,n_orders,2)[:,None,None],leg_args),0,-1) #Q,phi,l
            Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,n_orders)[:,None,None],leg_args),0,-1) #Q,phi,l
            l_inv = np.linalg.pinv(Legendre_matrices)       
            b_coeffs=np.sum(l_inv*cc[:,None,:],axis = -1)
        elif mode == 'QR':
            Legendre_matrices=np.moveaxis(1/(4*np.pi)*mLib.eval_legendre(np.arange(0,n_orders,2)[:,None,None],leg_args),0,-1) #Q,phi,l
            l_inv = np.linalg.pinv(Legendre_matrices)       
            b_coeffs=np.sum(l_inv*cc[:,None,:],axis = -1)

        conds=np.linalg.cond(Legendre_matrices,p=-2)
        log.info('part finished')
        #b_coeffs[:,1::2]=0
        return b_coeffs
    @staticmethod
    def extract_b_coeff_2d(CC):
        '''
        Calculates B_m values from Cross-Correlation data CC using
        B_m = C_m where C_m are the harmonic coefficients of CC.  
               
        :type kind: ndarray (shape = (n_q,n_q,n_delta))
        :return: B_m coeff.
        :rtype: ndarray ((n_delta,n_q,n_q)))
        '''
        n_angular_points=CC.shape[-1]
        
        #log.info('n_angular_points={}'.format(n_angular_points))
        harm_trf = HarmonicTransform('real',{'dimensions':2,'n_phi':n_phi_points})
        ht_forward,ht_inverse = harm_trf.forward,harm_trf.inverse 
        #log.info('cc shape ={}'.format(cc.shape))
        #log.info('cc={}'.format(cc[1,:100,0]))
        b_coeff=np.moveaxis(ht_forward(CC),2,0)
        #log.info('b_coeff shape ={}'.format(b_coeff))
        return b_coeffs
    
    @classmethod
    def extract_b_coeff(cls,cc,mode):
        dim=cls.dimensions
        if dim == 2:
            b_coeff = cls.extract_b_coeff_2d(cc)
        elif dim == 3:
            b_coeff=cls.extract_b_coeff_3d(cc,mode)
        return b_coeff

    @staticmethod
    def _eig(b_matrix,order):
        b_mean = np.abs(b_matrix).mean()
        #log.info('bmean = {}'.format(b_mean))
        is_zero = np.isclose(b_matrix,0).all() 
        if not is_zero:
            #b_matrix /= b_mean
            eig_vals,eig_vect=np.linalg.eig(b_matrix)
            #eig_vals *= b_mean
        else:
            eig_vect = np.zeros(b_matrix.shape)
            eig_vals = np.zeros(b_matrix.shape[0])
            
        real_and_positive_mask=((eig_vals.imag == 0) & (eig_vals >=0))
        try:            
            assert real_and_positive_mask.all(), 'Eigenvalues of B_l ar not real or positive!'.format(eig_vals)
        except AssertionError as e:            
            log.warning('Negative eigenvalues detected for order {}. Setting them to 0'.format(order))
            eig_vals[eig_vals.real<0]=0
            #eig_vect=eig_vect[:,real_and_positive_mask]
        sorted_ids=np.argsort(eig_vals)[::-1]
        return eig_vals,eig_vect,sorted_ids
    
    @staticmethod
    def _projection_matrix_worker(dim,b_matrix,order):
        eig_vals,eig_vect,sorted_ids = ReciprocalProjection._eig(b_matrix,order)
        if dim == 2:
            V=eig_vect[:,sorted_ids[0]]
            G=eig_vals[sorted_ids[0]]
#            log.info('lambda={}'.format(G))
            proj_matrix=V*np.sqrt(G)
        elif dim == 3:
            #log.info('nradial_points={}'.format(len(eig_vect)))
            N=min(len(eig_vect),2*order+1)
            V=eig_vect[:,sorted_ids[:N]] # n_radial_points x N array
            G=np.diag(np.sqrt(eig_vals[sorted_ids[:N]]))
            proj_matrix=V @ G
        return proj_matrix
        
    @classmethod
    def calc_projection_matrices(cls,b_coeff):        
        dim = cls.dimensions
        log.info('max possible projection order={}'.format(len(b_coeff)))        
        
        get_projection_matrix=cls._projection_matrix_worker
        proj_matrices=tuple(get_projection_matrix(dim,bmatrix,order) for order,bmatrix in enumerate(b_coeff.copy()))
        if dim == 2:
            proj_matrices=np.vstack(proj_matrices)
            log.info('projection_matrices shape = {}'.format(proj_matrices.shape))
#        elif dim==3:
#            log.info('0 matrix shape ={}'.format(proj_matrices[0].shape))
#            proj_matrices= np.squeeze(proj_matrices[0]) + proj_matrices[1:]
        return proj_matrices
    
    
    @classmethod
    def pre_init_check(cls):
        class_variables_initialized= not (isinstance(cls.dimensions,bool) or isinstance(cls.xray_wavelength,bool) or isinstance(cls.average_intensity,bool) or isinstance(cls.data_radial_points,bool) or isinstance(cls.data_projection_matrices,bool) or isinstance(cls.max_order,bool) or isinstance(cls.comm_module,bool))
        try:
            assert class_variables_initialized, 'ReciprocalProjection is not yet pre_initialized, call pre_init_from_ccd or pre_init_from_file first.'
        except AssertionError as e:
            log.error(e)
            raise

    def __init__(self,opt,grid):
        self.pre_init_check()        
        self.opt=opt
        self.grid=grid
        self.radial_points=self.grid.__getitem__((slice(None),)+(0,)*self.dimensions)


        self.input_is_halfcomplex=opt['input_is_halfcomplex']
        self.positive_orders=opt['pos_orders']        
        self.used_order_ids=opt['used_order_ids']
        self.used_orders={order:id for (order,id) in zip(self.positive_orders,self.used_order_ids)}

        use_SO_freedom=opt['SO_freedom']['use']
        
        self.projection_matrices=self._regrid_data()
        self.projection_matrices=self.modify_projection_matrices(opt)
        self.radial_mask=self.generate_radial_mask(opt.get('mask',True))
        self.order_mask=self.generate_order_mask()
        self.mask=self.radial_mask[:,None]*self.order_mask[None,:]
        self.mtip_projection=self.generate_coeff_projection()
        self.approximate_unknowns=self.generate_approximate_unknowns()

        if use_SO_freedom:
            self.apply_SO_freedom=generate_apply_SO_freedom(opt)
            def calc_unknowns(intensity_harmonic_coefficients):
                return apply_SO_freedom(approximate_unknowns(intensity_harmonic_coefficients))
            self.calc_unknowns=calc_unknowns
        else:
            self.calc_unknowns=self.approximate_unknowns


    def _regrid_data(self):
        dim = self.dimensions
        order_ids=list(self.used_orders.values())
        have_same_shapes = self.radial_points.shape == self.data_radial_points.shape
        projection_matrices = self.data_projection_matrices
        needs_regridding = True
        if have_same_shapes:
            if (self.data_radial_points == self.radial_points).all():
                needs_regridding = False
        log.info('needs regridding = {}'.format(needs_regridding))   
        if needs_regridding:
            r_pt=NestedArray(self.radial_points[:,None],1)
            data_r_pt=NestedArray(self.data_radial_points[:,None],1)
            #log.info('n new points={} n old points ={}'.format(len(r_pt[:]),len(data_r_pt[:])))
            if dim == 2:                
                self.average_intensity.regrid(r_pt)
                projection_matrices = ReGrider.regrid(self.data_projection_matrices[...,order_ids],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1})
            elif dim == 3:
                self.average_intensity.regrid(r_pt)
                data_projection_matrices=self.data_projection_matrices
                projection_matrices = tuple(ReGrider.regrid(data_projection_matrices[o_id],data_r_pt,'cartesian',r_pt,'cartesian',options={'apply_over_axis':1}) for o_id in order_ids)
        log.info('a int type = {}'.format(type(self.average_intensity)))
        log.info('a int shape = {}'.format(self.average_intensity.data.shape))
        return projection_matrices

    
    def generate_order_mask(self):
        dim=self.dimensions
        positive_orders=self.positive_orders
        used_orders=self.used_orders
        
        mask=np.zeros(len(positive_orders),dtype=bool)
        #log.info('mask shape={}'.format(mask.shape))
        mask[list(used_orders.values())]=True
        
        if dim == 2:
            if self.positive_orders[0] == 0:
                mask = np.concatenate((mask,mask[:0:-1]))
            else:
                mask = np.concatenate((mask,mask[::-1]))

        return mask
        
    def generate_radial_mask(self,mask_opt):
        grid=self.grid
        if isinstance(mask_opt,dict):
            mtype=mask_opt['type']
            if mtype == 'region':
                region=mask_opt('region')
                radial_points=self.radial_points
                if (region[0] == False) and (region[1] != False):
                    mask=np.where(grid[...,0]<region[1],True,False)
                elif (region[0] != False) and (region[1] == False):
                    mask=np.where(grid[...,0]>region[0],True,False)
                elif (region[0] != False) and (region[1] != False):
                    mask=np.where((grid[...,0]>region[0]) & (grid[...,0]<region[1]),True,False)
                else:
                    mask=True
        else:
            mask=np.full(grid.shape[0],1,dtype=bool)
        return mask


    def modify_projection_matrices(self,opt):
        dim = self.dimensions
        rescale_projection=opt.get('rescale_projection_to_1',False)
        use_averaged_intensity=opt.get('use_averaged_intensity',False)
        use_odd_orders_to_0=opt.get('odd_orders_to_0',False)
        use_q_window=opt.get('use_q_window',False)
        
        average_intensity=self.average_intensity.data.astype(np.complex)
        used_orders=self.used_orders
        odd_order_mask=np.array(tuple(used_orders))%2==1

        if dim == 2:
            proj_matrices=self.projection_matrices.copy()
            if use_odd_orders_to_0:
                proj_matrices[:,odd_order_mask]=0
        elif dim == 3:
            proj_matrices=[matrix.copy() for matrix in self.projection_matrices]
            if use_odd_orders_to_0:
                for odd_order in np.array(tuple(used_orders))[odd_order_mask]:
                    #log.info('odd order={}'.format(odd_order))
                    proj_matrices[used_orders[odd_order]][:]=0

            if use_q_window:
                qs = self.grid[:,0,0,0]
                #log.info('qs = {}'.format(qs))
                limits=opt['q_window']
                mask = (qs<limits[0]) |  (qs>limits[1])
                for order in used_orders:
                    if order != 0 :
                        #log.info('applo order wondow in order {}'.format(order))
                        temp = proj_matrices[used_orders[order]]
                        temp[mask]=0
                        proj_matrices[used_orders[order]] = temp
                        #log.info('proj_matrices= {}'.format((proj_matrices[used_orders[order]]==0).any()))
                    
                
        #if use_averaged_intensity:
        #    if dim == 2:
        #        proj_matrices[:,used_orders[0]]=average_intensity
        #    elif dim == 3:
        #        raise NotImplementedError
    
        if rescale_projection:
            max_value=max( (np.max(np.abs(matrix)) for matrix in proj_matrices))
            if max_value==0:
                max_value=1e-12                
            for matrix in proj_matrices:
                matrix/=max_vector_value                
        return proj_matrices

    def generate_coeff_projection(self):
        dim = self.dimensions
        projection_matrices=self.projection_matrices

        used_orders=self.used_orders
        zero_id = used_orders.get(0,'')
        use_averaged_intensity=self.opt.get('use_averaged_intensity',False)
        average_intensity = self.average_intensity.data[:,None]
        #log.info('projection_matrices type={}'.format(type(projection_matrices)))
        if dim == 2:
            mask=self.mask
            where=np.where
            if use_averaged_intensity:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    intensity_harmonic_coefficients[:,zero_id]=average_intensity
                    return intensity_harmonic_coefficients
            else:
                def mtip_projection(intensity_harmonicCoefficients,unknowns):
                    projected_intensity_coefficients_array=intensity_harmonicCoefficients.copy()                
                    projected_intensity_coefficients_array[:,positive_harmonic_orders]=projection_matrices*unknowns
                    #apply mask
                    intensity_harmonic_coefficients=where(mask,projected_intensity_coefficients_array,intensity_harmonicCoefficients).astype(complex)
                    return intensity_harmonic_coefficients
        elif dim == 3:
            radial_mask=self.radial_mask
            dot=np.dot
            copy=np.array #array is faster than copy
            a_factor = 2*np.sqrt(np.pi)

            if use_averaged_intensity:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]           
                        #projected_intensity_coefficients[o_id][~radial_mask,...] = intensity_harmonic_coefficients[o_id][~radial_mask,...]
                        #apply mask
                        #log.info('I0 shape = {}'.format(projected_intensity_coefficients[zero_id].shape))
                    projected_intensity_coefficients[zero_id]=a_factor*average_intensity
                    return projected_intensity_coefficients
            else:
                def mtip_projection(intensity_harmonic_coefficients,unknowns):
                    projected_intensity_coefficients=[coeff.copy() for coeff in intensity_harmonic_coefficients]
                    #log.info('unknowns mean = {}'.format(np.mean(np.abs(unknowns[2]))))
                    #log.info('unknowns  = {}'.format(np.abs(unknowns[0])))
                    #log.info('projection matrix mean = {}'.format(np.mean(np.abs(projection_matrices[2]))))
                    #log.info('projection matrix = {}'.format(np.abs(projection_matrices[0])))
                    for o_id in self.used_orders.values():
                        #log.info('unknowns type={}'.format(type(unknowns)))
                        projected_intensity_coefficients[o_id] = projection_matrices[o_id] @ unknowns[o_id]           
                        #projected_intensity_coefficients[o_id][~radial_mask,...] = intensity_harmonic_coefficients[o_id][~radial_mask,...]
                        #apply mask
                    return projected_intensity_coefficients                
        return mtip_projection

    def generate_approximate_unknowns(self):
        dim = self.dimensions
        used_orders=self.used_orders
        order_ids=tuple(used_orders.values())
        proj_matrices=self.projection_matrices
        radial_points=self.radial_points
        if dim == 2:
            sum=np.sum
            where=np.where
            abs=np.abs
            conjugate=np.conjugate
            def approximate_unknowns(intensity_harmonic_coefficients):
                scalar_prod_Im_vm_summands=intensity_harmonic_coefficients[:,order_ids]*conjugate(projection_matrices)*radial_points[:,None]
                scalar_prod_Im_vm=sum(scalar_prod_Im_vm_summands,axis=0)
                #        log.info(scalar_prod_Im_vm)
                unknowns=scalar_prod_Im_vm/where(abs(scalar_prod_Im_vm)==0,1,abs(scalar_prod_Im_vm))
                unknowns[0]=1
                #log.info('unknowns={}'.format(unknowns))
                return unknowns
        if dim == 3:
            D=np.diag(radial_points) #diagonal matrix of radial points
            PDs=tuple(proj_matrices[_id].T.conj() @ D**2 for _id in order_ids)
            matmul=np.matmul
            svd=np.linalg.svd
            def approximate_unknowns(intensity_harmonic_coefficients):
                #log.info('len harmonic coeff = {}'.format(len(intensity_harmonic_coefficients)))
                BAs=tuple(PD @ I for PD,I in zip(PDs,intensity_harmonic_coefficients))  #each element is B^\dagger A in a Procrustres Problem min|A-BR|
                unknowns=tuple(matmul(*svd(BA,full_matrices=False)[::2]) for BA in BAs)
                #log.info('unknowns={}'.format(unknowns))
                return unknowns             
        return approximate_unknowns
    def generate_apply_SO_freedom_2D(self,opt):
        dim=self.dimensions
        if dim ==2 :
            ordes_dict=self.used_orders
            orders=np.array(tuple(orders_dict))
            order_ids=tuple(orders_dict.values())
            
            radial_high_pass_index=opt['SO_freedom'].get('radial_high_pass',np.nan)
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


class Projections:
    def rank_orders(projection_vector,reciprocal_grid,harmonic_orders):
        weighted_vect=projection_vector*reciprocal_grid[:,0,0,None]/(np.concatenate(([100.0,100.0,6,8],harmonic_orders[4:])))[None,:]
        #log.info('weighted_vect =\n {}'.format(weighted_vect))
        return np.argsort(np.sum(np.sqrt((weighted_vect*weighted_vect.conjugate()).real),axis=0))[::-1]
    
    
class OutpUtprojections:
    pass


# particle number approximation and projection as described in
# K. Pande et.al. PNAS 2018 'Ab initio structure determination from experimental fluctuation X-ray scattering data'
def generate_estimate_particle_number(radial_grid,projection_matrices,used_orders):
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
def generate_negative_shift_operator(reciprocal_grid,fourier_type):
    dim=reciprocal_grid.n_shape[0]
    pre_exponent=2*np.pi
    pi=np.pi
    if fourier_type=='Zernike':
        pre_exponent=1
    if dim == 2:
        def negative_s2hift(reciprocal_density,vector):                        
            v_r=vector[0]
            v_phi=vector[1]
            if v_phi<pi:
                v_phi+=pi
            else:
                v_phi-=pi
            #log.info('center2={}'.format((v_r,v_phi)))
            qs=reciprocal_grid[...,0]
            thetas=reciprocal_grid[...,1]
            #log.info('qs sahpe={}'.format(qs.shape))
            phases=np.exp(-1.j*pre_exponent*v_r*qs*np.cos(thetas-v_phi))
            reciprocal_density*=phases
#            pres.present(np.swapaxes(phases.real,0,1),grid=reciprocal_grid)
            return reciprocal_density
    elif dim == 3:
        cart_grid = mLib.spherical_to_cartesian(reciprocal_grid)
        def negative_shift(reciprocal_density,vector):
            cart_vect = mLib.spherical_to_cartesian(vector)
            log.info('cart_vector = {}'.format(cart_vect))
            
            phases = np.exp(-1.j*pre_exponent*(cart_grid*cart_vect).sum(axis=-1))
            log.info('phases shape = {}'.format(phases.shape))
            reciprocal_density*=phases
            return reciprocal_density        
    return negative_shift

def generate_shift_by_operator(grid,opposite_direction=False):
    dim=grid.n_shape[0]
    pi = np.pi
    if opposite_direction:
        prefactor = -1
    else:
        prefactor = 1
    if dim == 2:
        def shift_by(reciprocal_density,vector):                        
            v_r=vector[0]
            v_phi=vector[1]
            if v_phi<pi:
                v_phi+=pi
            else:
                v_phi-=pi
            #log.info('center2={}'.format((v_r,v_phi)))
            qs=grid[...,0]
            thetas=grid[...,1]
            #log.info('qs sahpe={}'.format(qs.shape))
            phases=np.exp(-1.j*prefactor*v_r*qs*np.cos(thetas-v_phi))
            reciprocal_density*=phases
            #pres.present(np.swapaxes(phases.real,0,1),grid=reciprocal_grid)
            return reciprocal_density
    elif dim == 3:
        cart_grid = mLib.spherical_to_cartesian(grid)
        def shift_by(reciprocal_density,vector):
            #log.info('vector = {}'.format(vector))
            cart_vect = mLib.spherical_to_cartesian(vector)
            log.info('cart shift vect = {}'.format(-cart_vect*prefactor))
            phases = np.exp(-1.j*prefactor*(cart_grid*cart_vect).sum(axis=-1))
            reciprocal_density*=phases
            return reciprocal_density        
    return shift_by

def generate_point_inversion_projection(reciprocal_grid):
    dim=reciprocal_grid.n_shape[0]
    n_angular_points = np.array(reciprocal_grid.shape[1:])
    indices = n_angular_points//2-n_angular_points%2
    slices = (slice(None),)+tuple(slice(None,index) for index in indices)
    def inversion_projection(reciprocal_density):
        inversion_indicator=np.sum(reciprocal_density[slices]).imag
        log.info('inversion indicator ={}'.format(inversion_indicator))
        if inversion_indicator<0:
            reciprocal_density=reciprocal_density.conjugate()
        return reciprocal_density
    return inversion_projection


def rank_orders(projection_vector,reciprocal_grid,harmonic_orders):
    weighted_vect=projection_vector*reciprocal_grid[:,0,0,None]/(np.concatenate(([100.0,100.0,6,8],harmonic_orders[4:])))[None,:]
    #log.info('weighted_vect =\n {}'.format(weighted_vect))
    return np.argsort(np.sum(np.sqrt((weighted_vect*weighted_vect.conjugate()).real),axis=0))[::-1]


def generate_remaining_SO_projection_2D(specifier,reciprocal_grid):
    fxs_data=specifier['fxs_data']
    n_angular_points=reciprocal_grid.shape[1]
    projection_orders=np.concatenate((np.arange(int(n_angular_points/2)+1),-1*np.arange(int(n_angular_points/2)+n_angular_points%2)[:0:-1]))    
    positive_harmonic_orders=specifier['positive_harmonic_orders']
    radial_high_pass_index=specifier['SO_freedom'].get('radial_high_pass',np.nan)
    even_order_mask=positive_harmonic_orders%2==0
    non_zero_mask=positive_harmonic_orders!=0
    order_mask=even_order_mask*non_zero_mask
    harmonic_orders=positive_harmonic_orders[order_mask]
    max_order=np.max(harmonic_orders)
    projection_vector=modify_projection_vector_2D(fxs_data.projection_vector,specifier)[radial_high_pass_index:,order_mask]
    sorted_order_indices=rank_orders(projection_vector,reciprocal_grid[radial_high_pass_index:],harmonic_orders)
    log.info('sorted_order_indices={}'.format(sorted_order_indices))
    first_order_index=sorted_order_indices[0]
    first_order=harmonic_orders[first_order_index]
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
        log.info('remaining_orders={}'.format(harmonic_orders[sorted_order_indices[free_orders_mask]]))
        if free_orders_mask.any()==False:
            break
        else:
            #select next order
            current_order_index=sorted_order_indices[free_orders_mask][0]
            current_order=harmonic_orders[current_order_index]
            log.info('current order ={}'.format(current_order))
            #calculate remaining rotations
            gcd=np.gcd(remaining_rotations,current_order)
            log.info('gcd={}'.format(gcd))
            #use rotational freedom on current_order
            n_independent_rotations=remaining_rotations/gcd
            log.info('n independent rotations={}'.format(n_independent_rotations))
            smallest_angle=2*np.pi/n_independent_rotations
            smallest_angle_coeff=np.argmin((np.arange(1,n_independent_rotations)*current_order/gcd)%n_independent_rotations)+1

            order_indices+=(current_order_index,)
            angle_coeffs+=(smallest_angle_coeff,)
            angles+=(smallest_angle,)
            gcds+=(gcd,)
            remaining_rotations=gcd
    
    def apply_SO_freedom(harmonic_coefficients,fxs_unknowns):
        #log.info('fxs unknowns ={}'.format(fxs_unknowns))
        phases=(-1.j*np.log(fxs_unknowns[order_mask])).real
        #log.info('complete phases ={}'.format(-1.j*np.log(fxs_unknowns[::2])))
        #log.info('phases ={}'.format(phases))
        rotation_phase=0
        for order_index,angle,angle_coeff,gcd in zip(order_indices,angles,angle_coeffs,gcds):
            rotation_phase-=(phases[order_index]//angle)*angle_coeff*angle/gcd
            log.info('order={} phase={}'.format(harmonic_orders[order_index],phases[order_index]))
            log.info('min angle*gcd={}'.format(angle))
            log.info('rotated phase={}'.format((phases[order_index]+harmonic_orders[order_index]*rotation_phase)%(2*np.pi)))
        
        harmonic_coefficients*=np.exp(1.j*projection_orders*rotation_phase)
        return harmonic_coefficients
    return apply_SO_freedom

def generate_remaining_SO_projection(specifier,reciprocal_grid):
    dim=reciprocal_grid.n_shape[0]
    if dim == 2:
        apply_SO_freedom=generate_remaining_SO_projection_2D(specifier,reciprocal_grid)
    elif dim == 3:
        def apply_SO_freedom(harmonic_coefficients,fxs_unknowns):            
            raise NotImplementedError()        
    return apply_SO_freedom


def generate_q_window_projection(r_qrid,limits):
    qs = r_grid[:,0,0,0]
    mask = (qs < limit[0]) | (qr > limit[1])
