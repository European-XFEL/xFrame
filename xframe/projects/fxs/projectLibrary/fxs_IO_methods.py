import numpy as np
import logging

from xframe.library.gridLibrary import NestedArray
from xframe.library import pythonLibrary as pyLib
from xframe.library.pythonLibrary import get_L2_cache_split_parameters
#from xframe.presenters.matplolibPresenter import heatPolar2D
from xframe.library.mathLibrary import SphericalIntegrator,PolarIntegrator
from xframe import settings
from xframe import Multiprocessing
from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_2d
from .fxs_invariant_tools import harmonic_coeff_to_deg2_invariants_3d
from .fxs_invariant_tools import ccd_associated_legendre_matrices_single_m
from .fxs_invariant_tools import ccd_associated_legendre_matrices
from .fxs_invariant_tools import rank_projection_matrices

from xframe.library.pythonLibrary import generate_conjugate
from xframe.library.pythonLibrary import DictNamespace
from xframe.library.physicsLibrary import ewald_sphere_theta_pi
log=logging.getLogger('root')
#pres=heatPolar2D()

### FXS Input Output Methods
class HIOProjection:
    def __init__(self,beta, considered_projections = ['all']):
        self._beta = [beta]
        if not isinstance(considered_projections,(list,tuple)):
            considered_projections = ['all']
        elif len(considered_projections)==0:
            considered_projections = ['all']            
        self.considered_projections = considered_projections
        self.projection = self.generate_HIO()
    @property
    def beta(self):
        return self._beta[0]
    @beta.setter
    def beta(self,value):
        self._beta[0] = value

    def generate_HIO(self):
        beta = self._beta
        where = np.where
        considered_projections = self.considered_projections

        if len(considered_projections)==1:
            name = considered_projections[0]
            def assemble_masks(masks_dict):
                return masks_dict[name]
        else:
            def assemble_masks(masks_dict):
                out_invalid_mask = False
                for name in considered_projections:
                    out_invalid_mask |= mask_dict[name]
                return out_invalid_mask
            
        def hybrid_input_output(without_projection,projection_out,_input):
            out,mask_dict = projection_out
            out_invalid_mask = assemble_masks(mask_dict)
            diff = without_projection-out
            negative_feedback = _input-beta[0]*(diff)
            # negative feedback above is equivalent to Fienup HIO if out is 0 on out_invalid_mask.
            new_input = where(out_invalid_mask,negative_feedback,out)
            return new_input
        return hybrid_input_output

    
def error_reduction(out_without_projection,out,_input):
    return np.array(out[0])


def no_zero_decorator(fun):
    def new_fun(*args,**kwargs):
        val = fun(*args,**kwargs)
        if val <= 0:
            val = 1e-12
        return val
    return new_fun

def get_integrator(dim,grid):
    if dim == 2:
        i = PolarIntegrator(grid[:])
    elif dim == 3:
        i = SphericalIntegrator(grid[:])
    return i

def generate_fxs_error_routine(grid_pair,opt):
    reciprocal_grid = grid_pair.reciprocalGrid
    dim=reciprocal_grid.n_shape[0]
    integrate = get_integrator(dim,reciprocal_grid).integrate_normed
    def error_routine(intensity,projected_intensity):
        diff_norm = integrate(np.square(np.abs(intensity-projected_intensity)))
        proj_norm = integrate(np.square(np.abs(projected_intensity)))
        error = diff_norm/proj_norm
        return error
    return error_routine

def generate_l2_rel_diff_error_routine(grid_pair,_type='real',mask = True):
    if _type == 'reciprocal':
        grid = grid_pair.reciprocalGrid
        pair_id = 0
        power = 1
    else:            
        grid = grid_pair.realGrid
        pair_id = 1
        power = 2
    dim=grid.n_shape[0]
    integrate = get_integrator(dim,grid).integrate

    nabs = np.abs
    nsquare = np.square
    def error_routine(values,projected_values):
        if _type=='real':
            projected_values = projected_values[0]
        #square_diff = (nabs(values - projected_values).real)**2
        #square = (nabs(values).real)**2
        diff = values - projected_values
        square_diff = (diff*diff.conj()).real
        square = (values*values.conj()).real
        square_diff[~mask]=0
        square[~mask]=0
        diff_l2 = integrate(square_diff)
        value_l2 = integrate(square)
        if value_l2!=0:
            error = diff_l2/value_l2
        else:
            error = np.inf
        return error
    return error_routine


def generate_l2_rel_diff_error_routine_cache_aware(grid_pair,L2_cache,_type='real',mask = True):
    if _type == 'reciprocal':
        grid = grid_pair.reciprocalGrid
        pair_id = 0
        power = 1
    else:            
        grid = grid_pair.realGrid
        pair_id = 1
        power = 2
    dim=grid.n_shape[0]
    integrate = get_integrator(dim,grid).integrate

    nabs = np.abs
    nsquare = np.square

    data_shape = grid[:].shape[:-1]
    tmp_square_diff = np.zeros(data_shape,dtype = complex)
    tmp_square = np.zeros(data_shape,dtype = complex)
    #L2 cache divided by two since I will be working on two equally sized matrices tmp_diff and tmp_square_diff
    splitting_index, step = get_L2_cache_split_parameters(data_shape,tmp_square.dtype,L2_cache/2)
    mult = np.multiply

    def loop_0(diff,values):
        for i in range(0,data_shape[0],step):
            i2 = i+step
            d = diff[i:i2]
            v = values[i:i2]
            mult(d,d.conj(),out = tmp_square_diff[i:i2])
            mult(v,v.conj(),out = tmp_square[i:i2])

    def loop_1(diff,values):
        for j in range(data_shape[0]):
            for i in range(0,data_shape[1],step):
                i2 = i+step
                d = diff[j,i:i2]
                v = values[j,i:i2]
                mult(d,d.conj(),out = tmp_square_diff[j,i:i2])
                mult(v,v.conj(),out = tmp_square[j,i:i2])

    def loop_2(diff,values):
        for k in range(data_shape[0]):
            for j in range(data_shape[1]):
                for i in range(0,data_shape[2],step):
                    i2 = i+step
                    d = diff[k,j,i:i2]
                    v = values[k,j,i:i2]
                    mult(d,d.conj(),out = tmp_square_diff[k,j,i:i2])
                    mult(v,v.conj(),out = tmp_square[k,j,i:i2])
                    
    if splitting_index == 0:
        square = loop_0
    elif splitting_index == 1:
        square = loop_1
    elif splitting_index == 2:
        square = loop_2
        
    def error_routine(values,projected_values):
        if _type=='real':
            projected_values = projected_values[0]
        diff = values - projected_values
        square(diff,values)
        tmp_square_diff[~mask]=0
        tmp_square[~mask]=0
        diff_l2 = integrate(tmp_square_diff.real)
        #diff_l2 = np.sum(tmp_square_diff.real)
        value_l2 = integrate(tmp_square.real)
        #value_l2 = np.sum(tmp_square.real)
        if value_l2!=0:
            error = diff_l2/value_l2
        else:
            error = np.inf
        return error
    
    if splitting_index<0:
        error_routine = generate_l2_rel_diff_error_routine(grid_pair,_type=_type)
    return error_routine


def generate_l2_rel_diff_error_routine_gpu(grid_pair,_type='real'):
    if _type == 'reciprocal':
        grid = grid_pair.reciprocalGrid
        pair_id = 0
        power = 1
    else:            
        grid = grid_pair.realGrid
        pair_id = 1
        power = 2
    dim=grid.n_shape[0]
    integrate = get_integrator(dim,grid).integrate

    nabs = np.abs
    nsquare = np.square
    shape = grid[:].shape[:-1]
    size = np.prod(shape)
    
    kernel_str = """
    __kernel void
    calc_squares(
     double2* in_val, 
     double2* in_projected_val, 
     double* out_val_abs,
     double* out_diff_abs,
     long nr_ntheta,
     long ntheta
    )
    {
    
    long i = get_global_id(0); 
    long j = get_global_id(1);
    long k = get_global_id(2); 
    long index = i*nr_ntheta+j*ntheta+k;

    //compute abs^2 of in_val
    double2 val = in_val[index];
    out_val_abs[index] = pown(val.x,2)+pown(val.y,2);
    //compute abs^2 of in_projected_val-in_val
    double2 diff = in_projected_val[index]-val;
    out_diff_abs[index] =  pown(diff.x,2)+pown(diff.y,2);   

    }  
    """

    kernel_dict={
            'kernel': kernel_str,
            'name': 'error_squares',
            'functions': [{
                'name': 'calc_squares',
                'dtypes' : [complex,complex,float,float,np.int64,np.int64],
                'shapes' : [shape,shape,shape,shape,None,None],
                'arg_roles' : ['input','input','output','output','const_input','const_input'],
                'const_inputs' : [None,None,None,None,shape[0]*shape[1],shape[1]],
                'global_range' : shape,
                'local_range' : None
            }]
        }

    calc_error_squares_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict)
    calc_error_squares = Multiprocessing.comm_module.add_gpu_process(calc_error_squares_process)
    
    def error_routine(values,projected_values):
        #square_diff = (nabs(values - projected_values).real)**2
        #square = (nabs(values).real)**2
        square,square_diff =  calc_error_squares(values,projected_values)
        #log.info(square)
        diff_l2 = integrate(square_diff)
        value_l2 = integrate(square)
        if value_l2!=0:
            error = diff_l2/value_l2
        else:
            error = np.inf
        #log.info('error = {}'.format(error))
        return error
    log.info('created error routine.')
    return error_routine


def generate_real_l2_rel_diff_error_routine(grid_pair,**kwargs):
    use_gpu = settings.project.GPU.get('error_squares',False)
    limit_to_initial_mask = kwargs.get('inside_initial_support',False)
    if limit_to_initial_mask:
        #log.info('limit to initial support \n\n\n\n') 
        mask = kwargs['initial_mask']
    else:
        mask = True
        
    if settings.general.cache_aware:
        error_routine = generate_l2_rel_diff_error_routine_cache_aware(grid_pair,settings.general.L2_cache,_type='real',mask = mask)
    else:
        error_routine = generate_l2_rel_diff_error_routine(grid_pair,_type='real',mask = mask)
    return error_routine
def generate_reciprocal_l2_rel_diff_error_routine(grid_pair,**kwargs):
    use_gpu = settings.project.GPU.get('error_squares',False)
    if settings.general.cache_aware:
        error_routine = generate_l2_rel_diff_error_routine_cache_aware(grid_pair,settings.general.L2_cache,_type='reziprocal')
    else:
        error_routine = generate_l2_rel_diff_error_routine(grid_pair,_type='reciprocal')
    def error_correct_arguments(values,projected_values,harmonic_coefficients):
        return error_routine(values,projected_values)
    return error_correct_arguments


def generate_deg2_invariant_l2_diff(grid_pair,**opt):
    reference_invariant = opt['deg2_invariants']
    #log.info('Bm shape = {}'.format(reference_invariant.shape))
    used_orders = opt['used_orders']
    #log.info('\n\n used_orders = {}'.format(used_orders))
    #log.info('deg 2 invariant shape = {}\n'.format(reference_invariant.shape))
    n_particles = opt['n_particles']
    invariant_mask = opt['invariant_mask']
    #log.info('all qs valid = {}'.format(invariant_mask.all()))
    reference_invariant=reference_invariant[np.array(list(used_orders.keys())).astype(int)]
    grid = grid_pair.reciprocalGrid[:]
    dimensions = grid.shape[-1]
    radial_points = grid.__getitem__((slice(None),)+(0,)*dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
    if dimensions == 2:
        invariant_error = _generate_deg2_invariant_diff_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask)
    elif dimensions == 3:
        invariant_error = _generate_deg2_invariant_diff_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask)
    return invariant_error
def generate_deg2_ranked_invariant_l2_diff(grid_pair,**opt):
    reference_invariant = opt['deg2_invariants']
    projection_matrices = opt['projection_matrices']
    #log.info('Bm shape = {}'.format(reference_invariant.shape))
    used_orders = opt['used_orders']
    order_array=np.array(list(used_orders.keys())).astype(int)    
    #log.info('\n\n used_orders = {}'.format(used_orders))
    #log.info('deg 2 invariant shape = {}\n'.format(reference_invariant.shape))
    n_particles = opt['n_particles']
    invariant_mask = opt['invariant_mask']
    #log.info('all qs valid = {}'.format(invariant_mask.all()))
    reference_invariant=reference_invariant[np.array(list(used_orders.keys())).astype(int)]
    grid = grid_pair.reciprocalGrid[:]
    dimensions = grid.shape[-1]
    radial_points = grid.__getitem__((slice(None),)+(0,)*dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
    error_order = opt.get('order',False)
    no_error_order_specified = (not isinstance(error_order,int)) or isinstance(error_order,bool)
    
    if dimensions == 2:        
        invariant_error = _generate_deg2_invariant_diff_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask)
        if no_error_order_specified:
            ranked_order_indices,ranked_orders,metric = rank_projection_matrices(2,projection_matrices,order_array,radial_points)
            error_id = ranked_order_indices[0]
        else:
            error_id = used_orders[error_order]            
                                                                             
    elif dimensions == 3:
        invariant_error = _generate_deg2_invariant_diff_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask)
        error_order = opt.get('order',False)
        if no_error_order_specified:
            ranked_order_indices,ranked_orders,metric = rank_projection_matrices(3,projection_matrices,order_array,radial_points)
            error_id = ranked_order_indices[0]
        else:
            error_id = used_orders[error_order]
            
    def ranked_deg2_error(values,projected_values,Ilms):
        invariant_errors = invariant_error(values,projected_values,Ilms)
        #log.info('error_order = {}'.format(error_id))
        return invariant_errors[error_id]
    return ranked_deg2_error
def _generate_deg2_invariant_diff_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask):
    qq = radial_points[None,:,None]*radial_points[None,None,:]
    reference_masked = reference_invariant.copy()
    reference=reference_masked.copy()
    #log.info(qq)
    norm= np.sum(reference*reference.conj(),axis=(1,2)).real
    #log.info(norm)
    non_zero_mask = norm !=0
    n_orders=len(norm)
    errors = np.full(n_orders,-1,dtype = float)
    non_zero_norm = norm[non_zero_mask]
    order_array = np.array(tuple(used_orders.values()))    
    mask = np.zeros(reference_masked.shape,dtype = bool)
    zero_id = used_orders[0]
    def invariant_error(values,projected_values,Ims):
        Bm =  harmonic_coeff_to_deg2_invariants_2d(Ims)
        #log.info('Bl0-ref0 = {}'.format(reference_masked/sqrt(n_particles[0])))
        #log.info("bm shape/type = {}/{} reference shape/type = {}/{}".format(Bm.shape,Bm.dtype,reference.shape,reference.dtype))
        Bm= Bm[order_array].copy()
        #Bm[mask] = 0
        #log.info('ref0 = {}'.format(reference_masked[0,1,:10]/n_particles[0]))
        #log.info('values= {}'.format(Ims[9][1,:10]))
        #log.info('Bl0-ref0= {}'.format((Bl[0,1,:10]*n_particles[0]-reference_masked[0,1,:10])/reference_masked[0,1,:10]))
        reference[zero_id] = reference_masked[zero_id]/n_particles[0]   
        diff = reference - Bm
        #squared_scaled_diff = (qq*diff*diff.conj()).real
        squared_scaled_diff = (diff*diff.conj()).real
        norm_diff = np.sum(squared_scaled_diff,axis=(1,2))
        errors[non_zero_mask] = norm_diff[non_zero_mask]/non_zero_norm
        return errors.copy()
    
    def invariant_error_old(values,projected_values,Ims):
        Bm = harmonic_coeff_to_deg2_invariants_2d(Ims)
        diff = reference_invariant - Bm[order_array]
        errors[non_zero_mask] = np.sum(qq*diff*diff.conj(),axis=(1,2))[non_zero_mask]/non_zero_norm
        return errors.copy()
    return invariant_error

def _generate_deg2_invariant_diff_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask):
    qq = (radial_points[None,:,None]*radial_points[None,None,:])**2
    #reference_invariant=reference_invariant[:,30:,30:]
    #log.info('deg2 shape = {}'.format(reference_invariant.shape))
    #log.info('first values = {}'.format(reference_invariant[0,1,:10]))
    order_array = np.array(tuple(used_orders.values()))
    log.info('used orders = {}'.format(order_array))
    zero_id = used_orders[0]
    reference_masked = reference_invariant.copy()
    #log.info('refference shape = {}'.format(reference_masked.shape))
    mask = np.zeros(reference_masked.shape,dtype = bool)
    #log.info('invariant mask shape = {}, containes unmasked points = {}'.format(invariant_mask.shape,invariant_mask.any()))
    mask[:] = ~invariant_mask[order_array]
    #log.info('mask shape = {} '.format(reference_masked.shape))
    reference_masked[mask]=0
    reference=reference_masked.copy()
    sqrt = np.sqrt
    #norm= np.sum(qq*reference*reference.conj(),axis=(1,2))
    norm= np.sum(reference*reference.conj(),axis=(1,2))
    non_zero_mask = norm !=0
    n_orders=len(norm)
    errors = np.full(n_orders,-1,dtype = float)
    non_zero_norm = norm[non_zero_mask]
    log.info('first values = {}'.format(reference[0,1,:10]))
    def invariant_error(values,projected_values,Ims):
        Bl =  harmonic_coeff_to_deg2_invariants_3d(Ims)
        #log.info('Bl0-ref0 = {}'.format(reference_masked/sqrt(n_particles[0])))
        Bl= Bl[order_array].copy()
        Bl[mask] = 0
        #log.info('ref0 = {}'.format(reference_masked[0,1,:10]/n_particles[0]))
        #log.info('values= {}'.format(Ims[9][1,:10]))
        #log.info('Bl0-ref0= {}'.format((Bl[0,1,:10]*n_particles[0]-reference_masked[0,1,:10])/reference_masked[0,1,:10]))
        reference[zero_id] = reference_masked[zero_id]/n_particles[0]   
        diff = reference - Bl
        #squared_scaled_diff = (qq*diff*diff.conj()).real
        squared_scaled_diff = (diff*diff.conj()).real
        norm_diff = np.sum(squared_scaled_diff,axis=(1,2))
        errors[non_zero_mask] = norm_diff[non_zero_mask]/non_zero_norm
        return errors.copy()
    return invariant_error

def _generate_deg2_invariant_diff_3d_old(radial_points,reference_invariant,used_orders):
    qq = (radial_points[None,:,None]*radial_points[None,None,:])**2
    #reference_invariant=reference_invariant[:,30:,30:]
    norm= np.sum(reference_invariant*reference_invariant.conj(),axis=(1,2))
    non_zero_mask = norm !=0
    n_orders=len(norm)
    errors = np.full(n_orders,-1,dtype = float)
    non_zero_norm = norm[non_zero_mask]
    order_array = np.array(tuple(used_orders.values()))
    def invariant_error(Ims):
        Bl = harmonic_coeff_to_deg2_invariants_3d(Ims)
        #Bl = Bl[:,30:,30:]
        #log.info('len ref = {} len bl = {}'.format(len(reference_invariant),len(Bl)))
        diff = reference_invariant - Bl[order_array]
        norm_diff = np.sum(diff*diff.conj(),axis=(1,2))
        #log.info(norm_diff)
        errors[non_zero_mask] = norm_diff[non_zero_mask]/non_zero_norm
        #log.info(errors)
        return errors.copy()
    return invariant_error



def generate_fqc_error(grid_pair,**opt):
    reference_invariant = opt['deg2_invariants']
    used_orders = opt['used_orders']
    #log.info('\n\n used_orders = {}'.format(used_orders))
    #log.info('deg 2 invariant shape = {}\n'.format(reference_invariant.shape))
    n_particles = opt['n_particles']
    invariant_mask = opt['invariant_mask']
    xray_wavelength = opt['xray_wavelength']
    log.info('all qs valid = {}'.format(invariant_mask.all()))
    reference_invariant=reference_invariant[np.array(list(used_orders.keys())).astype(int)]
    grid = grid_pair.reciprocalGrid[:]
    dimensions = grid.shape[-1]
    radial_points = grid.__getitem__((slice(None),)+(0,)*dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
    if dimensions == 2:
        invariant_error = _generate_fqc_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength)
    elif dimensions == 3:
        invariant_error = _generate_fqc_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength)
    return invariant_error
def _generate_fqc_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength):
    qq = radial_points[None,:,None]*radial_points[None,None,:]
    log.info(qq)
    norm= np.sum(qq*reference_invariant*reference_invariant.conj(),axis=(1,2))
    log.info(norm)
    non_zero_mask = norm !=0
    n_orders=len(norm)
    errors = np.full(n_orders,-1,dtype = float)
    non_zero_norm = norm[non_zero_mask]
    order_array = np.array(tuple(used_orders.values()))
    def invariant_error(values,projected_values,Ims):
        Bm = harmonic_coeff_to_deg2_invariants_2d(Ims)
        diff = reference_invariant - Bl[order_array]
        errors[non_zero_mask] = np.sum(qq*diff*diff.conj(),axis=(1,2))[non_zero_mask]/non_zero_norm
        return errors.copy()
    return invariant_error

def _generate_fqc_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength):
    order_array = np.array(tuple(used_orders.keys())).astype(int)
    order_ids = np.array(tuple(used_orders.values()))
    max_order = np.max(order_array)
    thetas = ewald_sphere_theta_pi(xray_wavelength,radial_points)    
    P_matrix = np.moveaxis(ccd_associated_legendre_matrices(thetas,max_order,max_order),-1,0)

    reference_bl_masked = reference_invariant.copy()
    mask = np.zeros(reference_bl_masked.shape,dtype = bool)
    mask[:] = ~invariant_mask[order_ids]
    reference_bl_masked[mask]=0
    
    def calc_ccn(bl):
        return np.sum(bl[1:,...,None]*P_matrix[1:],axis = 0)
    def calc_2ccn_average(c1,c2):
        return (c1[...,0]*c2[...,0]).real + 2*np.sum(c1[...,1:]*c2[...,1:].conj(),axis = -1).real
    
    reference_ccn = calc_ccn(reference_bl_masked)
    reference_average = calc_2ccn_average(reference_ccn,reference_ccn)
    reference_weights = (P_matrix[1:,...,0]*reference_ccn[None,...,0]).real + 2*np.sum(P_matrix[1:,...,1:]*reference_ccn[None,...,1:].conj(),axis = -1).real    
    
    sqrt = np.sqrt
    fqc = np.ones_like(reference_average)
    
    def fqc_error(values,projected_values,Ims):
        Bl =  harmonic_coeff_to_deg2_invariants_3d(Ims)
        #Bl = np.random.rand(*(Bl_r.shape)).astype(Bl_r.dtype)
        #Bl[0]=Bl_r[0]
        Bl[mask]=0
        ccn = calc_ccn(Bl)
        average = calc_2ccn_average(ccn,ccn)
        norm = sqrt(average*reference_average)
        positive_mask = (norm>=0)

        #control_average = calc_2ccn_average(ccn,reference_ccn)
        control_average = np.sum(Bl[1:]*reference_weights,axis = 0)
        
        fqc[positive_mask] = control_average[positive_mask]/norm[positive_mask]
        fqc[~positive_mask] = 1
        errors = tuple(1-np.mean(fqc[i,:i+1]) for i in range(len(radial_points)))        
        return np.array(errors)
    return fqc_error



def generate_II_error(grid_pair,**opt):
    reference_invariant = opt['deg2_invariants']
    used_orders = opt['used_orders']
    #log.info('\n\n used_orders = {}'.format(used_orders))
    #log.info('deg 2 invariant shape = {}\n'.format(reference_invariant.shape))
    n_particles = opt['n_particles']
    invariant_mask = opt['invariant_mask']
    xray_wavelength = opt['xray_wavelength']
    log.info('all qs valid = {}'.format(invariant_mask.all()))
    reference_invariant=reference_invariant[np.array(list(used_orders.keys())).astype(int)]
    grid = grid_pair.reciprocalGrid[:]
    dimensions = grid.shape[-1]
    radial_points = grid.__getitem__((slice(None),)+(0,)*dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
    if dimensions == 2:
        invariant_error = _generate_II_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength)
    elif dimensions == 3:
        invariant_error = _generate_II_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength)
    return invariant_error
def _generate_II_2d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength):
    qq = radial_points[None,:,None]*radial_points[None,None,:]
    log.info(qq)
    norm= np.sum(qq*reference_invariant*reference_invariant.conj(),axis=(1,2))
    log.info(norm)
    non_zero_mask = norm !=0
    n_orders=len(norm)
    errors = np.full(n_orders,-1,dtype = float)
    non_zero_norm = norm[non_zero_mask]
    order_array = np.array(tuple(used_orders.values()))
    def invariant_error(values,projected_values,Ims):
        Bm = harmonic_coeff_to_deg2_invariants_2d(Ims)
        diff = reference_invariant - Bl[order_array]
        errors[non_zero_mask] = np.sum(qq*diff*diff.conj(),axis=(1,2))[non_zero_mask]/non_zero_norm
        return errors.copy()
    return invariant_error

def _generate_II_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,xray_wavelength):
    order_array = np.array(tuple(used_orders.keys())).astype(int)
    order_ids = np.array(tuple(used_orders.values()))
    max_order = np.max(order_array)
    thetas = ewald_sphere_theta_pi(xray_wavelength,radial_points)    
    P_matrix = np.moveaxis(ccd_associated_legendre_matrices(thetas,max_order,max_order),-1,0)

    reference_bl_masked = reference_invariant.copy()
    mask = np.zeros(reference_bl_masked.shape,dtype = bool)
    mask[:] = ~invariant_mask[order_ids]
    reference_bl_masked[mask]=0

    # The used spherical harmonic transform already include the factor sqrt(4 \pi/(2l+1)) in the spherical harmonics they are orthonormal.
    # This factor is as well included in the extraction routine for the Bl coefficients, which means we only need to sum over the Bl coefficients.
    reference_II = np.sum(reference_bl_masked[1:],axis = 0)
    reference_II_squared = (reference_II*reference_II.conj()).real
    positive_mask = reference_II_squared>0
    current_II = np.zeros_like(reference_II)
    #errors = np.zeros_like(reference_II)
    qq = (radial_points[:,None]*radial_points[None,:])**2
    
    def II_error(values,projected_values,Ims):
        #log.info('fuuuuuuu')
        #print("fu")
        Bl =  harmonic_coeff_to_deg2_invariants_3d(Ims)
        Bl[mask]=0
        np.sum(Bl[1:],axis = 0,out = current_II)
        diff = reference_II-current_II
        square_diff = (diff*diff.conj()).real
        #log.info('considered fraction {}'.format(np.sum(positive_mask)/np.prod(positive_mask.shape)))

        #errors=square_diff[positive_mask]/reference_II_squared[positive_mask]
        #error = np.median(errors)
        
        #error = np.sum(square_diff[positive_mask]*qq[positive_mask])/np.sum(reference_II_squared[positive_mask]*qq[positive_mask])

        error = 1- np.sum(current_II*reference_II*qq)/np.sqrt(np.sum(current_II**2*qq)*np.sum(reference_II**2*qq))
        return error
    return II_error


def generate_ccd_diff(grid_pair,**opt):
    '''
    calculates the difference in l_2 norm between the angularly integrated averaged cross-correlation $\frac{1}{2\pi} \int_0^{2\pi} C(q,q',\phi) d\phi$( or equivalently the zeros harmonic coefficient $C_0(q,q')$ of the averaged cross-correlation ) calculated from reconstructed intensity and the one coming from the input parameters.
    All this is possible due to the following relation $$ C_0(q,q') =  \sum_l B_l \overline{P}^0_l(q) \overline{P}^0_l(q') $$.
    Number of particles scaling is taken into account.
    min_order: Defines the starting order in the sum $\sum_l B_l \overline{P}^0_l(q) \overline{P}^0_l(q')$. Can be used to exclude 0'th order.
    '''
    reference_invariant = opt['deg2_invariants']
    C_order = opt['C_order']
    used_orders = opt['used_orders']
    n_particles = opt['n_particles']
    invariant_mask = opt['invariant_mask']
    xray_wavelength = opt['xray_wavelength']
    reference_invariant=reference_invariant[np.array(list(used_orders.keys())).astype(int)]
    grid = grid_pair.reciprocalGrid[:]
    dimensions = grid.shape[-1]
    radial_points = grid.__getitem__((slice(None),)+(0,)*dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
    if dimensions == 2:
        NotImplementedError('ccd error routine is not jet implemented for 2D reconstructions.')
    elif dimensions == 3:
        invariant_error = _generate_ccd_diff_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,C_order,xray_wavelength)
    return invariant_error

def _generate_ccd_diff_3d(radial_points,reference_invariant,used_orders,n_particles,invariant_mask,C_order,xray_wavelength):
    #qq = (radial_points[None,:,None]*radial_points[None,None,:])**2        
    order_array = np.array(tuple(used_orders.values()))
    zero_id = used_orders[0]

    relevant_orders_mask = order_array>=C_order
    thetas = ewald_sphere_theta_pi(xray_wavelength,radial_points)                       
    PlmPlm = np.moveaxis(np.squeeze(ccd_associated_legendre_matrices_single_m(thetas,np.max(order_array),C_order)),-1,0)[order_array]
    PlmPlm[zero_id] = 0
    PlmPlm = PlmPlm[relevant_orders_mask]
    PlmPlm[np.isnan(PlmPlm)] =0
    
    reference_masked = reference_invariant.copy()    
    mask = np.zeros(reference_masked.shape,dtype = bool)    
    mask[:] = ~invariant_mask[order_array]    
    reference_masked[mask]=0    
    reference=reference_masked.copy()
    ref_C_m = np.sum(reference_masked[relevant_orders_mask]*PlmPlm,axis = 0)
    norm= np.sum(ref_C_m*ref_C_m.conj())
    assert norm !=0,'C_m seems to be zero, abbort creation of error routine.'
    
    sqrt = np.sqrt    
    def ccd_error(values,projected_values,Ims):
        Bl =  harmonic_coeff_to_deg2_invariants_3d(Ims)[order_array]
        Bl[mask] = 0
        Bl[zero_id] *= sqrt(n_particles[0])
        C_m_diff = np.sum(Bl[relevant_orders_mask]*PlmPlm,axis = 0)-ref_C_m
        squared_scaled_diff = (C_m_diff*C_m_diff.conj()).real
        norm_diff = np.sum(squared_scaled_diff)
        error = norm_diff/norm
        return error
    return ccd_error


def generate_support_size(grid_pair,**opt):    
    def support_size(values,projected_values):
        return np.sum((values.real == projected_values.real)**2)/np.prod(values.shape)
    return support_size
    
error_generators={'real':
                  {'l2_projection_diff':generate_real_l2_rel_diff_error_routine,
                   'support_size':generate_support_size
                   },
                  'reciprocal':{                  
                      'l2_projection_diff':generate_reciprocal_l2_rel_diff_error_routine,
                      'deg2_invariant_l2_diff':generate_deg2_invariant_l2_diff,
                      'ccd_diff':generate_ccd_diff,
                      'fqc_error':generate_fqc_error,
                      'II_error':generate_II_error,
                      'deg2_ranked_invariant_l2_diff':generate_deg2_ranked_invariant_l2_diff
                  }
                  }
error_number_of_arguments = {'real':2,'reciprocal':3}

def generate_error_routines(error_opt,grid_pair,**kwargs):
    routine_names = error_opt.methods    
    #log.info('error opts = {}'.format(routine_options))
    error_routines={}
    #log.info('gen_errors {}'.format(routine_names))
    for category,copt in routine_names.items():
        if not 'calculate' in copt:
            continue
        name_list = copt.calculate
        if isinstance(name_list,list):
            error_routines[category]={}            
            for err_name in name_list:
                generator = error_generators[category][err_name]
                opt = copt.get(err_name,DictNamespace()).dict()
                #log.info(opt)
                kwargs.update(opt)
                error_routines[category][err_name] = generator(grid_pair,**kwargs)
    category_routines={}
    for category,routines in error_routines.items():
        if len(routines)>=1:
            combined_routine = combine_error_routines(routines)
        else:
            def mock_routine(*args):
                return {}
            combined_routine = mock_routine
        category_routines[category] = [combined_routine,error_number_of_arguments[category]]

    #log.info('return all error routines')
    return category_routines

def combine_error_routines(error_function_dict,n_arguments=2):
    ''' Returns callable that calculates all error routines in error_function_dict and stores their values in a dict with the same keys.
    If error_flnction_dict containes no entries it just returns False'''
    def error_metrics(*args):
        #log.info('len error arguments = {}'.format(len(args)))
        errors={}
        for key,f in error_function_dict.items():
            errors[key]=f(*args)
        return errors
    return error_metrics

def generate_main_error_routine(error_names,_type):    
    if _type == 'mean':
        method = np.mean
    elif _type == 'min':
        method = np.min
    elif _type == 'max':
        method = np.max
    elif _type == 'prod':
        method = np.prod
    
    def main_error(error_dict):
        try:
            real_errors = [error_dict['real'][name][-1] for name in error_names['real']]
            reciprocal_errors = [error_dict['reciprocal'][name][-1] for name in error_names['reciprocal']]
            err_array=np.array(real_errors+reciprocal_errors)
            return method(err_array)
        except IndexError:
            return -1
        #return method(err_array)
    return main_error
