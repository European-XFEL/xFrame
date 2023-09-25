import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import numpy as np
import scipy
import logging
import traceback
from xframe import log
os.chdir(os.path.expanduser('~/Programs/xframe'))
from xframe.startup_routines import load_recipes
from xframe.startup_routines import dependency_injection_no_soft
from xframe.library import mathLibrary
from xframe.externalLibraries.flt_plugin import LegendreTransform
mathLibrary.leg_trf = LegendreTransform
from xframe.externalLibraries.shtns_plugin import sh
mathLibrary.shtns = sh
from xframe.control.Control import Controller
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_to_deg2_invariant
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_remove_0_order
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import cross_correlation_mask
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_apply_precision_filter
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import ccd_to_deg2_invariant_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import bl_3d_pseudo_inverse_worker
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_cc_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import density_to_deg2_invariants_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_to_projection_matrices_3d
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import deg2_invariant_eigenvalues
from xframe.plugins.MTIP.analysisLibrary import fxs_invariant_tools as i_tools
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict_zernike_spherical
from xframe.plugins.MTIP.analysisLibrary.harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_zernike_spherical_ft
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid
from xframe.plugins.MTIP.analysisLibrary.fxs_Projections import ReciprocalProjection
from xframe import settings
from xframe.library import mathLibrary as mLib
from xframe import Multiprocessing
from xframe.library import physicsLibrary as pLib
from xframe.library.gridLibrary import GridFactory
from xframe.presenters.matplolibPresenter import heat2D_multi,plot1D
log=log.setup_custom_logger('root','INFO')

dependency_injection_no_soft()
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_100')[0]
db = analysis_sketch[1]
opt = settings.analysis
controller = Controller(analysis_sketch)
from xframe.presenters.matplolibPresenter import heat2D,heat2D_multi

ccd = db.load('ccd')
cc = ccd['cross_correlation']


def create_functions():
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
    floatSum(__global double* inVector, __global double* outVector, const int inVectorSize, __local double* resultScratch){
    int gid = get_global_id(0);
    int wid = get_local_id(0);
    int wsize = get_local_size(0);
    int grid = get_group_id(0);
    int grcount = get_num_groups(0);

    int i;
    int workAmount = inVectorSize/grcount;
    int startOffset = workAmount * grid + wid;
    int maxOffset = workAmount * (grid + 1);
    if(maxOffset > inVectorSize){
        maxOffset = inVectorSize;
    }
    resultScratch[wid] = 0.0;
    for(i=startOffset;i<maxOffset;i+=wsize){
            resultScratch[wid] += inVector[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid == 0){
    for(i=1;i<wsize;i++){
    resultScratch[0] += resultScratch[i];
    }
    outVector[grid] = resultScratch[0];
    }
    }


    __kernel void
    reduce(__global double* buffer,
    __global double* result,
    __const int length,
    __local double* scratch
    ) {
    int global_index = get_global_id(0);
    double accumulator = INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
    double element = buffer[global_index];
    accumulator = (accumulator < element) ?
    accumulator : element;
    global_index += get_global_size(0);
    }
    // Perform parallel reduction
    int local_id = get_local_id(0);
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
    
    """).build()
    sum1 = kernel.floatSum
    sum2 = kernel.reduce
    return locals()

locals().update(create_functions())

n=60000
a = np.ones(n,dtype = np.float64)

local_range = (np.int32(32),)
global_range = (np.int32(n),)

scratch=np.zeros(local_range,dtype = np.float64)
out = np.zeros(n,dtype = np.float64)

mf = cl.mem_flags
in_buff = cl.Buffer(ctx, mf.READ_WRITE,size=a.nbytes)
out_buff = cl.Buffer(ctx , mf.READ_WRITE, size=out.nbytes)
scratch_buff = cl.Buffer(ctx , mf.READ_WRITE, size=scratch.nbytes)


cl.enqueue_copy(queue,in_buff,np.ascontiguousarray(a))
sum2(queue,global_range,local_range,in_buff,out_buff,np.int32(n),cl.LocalMemory(a.nbytes))
cl.enqueue_copy(queue, out_buff,out)

#global_range_count = (nN,nq,ntheta)
#global_range_sum = (nN,)
#neg_counts = np.zeros(nN,dtype = float)
#I_mock = np.zeros((nq,ntheta,nphi),dtype = float)
#
#mf = cl.mem_flags
#summands_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(summands))
#I_buff = cl.Buffer(ctx , mf.READ_WRITE, size=I_mock.nbytes)
#out_buff = cl.Buffer(ctx , mf.READ_WRITE, size=neg_counts.nbytes)
