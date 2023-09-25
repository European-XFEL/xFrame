import numpy as np
import time
from xframe.settings import general
general.n_control_worker=0
import xframe
xframe.setup_experiment('fxs3046_online','normalized_proc')
from psutil import virtual_memory


db = xframe.database.experiment
#for m in range(16):
module = 3
#f0 = db.load('vds_module',as_h5_object=True,path_modifiers={'run':169,'module':3,'data_mode':'proc'})
f0_2 = db.load('/gpfs/exfel/theory_group/user/berberic/p3046/vds/r0169/vds_proc.h5',as_h5_object=True)
f0 = db.load('/gpfs/exfel/theory_group/user/berberic/p3046/vds/r0169/vds_proc_module_%d.h5'%module,as_h5_object=True)
f0_check = db.load('/gpfs/exfel/exp/SPB/202202/p003046/proc/r0169/CORR-R0169-AGIPD%02d-S00000.h5'%module,as_h5_object=True)

data_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data'
mask_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/mask'
gain_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/gain'
train_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/trainId'
pulse_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/pulseId'
cell_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/cellId'

check_data_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%module
check_mask_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/mask'%module
check_gain_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/gain'%module
check_train_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%module
check_pulse_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/pulseId'%module
check_cell_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%module

pulses = f0[pulse_path][:]
p_any_false = (pulses == 65535).any()
cells = f0[cell_path][:]
c_any_false = (cells == 65535).any()
trains = f0[train_path][:]
t_any_false = (trains == 65535).any()

data2 = f0_2[data_path]
data = f0[data_path]
check_data = f0_check[check_data_path]


#for m in range(16):
#    log.info(i)
#    data = f0[data_path][i::n_parts]
#    any_non = np.isnan(data).any() 
    
#c = next(exp.get_data(opt))

#c = db.load('data_chunk',frame_ids = np.arange(1,10001), **opt)
#for c_id , c in enumerate(exp.get_data(opt)):
#    del(c)
#    log.info('chunk {} done!'.format(c_id))

#start = time.time()
#c = db.load_chunks('chunks',**opt)
#log.info(time.time()-start)
#for i in range(169,170):
#    db.create_vds(i,False,create_modulewise_vds = True, create_complete_vds = False,n_processes = 40)



if False:
    def lload(m,run,N,**kwargs):
        f = db.load('vds_module',skip_custom_methods = True ,as_h5_object=True,path_modifiers={'run':run,'data_mode':'proc','module':m})['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data']
        out = np.zeros((N,)+f.shape[1:],dtype = f.dtype)
        f.read_direct(out,slice(100,N+100))
        return 1
    def load(run,N,**kwargs):
        f = db.load('vds',skip_custom_methods = True ,as_h5_object=True,path_modifiers={'run':run,'data_mode':'proc'})['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data']
        out = np.zeros((16,N)+f.shape[2:],dtype = f.dtype)
        f.read_direct(out,(slice(None),slice(100,N+100)))
        return 1
    
    runs = np.array([112,113,114,115])
    runs = np.arange(4000)
    N = 1000
    start = time.time()
    out = np.zeros((16,N)+f.shape[2:],dtype = f.dtype)
    #load(169,N)
    xframe.Multiprocessing.comm_module.request_mp_evaluation(lload,argArrays=[np.arange(16)],const_args=[169,N],n_processes=16,callWithMultipleArguments=False,assemble_outputs = False)
    stop = time.time()-start
    print(stop)
