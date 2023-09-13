import numpy as np
import time
from xframe.settings import general
general.n_control_worker=0
import xframe
xframe.setup_experiment('fxs3046_online','normalized_proc')
from psutil import virtual_memory

log = xframe.log
db = xframe.database.experiment
opt = {
    'run':169,
    'data_mode': 'proc',
    'frame_range': slice(None),
    'selection': {
        'cells': slice(1,177),
        'pulses': slice(None),
        'trains': slice(0,-100,1)
                  },
    'modules':np.arange(16),
    'n_frames':20000,
    'good_cells':np.arange(1,202),
    'in_multiples_of':False,
    'free_mem':virtual_memory().free,
    'n_processes':False
}

exp = xframe.experiment_worker
#exp.set_ids_for_run(169)

#c = db.create_vds_module(169,False,modules = np.arange(16),n_processes = 10)
#c = db.create_vds(169,False,n_processes = 10)

c = next(exp.get_data(opt))

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
