'''
Module that contains all logic concerning multiprocessing (including GPU access management) and harware info.
'''
import logging
from itertools import repeat
import sys
import os

#sys.stderr = open(os.devnull, "w")
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Process
from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Queue
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
#sys.stderr = sys.__stderr__
from xframe.Multiprocessing_interfaces import OpenClDependency,PsutilDependency
from xframe.library.pythonLibrary import xprint
import traceback
import numpy as np
from xframe import settings

sa = False #will be dependency injected with SharedArray module
MPI = False #will be dependency injected with SharedArray module
mpi_comm = None #will be set by create_mpi_comm #None if not jet checked False if mpi not available otherwise MPI.COMM_WORLD
comm_module = 'Will be set by Controller.'
openCL_plugin = OpenClDependency
psutil_plugin = PsutilDependency
log = logging.getLogger('root')
master_name = 0

def create_mpi_comm():
    global mpi_comm
    if not isinstance(MPI,bool):
        mpi_comm = MPI.COMM_WORLD # global communicator
    
def get_global_number_of_processes():
    if mpi_comm == None:
        create_mpi_comm()
        
    if isinstance(mpi_comm,bool):        
        global_number_of_processes = 1
    else:
        global_number_of_processes=mpi_comm.Get_size()
        
        #comm=commWorld.Split_type(MPI.COMM_TYPE_SHARED) # Inter node communicator
        #globalRank=commWorld.Get_rank()
        #nodeRank=comm.Get_rank()    
        #nodeNumberOfThreads=comm.Get_size()
        
        #MPIPoolExecutor =  mpi4py.futures.MPIPoolExecutor
        #mpi_executor=MPIPoolExecutor()        
    return global_number_of_processes

def get_local_cpu_count():
    #cpu_count gives number of cpu_threads. To get rid of hyperthreading divide py 2 
    return int(cpu_count()//2)
    
def load_openCL_dict():
    if not isinstance(openCL_plugin,str):
        process_identity = multiprocessing.current_process()._identity
        if len(process_identity)<1:
            process_number = 0
        else:            
            process_number = process_identity[-1]
        #log.info("process number = {}".format(process_number))
        #log.info("process = {}".format(multiprocessing.current_process()))
        #log.info("process identity = {}".format(multiprocessing.current_process()._identity))
        state = openCL_plugin.load_openCL_dict(process_number = process_number)
    else:
        e = AssertionError('openCL plugin has not been injected.')
        raise e
    return state
        
def get_Event():
    return multiprocessing.Event()
def get_Queue(using_manager = False):
    #manager = multiprocessing.Manager()
    if using_manager:
        manager = multiprocessing.Manager()
        return manager.Queue()
    else:
        return multiprocessing.Queue()

def get_process_name():
    current_process = multiprocessing.current_process()
    if len(current_process._identity)<1:
        process_id = master_name
    else:            
        process_id = int(current_process.name)
    return process_id
def get_n_child_processes():
    non_sync_children = tuple(child for child in multiprocessing.active_children() if not 'SyncManager-' in str(child.name))
    return len(non_sync_children)

def get_free_memory():
    return psutil_plugin.get_free_memory()

class MPProcess:
    def __init__(self,process=False,queue=False,events={}):
        if isinstance(process,multiprocessing.Process):
            self.process = process
            self.name = process.name
            self.is_daemon = process.daemon
            self.start = process.start
            self.join = process.join
        else:
            #assume master process is meant
            self.process = 'master'
            self.name = 0
            self.is_daemon = False
            self.start = False
            self.join = False
        self.queue = queue
        self.events = events # dict of Events 

daemon_termination_queue = get_Queue()
master_process = MPProcess(queue = daemon_termination_queue,events={'gpu':get_Event()})

gpu_queues = {i:get_Queue() for i in range(settings.general.max_parallel_processes+1)}
gpu_events = {i:get_Event() for i in range(settings.general.max_parallel_processes+1)}
# Dict of MPProcess objects for worker processes
workers = {}
# Dict of MPProcess objects for daemon processes
daemons = {}

# Active_processes acts as a counter everytime new child processes are started they know how many mp processes where already started when they are called.
free_cpus = get_local_cpu_count()-get_n_child_processes()-1
free_threads = int( 2*get_local_cpu_count()-get_n_child_processes() )-1
default_n_processes = free_cpus

def create_process_names(n,is_daemon=False):
    '''
    Creates process names. There can only be one set of N processes, that are not daemons, active at the same time so give them the names (1,...,N).
    Daemons get unique consequtive negative numbers, the names of killed daemons will not be filled again.
    The name of the master process is by default 0.
    :param n: number of processes to name.
    :param daemon: Flag that specifies whether daemon or worker names should be created.
    :return: tuple of names (positive ingegers for workers, negative integers for daemons)
    '''
    if not is_daemon:
        # If not daemon ruturn (1,...,n)
        names = tuple(range(1,n+1))
    else:
        # If daemon ruturn (lowest_existing_daemon_name-1,...,lowest_existing_daemon-n)
        lowest_existing_name = 0
        for d in daemons.values():
            #log.info('daemon name = {}'.format(d.name))
            if int(d.name)<lowest_existing_name:
                lowest_existing_name = d.name
        names = tuple(range(lowest_existing_name-1,lowest_existing_name-n-1,-1))
    return names

def get_free_cpus():
    return get_local_cpu_count()-get_n_child_processes()-1
def update_free_cpus():
    global free_cpus
    global free_threads
    global default_n_processes    
    free_cpus = get_free_cpus()
    free_threads = int(2*get_local_cpu_count()-get_n_child_processes())-1
    default_n_processes = free_cpus
def check_available_cpus_before_spawning_childs(n_childs,are_daemons=False):
    update_free_cpus()
    would_use_hyperthreads = (free_cpus - n_childs<0)
    would_use_more_processes_than_available_hyper_threads =  (free_threads - n_childs<0)
    if would_use_hyperthreads:
        if are_daemons:
            log.warning('Total number of daemon processes exceedes CPU core count. Some of the spawned daemons will use hyperthreading threads!')
        else:
            log.info('Total number of processes exceedes CPU core count. Some of the spawned processes will use hyperthreading threads. This may slow down performance.')
    elif would_use_more_processes_than_available_hyper_threads:
        if are_daemons:
            log.warning('Total number of daemon processes exceedes CPU thread count including hyperthreads! This will slow down performance!')
        else:
            log.warning('Total number of processes exceedes CPU thread count including hyperthreads. This will slow down performance.')


def _register_processes(processes):
    global daemons
    global workers
    for process in processes:
        if process.is_daemon:
            daemons[process.name]=process
        else:
            workers[process.name]=process
            
def _deregister_processes(processes,exclude_daemons=True):
    global daemons
    global workers
    for process in processes:
        try:
            if process.is_daemon:
                if not exclude_daemons:                
                    daemons.pop(process.name)
            else:
                workers.pop(process.name)
        except KeyError as e :
            log.warning('Process to be deleted with name {} does not exist. Ignoring process deregister event.'.format(name_to_delete))
            
def _process_daemon_terminations():
    while daemon_termination_queue.qsize() != 0:
        name_to_delete = daemon_termination_queue.get()
        try:
            log.info(f"Daemon {name_to_delete} has terminated removing it from daemon list.")
            _deregister_processes([daemons[name_to_delete]],exclude_daemons=False)
        except KeyError:
            log.warning('Daemon to be deleted with name {} does not exist. Ignoring daemon termination event.'.format(name_to_delete))
def get_mp_process_by_name(global_name):
    if global_name<0:
        return daemons[global_name]
    elif global_name>0:
        return workers[global_name]
    else:
        return master_process

def check_if_daemon(global_name:int):
    return global_name<0

def get_process_queue(process_name):
    #log.info('request queue for process {} of {}'.format(process_id,len(queue_per_cpu)))
    process_name = int(process_name)
    try:
        if process_name>0:
            queue = workers[process_name].queue
        elif process_name<0:
            queue = daemons[process_name].queue
        else:
            queue = master_process.queue
    except Exception as e:
        #e = AssertionError('Process {} does not have a queue.'.format(process_name))
        raise e
    return queue
def get_process_event(process_name,event_name=False):
    process_name = int(process_name)    
    try:
        if process_name<0:
            event = daemons[process_name].events[event_name]
        elif process_name>0:
            event = workers[process_name].events[event_name]
        else:
            event = master_process.events[event_name]                            
    except Exception:
        e = AssertionError('process with name {} does not exist or does not have an event called {}'.format(process_name,event_name))
        raise e
    return event
def put_to_process_queue(process_name,message):
    get_process_queue(process_name).put(message)
def get_from_process_queue(process_name,timeout = None):
    return get_process_queue(process_name).get(timeout = timeout)

def get_gpu_queue(process_name):
    #log.info('request queue for process {} of {}'.format(process_id,len(queue_per_cpu)))
    process_name = int(process_name)
    try:
        queue = gpu_queues[process_name]
    except Exception as e:
        raise e
    return queue
def get_gpu_event(process_name):
    process_name = int(process_name)    
    try:
        event = gpu_events[process_name]
    except Exception as e:
        raise e
    return event
    
def put_to_gpu_queue(process_name,message):
    get_gpu_queue(process_name).put(message)
def get_from_gpu_queue(process_name,timeout = None):
    return get_gpu_queue(process_name).get(timeout = timeout)

            

#################################
## Shared Memory (SharedArray) ##

def delete_shared_arrays():
    global sa    
    shared_arrays = sa.list()
    for array in shared_arrays:
        sa.delete(array.name.decode('utf-8'))        

def create_shared_arrays(shapes,dtypes):
    global sa
    names = []
    shared_outputs = []
    #log.info('all output dtypes in Shared array = {}'.format(out_dtypes))
    for i,shape,dtype in zip(np.arange(len(shapes)),shapes,dtypes):
        # The probability of identical hashes from hash(np.random.rand()) should be low enough to not care about it :p 
        name = 'shared_output_'+ str(hash(np.random.rand()))
        names.append(name)
        #log.info('output dtypes in Shared array = {}'.format(out_dtype))
        shared_outputs.append(sa.create('shm://'+name,int(np.prod(shape)),dtype).reshape(shape))
    return shared_outputs, names 


#############################
##  Multiprocessing modes  ##

additional_argument_names = ['number_of_processes','local_name','global_name','events','queue','available_memory','synchronize']


class MPFunctionArgumentsStandard:
    def __init__(self, function, arguments, argument_indices, constant_args = [],call_with_multiple_arguments = False, split_together = False, additional_kwargs = {}):
        self.function = function
        self.arguments = arguments
        self.argument_indices = argument_indices
        self.split_together = split_together
        self.constant_args = constant_args
        self.call_with_multiple_arguments = call_with_multiple_arguments
        self.additional_kwargs = additional_kwargs
        for kw in additional_argument_names:
            assert kw in additional_kwargs, 'expected argument {} not provided.'.format(kw)
        
class MPMode:
    name = 'none'
    def __init__(self):
        self.pre_processed = False
    def pre_processing(self,n_processes):
        self.pre_processed = True
    def get_mp_function(self,mp_arguments:MPFunctionArgumentsStandard):
        assert self.pre_processed , 'call to get_mp_function before mode attributes where processed via pre_process'
    def collect_outputs(self,result,split_dict):
        return result
    def exception_decorator(self,func,additional_kwargs):
        def outer(*args,**kwargs):
            try:
                func(*args,**kwargs)
                # if daemon request deletion from daeomes dict in master process
                global_name = additional_kwargs['global_name']
                if check_if_daemon(global_name):
                    master_process.queue.put(global_name)
                    #log.info('Request deletion of exiting daemon with global_name {}'.format(global_name))                
            except Exception as e:
                traceback.print_exc()
                log.error(e)
                self.exception_handler(e,**additional_kwargs)
            #log.info('calculation done in process {}'.format(additional_kwargs['global_name']))
        return outer
    def exception_handler(self,exception,**kwargs):
        global_name = kwargs['global_name']
        if check_if_daemon(global_name):
            global_name = kwargs['global_name']
            master_process.queue.put(global_name)
            #log.info('Request deletion of failed daemon with global_name {}'.format(global_name))
        else:
            raise exception
                      
class MPMode_Queue(MPMode):
    name = 'Queue'
    def __init__(self,assemble_outputs=True,ignore_first_dimension=False):
        super().__init__()
        self.assemble_outputs = assemble_outputs
        self.queue = get_Queue(using_manager = True)
        self.ignore_first_dimension = ignore_first_dimension
    def get_mp_function(self,mp_arguments:MPFunctionArgumentsStandard):
        assert self.pre_processed , 'call to get_mp_function before mode attributes where processed via pre_process'      
        # handle edge case of a no argument
            
        queue = self.queue
        additional_kwargs = mp_arguments.additional_kwargs
        local_name = additional_kwargs['local_name']
        function = mp_arguments.function
        args = mp_arguments.arguments                
        c_args = mp_arguments.constant_args
        if mp_arguments.call_with_multiple_arguments:        
            def mp_function():
                #log.info(additional_kwargs.keys())
                result=function(*args,*c_args,**additional_kwargs)
                #            log.info('result type={}'.format(type(result)))
                if not isinstance(result,np.ndarray):
                    result=[result]
                    
                result_dict={local_name:result}
                #           log.debug('processID type={}'.format(type(processID)))
                queue.put(result_dict,False)
                
        else:
            def mp_function():
                #log.info('arguments = {}'.format(args))
                #log.info(tuple(zip(*args)))
                if len(args)==0:
                    result=[function(*c_args,**additional_kwargs)]
                else:
                    #log.info(args)
                    n_args=len(args[0])
                    #log.info( list( [[arg[i] for arg in args],c_args,additional_kwargs] for i in range(n_args)))
                    result=list(function(*[arg[i] for arg in args],*c_args,**additional_kwargs) for i in range(n_args))
                #log.info("len result = {}".format(len(result)))
                #combine individual results if possible
                if len(result)>0:
                    all_results_are_numpy_arrays = np.array([isinstance(r,np.ndarray) for r in result]).all()
                    if all_results_are_numpy_arrays:
                        all_results_have_same_shape = np.array([r.shape==result[0].shape for r in result]).all()
                        #log.info('result shapes = {}'.format(np.array([r.shape for r in result])))
                        if all_results_have_same_shape:
                            result = np.stack(result,axis = 0)
                            #log.info('result shape = {}'.format(result.shape))
                #log.info(args)
                #log.info(result[0])
                #if (len(args)<=1) and (len(result[0])<=1):
                #    pass# result = result[0]
                result_dict={local_name:result}
                queue.put(result_dict,False)
                #log.info('put to queue in process {}'.format(additional_kwargs['global_name']))
                
        return self.exception_decorator(mp_function,additional_kwargs)
        
    def collect_outputs(self,split_dict):
        out_dict={}
        queue = self.queue
        while queue.qsize() != 0:
            out_dict.update(queue.get())
        #log.info('assemble outputs = {}'.format(self.assemble_outputs))
        if self.assemble_outputs:
            return self.assemble_results(out_dict,split_dict)
        else:
            #log.info('out_dict keys = {}'.format(out_dict.keys()))
            #log.info('split_dict = {}'.format(split_dict))
            #log.info('shape part = {}'.format(split_dict['shape_part']))
            one_output_per_process = np.prod(split_dict['shape_part'])<=len(split_dict['indices'])
            #log.info('one_ouput_per_process = {}'.format(one_output_per_process))
            #log.info('shape part = {},{}'.format(split_dict['shape_part'],len(split_dict['indices'])))
            if one_output_per_process:
                out_dict = {key:val[0] for key,val in out_dict.items()}
            return out_dict
    def assemble_results(self,result_dict,split_dict):
        '''
        Uses the splitted_indices from _split_arguments and tries to assemble the multiprocessing outputs into an numpy array or list depending on the return type of the first element in the results dict.    
        '''
        #log.info('type of result before assembly = {}'.format(type(result_dict)))
        #log.info('start assembling output')
        splitted_indices = split_dict['indices']
        result_shape_part = split_dict['shape_part']
        split_mode = split_dict['mode']
        #result_shape_end=(max(len(argArrays),1),)
        if len(result_dict)<=0:
            return result_dict
        
        first_result=next(iter(result_dict.values()))
        result_type = type(first_result)        
        
        #log.info('type of result = {}'.format(result_type))
        #log.info(tuple(i.shape for i in next(iter(result_dict.values()))))
        if result_type == np.ndarray:
            if len(first_result.shape)>1 or self.ignore_first_dimension:
                n=1
            else:
                n=0
            first_shape=first_result.shape[n:]
            #log.info('shape part ={} result[0] shape ={}, first shape = {}'.format(result_shape_part,first_result.shape,first_shape))
            shapes_equal = False
            if np.array(tuple(isinstance(r,np.ndarray) for r in result_dict.values())).all():
                shapes_equal = np.array([np.array(r.shape[n:])==np.array(first_shape) for r in result_dict.values()]).all()
            
            if shapes_equal:
                #log.info('shapes are equal !')
                result_shape=result_shape_part+first_shape
                result_dtype=first_result.dtype        
                result=np.empty(result_shape,dtype=result_dtype)
                #log.info(f'result shape = {result.shape} result.dtype = {result.dtype}')
                for process_id in result_dict.keys():
                    if not split_dict['split_together']:
                        result_part_tuple=tuple(indice_part.astype(int) for indice_part in splitted_indices[process_id] )
                    else:
                        result_part_tuple=(splitted_indices[process_id][0],)
                    #log.info('result part tuple = {}'.format(result_part_tuple))
                    result[result_part_tuple]=np.array(result_dict[process_id])
            else:
                #log.info('shapes are NOT equal !')
                result = self._assemble_results_as_lists(result_dict,split_dict)
                
        elif (result_type == tuple) or (result_type==list):
            result = self._assemble_results_as_lists(result_dict,split_dict)
        else:
            result = np.array([])
        #log.info(f'result shape = {result.shape} result.dtype = {result.dtype}')
        return result

    def _assemble_results_as_lists(self,result_dict,split_dict):
        '''
        Assembles results as list
        '''
        #log.info('type of result before assembly = {}'.format(type(result_dict)))
        split_mode = split_dict['mode']
        if len(result_dict)<=0:
            return result_dict
                
        first_result=next(iter(result_dict.values()))
        result_type = type(first_result)
        
        n_processes = len(result_dict)
        if split_mode=='sequential':
            result = []
            for process_id in range(n_processes):
                try:
                    result+=list(result_dict[process_id])
                except KeyError as e:
                    log.error('Process {} seems to have failed. Skipping.'.format(process_id))
        elif split_mode=='modulus':
            parts = len(result_dict[0])
            result_parts = [[] for n in range(parts)]
            
            for process_id in range(n_processes):
                try:
                    for part,output in enumerate(result_dict[process_id]):
                        result_parts[part].append(output)
                except KeyError as e:
                    log.error('Process {} seems to have failed. Skipping.'.format(process_id))
            result = [item for sub_list in result_parts for item in sub_list]
        else:
            raise AssertionError('split_mode = {} not known.'.format(split_mode))
        return result
    
class MPMode_SharedArray(MPMode):
    name = 'SharedArray'
    def __init__(self,output_shapes,output_dtypes,reduce_arguments=False):
        super().__init__()
        self._initial_output_shapes = output_shapes
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes
        self.reduce_arguments = reduce_arguments
        
        ## The following attributes will be set by pre_processing
        self.shared_outputs = False
        self.output_names = False
    def pre_processing(self,n_processes):
        ## delete any existing and create new shared memory objects
        delete_shared_arrays()
        if self.reduce_arguments:
            new_shapes = [(n_processes,)+shape for shape in self._initial_output_shapes]
            self.output_shapes = new_shapes
                
        self.shared_outputs,self.output_names = create_shared_arrays(self.output_shapes,self.output_dtypes)
        self.pre_processed = True
    def _generate_output_indices(self,mp_arguments):
        local_name = mp_arguments.additional_kwargs['local_name']

        output_indices=tuple(indice_part.astype(int) for indice_part in mp_arguments.argument_indices)
            
        if mp_arguments.split_together:
            output_indices = output_indices[0]
        if self.reduce_arguments:
            output_indices = slice(local_name,local_name+1)
        return output_indices
        
    def get_mp_function(self,mp_arguments:MPFunctionArgumentsStandard):
        assert self.pre_processed , 'call to get_mp_function before mode attributes where processed via pre_process'
        #repeat constant arguments
        repeated_arguments=[repeat(arg) for arg in mp_arguments.constant_args]

        # handle edge case of a no argument
        args = mp_arguments.arguments
        if len(args)==0:
            args=[[None]]
            
        additional_kwargs = mp_arguments.additional_kwargs
        local_name = additional_kwargs['local_name']
        output_names = self.output_names
        output_shapes = self.output_shapes
        output_indices = self._generate_output_indices(mp_arguments)

        args = mp_arguments.arguments
        c_args = mp_arguments.constant_args
        function = mp_arguments.function
        
        if mp_arguments.call_with_multiple_arguments:        
            def mp_function():            
                output_files = [sa.attach('shm://'+ o_name).reshape(o_shape) for o_name,o_shape in zip(output_names,output_shapes)]
                additional_kwargs['outputs'] = output_files
                additional_kwargs['output_ids'] = output_indices
                # in this function output have to be placed in outputs[i][output_ids]
                function(*args,*c_args,**additional_kwargs)            
        else:
            def mp_function():
                output_files = [sa.attach('shm://'+o_name).reshape(o_shape) for o_name,o_shape in zip(output_names,output_shapes)]
                additional_kwargs['outputs'] = output_files
                for argument,out_ids in zip(zip(*args),zip(*output_indices)):
                    additional_kwargs['output_ids'] = out_ids
                    #In this function output have to be placed in outputs[i][output_ids]
                    function(*argument,*c_args,**additional_kwargs)
        return self.exception_decorator(mp_function,additional_kwargs)            

    def collect_outputs(self,split_dict):
        # The following only affects the ability to share the arrays it does not cause them to be physically deleted untill the last reference vanishes.
        delete_shared_arrays()
        return self.shared_outputs
    
    
def _run_processes(process_list):
    for p in process_list:
        p.start()
        
def _join_processes(process_list):
    for p in process_list:
        if not p.is_daemon:
            p.join()
            #log.info('process {} joined'.format(p.name))

def split_mp_arguments(argArrays,nProcesses,mode='sequential',split_together=False):
    """
    Routine that splits the list of argArrays into as nProcesses maky pieces.
    mode can either be 'sequential' or 'modulus'. In sequential mode an input array is split into sequential pieces,
    e.g. argArrays =[[1,2,3,4,5]] would be split for nProcesses=2 into [1,2],[3,4,5] (note that the last pice can be bigger by nProcesses-1 elements than the other parts).
    In 'modulus' the arguments are split in to equivalence classes of $\mathbb{Z}/nProcesses \mathbb{Z}$,
    e.g: argArrays =[[1,2,3,4,5]] would be split for nProcesses=2 into [1,3,5],[2,4] (in this case all part sizes can differ at most by 1).
    
    If more than 1 array is given as input by default all possible parameter combinations are considered and splitted into pices,
    e.g.: argArrays =[[1,2],[2,3]] is considered as [(1,2),(1,3),(2,2),(2,3)] which is then split into parts.
    If however split_together = True all input arrayys have to be of the same length and are split together,
    e.g.: argArrays =[[1,2],[2,3]] is considered as [(1,2),(2,3)] and then split into parts.
    
    This method not only returns the splitted arguments but also their corresbonding indices as part of an (len_argument_array_1,...,len_argument_array_N) shaped  array,
    which in case of splitTogether= true reduces to (len_argument_array_1,) rhaped array.
    """
    
    #log.info('argArrays={}'.format(argArrays))
    if isinstance(argArrays,np.ndarray):
    #    log.info('yep')
        argArrays=[argArrays]
    indiceArrays=[np.arange(len(array),dtype=int) for array in argArrays]
    arg_legths_equal= len(set(tuple(len(array) for array in argArrays))) == 1
    dtype_list = [arg.dtype for arg in argArrays]
    n_args = len(argArrays)
    #log.info('len argArrays ={}'.format(len(argArrays)))
    if split_together:
        try:
            assert arg_legths_equal,'split_together requires all argArrays to have the same length but the length are {}'.format(tuple(len(array) for array in argArrays)) 
        except AssertionError as e:
            log.error(e)
            raise
        arguments=tuple(argArrays)
        argIndices=indiceArrays
       
    else:
        #    log.info('indiceArrays={}'.format(indiceArrays))
        argIndices=tuple(reversed(tuple(dataset.flatten().astype(np.int_) for dataset in np.meshgrid(*reversed(indiceArrays)) )))
        arguments=tuple(argument[indices] for argument,indices in zip(argArrays,argIndices))
    nArguments=len(arguments[0])
    #print('arguments = {}'.format(argArrays))    
    #log.info('nArgs = {} nProcess {}'.format(nArguments,nProcesses))
    if nArguments<nProcesses:
        nProcesses=nArguments
    argPerProcess=nArguments//nProcesses    

    #    log.info('nArguments={}, nProcesses={}'.format(nArguments,nProcesses))
    #print(mode)

    if mode=='sequential':    
        splitPoints=[int(argPerProcess*process) for process in range(1,nProcesses)]
        #splittedArguments=tuple(np.split(argument,splitPoints) for argument in arguments)
        splittedArguments=tuple(np.array_split(argument,nProcesses) for argument in arguments)
        #splittedIndices=tuple(np.split(argIndice,splitPoints) for argIndice in argIndices)
        splittedIndices=tuple(np.array_split(argIndice,nProcesses) for argIndice in argIndices)
    elif mode=='modulus':
        def splitArgument(argument):
            dtype=argument.dtype
            def returnSlizedArgument(processID):
                arg_slice = np.array(argument[processID::nProcesses],dtype=dtype)
                #log.info('slice shape = {}'.format(arg_slice.shape))
                return arg_slice
            splitted_argument = tuple(map(returnSlizedArgument,range(nProcesses)))
            return splitted_argument
        
        splittedArguments=tuple(splitArgument(argument) for argument in arguments)
        splittedIndices=tuple(splitArgument(argIndice) for argIndice in argIndices)
        #log.info('dtype={} splitted Indices=\n {}'.format(splittedIndices.dtype,splittedIndices))
    else:
        raise AssertionError('Split mode "{}" not known. Known modes are "sequential" or "modulus".'.format(mode))
    arguments_per_process = list(zip(*splittedArguments))
    indices_per_process = list(zip(*splittedIndices))    
    return {'arguments':arguments_per_process,'indices':indices_per_process}

def _create_synchronization_routines(process_ids):
    '''
    Generates a synchronization routine to be used in shared memory multiprocessing calls.
    It allows for all participating processes to synchronize i.e. wait untill all have finished a given task.
    Implementateion details:
    Each process has a pair of events which function as a toggle between states (0,1) and (1,0)
    A synchronization is equivalent to wait until all processes have switched their state once.
    
    If a process fails to synchronice once (time out is reached), it will be excluded from all further synchronization attempts, which means:
    1. If it tries to synchronice an AssertionError is thrown.
    2. All other Processes ignore its state in further synchronization attempts.
    :param process_ids: names of the processes for which to create synchronization functions
    :param type: list of different hashable elements
    :return: Dictionary of synchronization functions whose keys are the corresponding process_ids. A call to any function blocks untill timout is reached or all other functions are called.
    :return type: dict
    '''
    #create events
    process_events = {p:[get_Event(),get_Event()] for p in process_ids}
    # initialize events to be in state (0,1)
    for events in process_events.values():
        events[1].set()

    def process_id_decorator(local_process_id):        
        def inner(func):
            def wrapper(timeout=30):
                return func(local_process_id,timeout=timeout)
            return wrapper
        return inner

    routines = {}
    for p in process_ids:
        @process_id_decorator(p)
        def synchronize(local_process_id,timeout=30):
            #log.info('timeout = {}'.format(timeout))
            # get local id of zero state event (is either 0 or 1)
            zero_id = int(process_events[local_process_id][0].is_set())
            one_id = (zero_id+1)%2
            local_event = process_events[local_process_id]
            has_failed = local_event[zero_id].is_set() == local_event[one_id].is_set()

            # Check wether local process did fail to synchronice in the past (i.e is in state 00 or 11)
            if has_failed:
                error = AssertionError('Synchronization Error: This process {} has failed to synchronice in the past and is thus prevented from further execution.')
                raise(error)
            else:            
                # Set flag that local process awaits toggle switch.
                process_events[local_process_id][zero_id].set()
            
            
            # wait until all other workers signal that they await toggle switch
            for p_id,event in process_events.items():
                event_in_failed_state = ( event[0].is_set()==event[1].is_set() )
                if (p_id == local_process_id) or event_in_failed_state:
                    # don't wait for myself or previousely failed processes 
                    continue
                success = event[zero_id].wait(timeout = timeout)
                if not success:
                    log.warning('Process {} didnt synchronice after {} seconds stop waiting in process {}'.format(p_id,timeout,local_process_id))
                    # set failed process in failed mode:
                    event[zero_id].clear()
                    event[one_id].clear()
                    
            # finalize local toggle switch
            process_events[local_process_id][one_id].clear()
            #log.info('sync_done process {}'.format(local_process_id))
        routines[p]=synchronize
    return routines
                
def _read_number_of_processes(n_processes):
    '''
    If number of processes is of the wrong type this method sets it to the default number.
    If number of processes is bigger than the default default_n_processes it is set to the default.
    The default is get_local_cpu_count()-active_processes.
    '''
    #log.info(f' input n processes = {n_processes}')
    update_free_cpus()
    global free_cpus    
    if isinstance(n_processes,bool) or (not isinstance(n_processes,int)):
        n_processes = free_cpus
    else:
        n_processes = n_processes
    if n_processes > free_cpus:
        n_processes = free_cpus
    n_processes = max(1,n_processes)
    return n_processes

def _split_arguments(arg_arrays,n_processes,mode,split_together):
    '''
    Wrapper around split_mp_arguments, that deals with the case of no argument_arrays beeing provided.
    '''        
    if len(arg_arrays)!=0:
        split_dict = split_mp_arguments(arg_arrays,n_processes,mode=mode,split_together=split_together)
        splitted_args = split_dict['arguments']
        splitted_indices= split_dict['indices']
        #            log.info('splittedIndices 1=/n{}'.format(tuple(splittedIndices)))
        #           log.info('splitted Args shape={}'.format(splittedArgs.shape))
        n_processes=len(splitted_args)
        if split_together:
            result_shape_part=(len(arg_arrays[0]),)
        else:
            result_shape_part=tuple([len(array) for array in arg_arrays])
    else:
        splitted_args=[[]]*n_processes
        splitted_indices=np.expand_dims(np.arange(n_processes),1)
        result_shape_part=(n_processes,)
    return {'arguments':splitted_args,'indices':splitted_indices,'shape_part':result_shape_part,'mode':mode,'split_together':split_together}

def _calc_available_mem_per_process(n_new_processes):
    assert get_process_name() == 0, 'available mem can only be calculated from master process.'
    # create multiprocessing wrapers 
    free_mem=get_free_memory()# virtual_memory()
    available_memory_per_process=free_mem/(n_new_processes + get_n_child_processes())
    return available_memory_per_process

def process_mp_request(function,
                       mode:MPMode = MPMode_Queue(),
                       is_daemon=False,
                       return_process_objects = False,
                       input_arrays=[],
                       const_inputs=[],
                       call_with_multiple_arguments=False,
                       split_mode='sequential',
                       split_together=False,
                       n_processes = False,
                       event_names=[],
                       create_queues = False):
    '''
    '''
    process_name = get_process_name()
    #log.info("mode = {}".format(function))
    assert process_name == master_name , 'Multi processing requests can only be sent from the master process whose name is {}. This request has been sent by process {}. Abborting MP evaluation.'.format(master_name,process_name)
    assert isinstance(mode,MPMode), 'Multiprocessing mode {} is not known. Available modes are {}'.format(mode,tuple(MPMode_Queue,MPMode_SharedArray))

    ## look for daemons that terminated
    _process_daemon_terminations()
    
    ## split input arguments 
    n_processes = _read_number_of_processes(n_processes)
    #log.info('split arguments')
    split_dict = _split_arguments(input_arrays,n_processes,split_mode,split_together)
    n_processes = len(split_dict['arguments'])    
    log.info(f'Starting {n_processes} process{"es"*(n_processes>1)}.')
    log.debug(f'On function: {function}')

    ## Throw warnings if active processes will use HyperThreads
    check_available_cpus_before_spawning_childs(n_processes,are_daemons=is_daemon)


    mode.pre_processing(n_processes)
    
    ## Build Processes
    process_names = create_process_names(n_processes,is_daemon=is_daemon)
    #log.info('New process names = {}'.format(process_names))
    sync_functions = _create_synchronization_routines(process_names)
    available_mem = _calc_available_mem_per_process(n_processes)
    mp_processes = []
 
    for local_name,global_name in enumerate(process_names):
        if create_queues:
            #log.info('creating queue for process {}'.format(global_name))
            queue = get_Queue()
        else:
            queue = False        
        events = {e_name:get_Event() for e_name in event_names}
        additional_kwargs = {}
        additional_kwargs['number_of_processes'] = n_processes
        additional_kwargs['local_name'] = local_name
        additional_kwargs['global_name'] = global_name
        additional_kwargs['events'] = events
        additional_kwargs['queue'] = queue
        additional_kwargs['available_memory'] = available_mem
        additional_kwargs['synchronize'] = sync_functions[global_name]

        arguments = split_dict['arguments'][local_name]
        argument_indices = split_dict['indices'][local_name]
        mp_args = MPFunctionArgumentsStandard(function, arguments, argument_indices, constant_args = const_inputs, call_with_multiple_arguments=call_with_multiple_arguments, split_together=split_together, additional_kwargs=additional_kwargs)
        mp_function  = mode.get_mp_function(mp_args)
        
        if is_daemon:
            log.info(f"Started global daemon with name {global_name}")
            process=Process(target = mp_function, name = global_name, daemon = True)
        else:
            process=Process(target = mp_function, name = global_name)
        mp_process=MPProcess(process, queue = queue, events = events)
        
        mp_processes.append(mp_process)    
    ## update_process_dicts
    _register_processes(mp_processes)
    
    ## Run and join Processes
    _run_processes(mp_processes)
    _join_processes(mp_processes)
    
    outputs = mode.collect_outputs(split_dict)

    _deregister_processes(mp_processes)
    if not is_daemon:
        #log.info('All processes done!'.format(n_processes))
        pass
    if return_process_objects:
        return outputs,mp_processes
    else:
        return outputs
    

#################
##  GPU Access ##
def get_number_of_gpus():
    ###### IMPORTANT ######
    # Due to pyOpenCL not beeing fork save one needs to querry the number of available GPUs in a child process. To avoid future opencl imports in child processes to fail.
    def callback(**kwargs):
        return openCL_plugin.get_number_of_gpus()
    n_gpus = process_mp_request(callback,n_processes =1)[0]
    return n_gpus
def create_openCL_context():
    try:
        openCL_plugin.create_context()
    except Exception as e:
        log.info(f'Could not create openCL context in process {get_process_name()}')
        raise e



def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
        

class SharedMemoryObject:
    def __init__(self,name,shape,dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = np.dtype(self.dtype).itemsize * np.prod(self.shape)
        self.shm = False
        self.array = False
        self.created = False
    def release(self):
        #log.info('Trying to delete shared memory object {} P {}'.format(self.name,get_process_name()))
        if isinstance(self.shm,SharedMemory):
#            log.info('trying to delete shared memory object {} P {}'.format(self.name,get_process_name()))
            try:
                self.shm.close()
                #xprint('Shared memory object {} is beeing closed P {}'.format(self.name,get_process_name()))
                if hasattr(self, 'array'):
                    del(self.array)
                if self.created:
                    ## IMPORTANT COMMENT ##
                    #xprint('Shared memory object {} is beeing unlinked P {}'.format(self.name,get_process_name()))
                    self.shm.unlink() #causes a lot of warnings see bugs.python.org/issue38119
                    # unlink is implemented as: 
                    #def unlink(self):
                    #    """Requests that the underlying shared memory block be destroyed.
                    #    In order to ensure proper cleanup of resources, unlink should be
                    #    called once (and only once) across all processes which have access
                    #    to the shared memory block."""
                    #    if _USE_POSIX and self._name:
                    #        _posixshmem.shm_unlink(self._name)
                    #        resource_tracker.unregister(self._name, "shared_memory")
                    # See https://github.com/python/cpython/blob/3.11/Lib/multiprocessing/shared_memory.py
                    # calling both shm_unlink and unregister or just shm_unlink causes a lot of resouce_tracker warnings so untill the bug is fixed
                    # we can only manually unregister the shared memory without calling shm_unlink that gets rid of most of the warnings.
                    # I somehow was not able to suppress the warnings using warnings.simplefilter only by using the python interpreter option -W
                    # which unfortunatelly is not an option in setuptools entry_points.
                    # Well maybe there is an other way that I didn't find.
                    #resource_tracker.unregister(self.shm._name,'shared_memory')                    
                    self.created = False
                    self.shm = False
            except FileNotFoundError as e:
                log.error(e)
                #pass
                #traceback.print_exc()
                
    def construct(self):
        #log.info('trying to create memory object shared {} in P {}'.format(self.name,get_process_name()))
        if not isinstance(self.shm,SharedMemory):
            try:
                self.shm=SharedMemory(self.name,create=True,size=self.nbytes)
            except FileExistsError as e :
                log.warning('left over shared memory file {}. Trying to overwrite. on process {}'.format(self.name,get_process_name()))
                shm=SharedMemory(self.name)
                shm.close()
                shm.unlink()
                self.shm=SharedMemory(self.name,create=True,size=self.nbytes)
                #log.info('create shared memory object {} of size {}'.format(self.name,self.nbytes))
            #resource_tracker.unregister(self.shm._name,'shared_memory') ## prevents some Warnings due to bugs.python.org/issue38119. However it causes a FileNotFoundError when trying to unlik the shared array ...
            self.created = True
            #log.info('create shared memory object {} of size {}'.format(self.name,self.nbytes))
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        return self
    def bind(self,unregister=False):
        #log.info('trying to bind to shared memory object {}'.format(self.name))
        if not isinstance(self.shm,SharedMemory):
            if unregister:
                remove_shm_from_resource_tracker()                
            self.shm=SharedMemory(self.name)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        return self

class GpuProcessHandle:
    def __init__(self,gpu_process,python_process_id):
        self.gpu_process = gpu_process
        self.python_process_id = python_process_id
        self.inputs,self.outputs = self.create_IO_instances()        
    def create_IO_instances(self):
        gpu_process = self.gpu_process
        inputs = []
        outputs = []
        for i in range(gpu_process.n_inputs):
            in_name = self.generate_shared_memeory_input_name(i,self.python_process_id)
            in_shape = gpu_process.input_shapes[i]
            in_dtype = gpu_process.input_dtypes[i]
            inputs.append(SharedMemoryObject(in_name,in_shape,in_dtype))
        for i in range(gpu_process.n_outputs):
            out_name = self.generate_shared_memeory_output_name(i,self.python_process_id)
            out_shape = gpu_process.output_shapes[i]
            out_dtype = gpu_process.output_dtypes[i]
            outputs.append(SharedMemoryObject(out_name,out_shape,out_dtype))
        return tuple(inputs),tuple(outputs)
    def IO_construct_shared_memory(self):
        for output in self.outputs:
            output.construct()
            #log.info(output.shm)
        for _input in self.inputs:
            _input.construct()
    def generate_shared_memeory_input_name(self,input_id,python_process_id):
        #return self.gpu_process.name+'-hash_{}'.format(self.gpu_process.hash)+'_input_'+str(input_id)+'_process_'+str(python_process_id)
        return self.gpu_process.name+'_input_'+str(input_id)+'_process_'+str(python_process_id)
    def generate_shared_memeory_output_name(self,input_id,python_process_id):
        #return self.gpu_process.name+'-hash_{}'.format(self.gpu_process.hash)+'_output_'+str(input_id)+'_process_'+str(python_process_id)
        return self.gpu_process.name+'_output_'+str(input_id)+'_process_'+str(python_process_id)                                             
    def create_client_function(self,put_to_queue):
        event = get_gpu_event(self.python_process_id)        
        wait_for_done_event=event.wait
        reset_event_flag=event.clear
        input_arrays = tuple(_input.bind(unregister=True).array for _input in self.inputs)
        output_arrays = tuple(_output.bind(unregister=True).array for _output in self.outputs)
        copyto = np.copyto
        copy = np.array
        input_range = range(len(input_arrays))
        output_range = range(len(output_arrays))
        _hash = self.gpu_process.hash

        request_arg=['eval_gpu_process',self.python_process_id,_hash]
        #log.info('input arrays = {}'.format(len(input_arrays)))
        #log.info('output arrays = {}'.format(len(output_arrays)))
        if len(output_arrays)==1:
            output_array=output_arrays[0]
            if len(input_arrays)==1:
                input_array= input_arrays[0]
                def f(arg):
                    #log.info('yay')
                    #log.info('trying to copy to shared memory')
                    #log.info('input array shape = {} shared memory shape = {}'.format(arg.shape,input_array.shape))
                    #log.info('input array shape = {} shared memory shape = {}'.format(arg.dtype,input_array.dtype))
                    #log.info('input array = {}'.format(input_array[:]))
                    #input_array[:] =
                    #log.info(arg)                    
                    copyto(input_array,arg)
                    #log.info('trying to send eval request to controller')
                    #log.info('requesting calculation from Process {}'.format(self.python_process_id))
                    put_to_queue(request_arg)
                    #log.info('Waiting for controller to finish')
                    wait_for_done_event()
                    #log.info('Client {} got done event.'.format(self.python_process_id))
                    #log.info('Unsetting event flag.')
                    reset_event_flag()
                    #log.info('Client {} reset event flag.'.format(self.python_process_id))
                    
                    #log.info('copy output array and return')
                    return copy(output_array)
            else:
                def f(*args):
                    for i in input_range:                        
                        copyto(input_arrays[i],args[i])
                    put_to_queue(request_arg)
                    wait_for_done_event()
                    reset_event_flag()
                    return copy(output_array)                
        else:
            if len(input_arrays)==1:
                input_array= input_arrays[0]
                def f(arg):
                    copyto(input_array,arg)
                    put_to_queue(request_arg)
                    wait_for_done_event()
                    reset_event_flag()
                    return tuple(copy(output_arrays[o]) for o in output_range)
            else:
                def f(*args):
                    for i in input_range:                        
                        copyto(input_arrays[i],args[i])
                    put_to_queue(request_arg)
                    wait_for_done_event()
                    reset_event_flag()
                    return tuple(copy(output_arrays[o]) for o in output_range)
        self.run = f
        return self

    def create_server_function(self):
        self.IO_construct_shared_memory()
        #log.info(self.inputs)
        #log.info([inp.shm for inp in self.inputs])
        #log.info('create_server_function {}'.format(self.python_process_id))
        event = get_gpu_event(self.python_process_id)
        set_event_flag=event.set
        input_arrays = [inp.array for inp in self.inputs]
        output_arrays = [outp.array for outp in self.outputs]
        run_process = self.gpu_process.assemble(input_arrays = input_arrays,output_arrays = output_arrays).run
        def fct():
            #log.info('run server for process {}'.format(self.python_process_id))
            run_process()
            #log.info('process {} server: done!'.format(self.python_process_id))
            set_event_flag()
        self.run = fct
        return self
    
    def remove_shared_memory(self):
        for i in self.inputs:
            #log.info('call __del__ of input {}'.format(i))
            i.release()
        for o in self.outputs:
            #log.info('call __del__ of output {}'.format(o))
            o.release()



class GpuProcessManager:
    @staticmethod
    def request_gpu_handle_client_old(cls,target_name,gpu_process):
        target_queue = get_process_queue(target_name)
        python_process_id = get_process_name()
        client_queue = get_gpu_queue(python_process_id)
        log.info('Sending question of gpu process exists from process {}'.format(python_process_id))
        target_queue.put(['does_gpu_process_exist',python_process_id,gpu_process.hash])
        gpu_process_exists = client_queue.get(timeout=60.0) #tries for 60 seconds 
        log.info('Got answer: {}'.format(gpu_process_exists))
        if not gpu_process_exists:
            log.info('Sending gpu_porocess creation request from id {}'.format(python_process_id))
            target_queue.put(['create_gpu_process',python_process_id,gpu_process.dict])
            gpu_creation_done = client_queue.get(timeout=60.0) #tries for 60 seconds 
            log.info('Gpu process creation is done = {}'.format(gpu_creation_done))
        process_handle = GpuProcessHandle(gpu_process,python_process_id)
        process_handle.create_client_function(target_queue.put)
        log.info('Return client handle for gpu process')
        return process_handle
        
    def __init__(self):
        self.process_dict={}
        self.server_functions={}
        self.process_handles={}
        eval_gpu_process = self.generate_eval_gpu_function_request()
        self.request_dict={
            'does_gpu_process_exist' : self.process_does_gpu_process_exist_request,
            'create_gpu_process':self.process_create_gpu_function_request,
            'eval_gpu_process':eval_gpu_process
        }        
        self.process_request = self.generate_process_request()
        self.client_handles = {} # collects handles in individual client processes        
        
    def request_gpu_handle_client(self,target_name,gpu_process):

        # Check if client handle exists
        _hash = gpu_process.hash
        client_handle_exists=False
        client_handle = self.client_handles.get(_hash,False)
        client_handle_exists = False
        if isinstance(client_handle,GpuProcessHandle):
            if client_handle.python_process_id == get_process_name():
                client_handle_exists = True
        #xprint(f'P{get_process_name()}: Client handle exists: {client_handle_exists}')

        # Check if Server handle exists
        target_queue = get_process_queue(target_name)
        python_process_id = get_process_name()
        client_queue = get_gpu_queue(python_process_id)
        #xprint(f'Sending question if gpu process exists from process {python_process_id} with hash {gpu_process.hash}')
        target_queue.put(['does_gpu_process_exist',python_process_id,gpu_process.hash])
        server_gpu_process_exists = client_queue.get(timeout=60.0) #tries for 60 seconds 
        #log.info('Got answer: {}'.format(server_gpu_process_exists))
        
        if not server_gpu_process_exists:
            #log.info('Sending gpu_porocess creation request from id {}'.format(python_process_id))
            target_queue.put(['create_gpu_process',python_process_id,{"process_dict":gpu_process.dict,"hash":gpu_process.hash}])
            gpu_creation_done = client_queue.get(timeout=60.0) #tries for 60 seconds 
            #log.info('Gpu process creation is done = {}'.format(gpu_creation_done))
        
            client_handle = GpuProcessHandle(gpu_process,python_process_id)
            client_handle.create_client_function(target_queue.put)
            self.register_client_function(client_handle)
        elif not client_handle_exists:
            #xprint("generating client handle")
            client_handle = GpuProcessHandle(gpu_process,python_process_id)
            client_handle.create_client_function(target_queue.put)
            self.register_client_function(client_handle)
        
        #log.info('Return client handle for gpu process')
        #xprint(f"P{get_process_name()}: handle inputs = {client_handle}")
        #xprint(f"P{get_process_name()}: handle inputs = {client_handle.run}")
        return client_handle
        
    def generate_eval_gpu_function_request(self):
        functions_dict = self.server_functions
        def eval_gpu_function(python_process_id,_hash):
            #log.info('eval gpu_func {} for process {}'.format(_hash,python_process_id))
            functions_dict[_hash][python_process_id]()
        return eval_gpu_function
            
    def generate_process_request(self):
        get_request_function = self.request_dict.get
        def do_nothing(python_process_id,data):
            pass
        def process_request(request_type,python_process_id,data):
            get_request_function(request_type,do_nothing)(python_process_id,data)
        return process_request
    def process_create_gpu_function_request(self,python_process_id,request_data):
        gpu_process_data = request_data["process_dict"]
        _hash = request_data["hash"]
        # check again if GPU process was not already created
        server_process_missing = _hash not in self.process_dict
        #xprint(f"Server add gpu: process is still missing {server_process_missing}")
        if server_process_missing:        
            # python_process_id is name of client process wanting to get access to a gpu process.
            gpu_process=openCL_plugin.ClProcess(gpu_process_data)
            self.add_gpu_process(gpu_process)            
            self.register_server_function(gpu_process.hash,python_process_id)
        elif python_process_id not in self.process_handles[_hash]:
            self.register_server_function(_hash,python_process_id)

        #send process created to client
        put_to_gpu_queue(python_process_id,True)
        
    def process_does_gpu_process_exist_request(self,python_process_id,_hash):
        gpu_process_exists = False
        #xprint(f"server hash {_hash} known = {_hash in self.process_dict}")
        if _hash in self.process_dict:
            server_function_for_python_process_is_already_registered =  python_process_id in self.process_handles[_hash]
            #xprint(f"server function exsists = {server_function_for_python_process_is_already_registered}")
            if not server_function_for_python_process_is_already_registered:
                self.register_server_function(_hash,python_process_id)
            
            gpu_process_exists = True
        put_to_gpu_queue(python_process_id,gpu_process_exists)
        #log.info('Process {}, gpu_function already exists {}.'.format(python_process_id,gpu_process_exists))
                        
    def add_gpu_process(self,gpu_process):
        _hash = gpu_process.hash
        process_dict = self.process_dict
        if _hash not in process_dict:
            process_list = openCL_plugin.create_process_buffers_on_all_gpus(gpu_process.dict)
            self.process_dict[_hash]=process_list
    def register_client_function(self,gpu_handle):
        #log.info('added client handle for gpu_process {} on cpu_process{}.'.format(gpu_handle.gpu_process.name,gpu_handle.python_process_id))
        self.client_handles[gpu_handle.gpu_process.hash] = gpu_handle
        
    def register_server_function(self,_hash,python_process_id):
        gpu_process = self.select_gpu_process(_hash,python_process_id)
        handle = GpuProcessHandle(gpu_process,python_process_id)
        handle = handle.create_server_function()
        server_function = handle.run
        functions_by_gpu_process = self.server_functions.get(_hash,{})
        handles_by_gpu_process = self.process_handles.get(_hash,{})
        functions_by_gpu_process[python_process_id]=server_function
        handles_by_gpu_process[python_process_id]=handle
        #xprint(f'registering server function with hash {_hash}' )
        self.server_functions[_hash]=functions_by_gpu_process
        self.process_handles[_hash]=handles_by_gpu_process
        #xprint(f'Server {get_process_name()}: added handle for gpu_process {handle.gpu_process.name} for client  {python_process_id}.')
        
    def select_gpu_process(self,process_hash,python_process_id):
        gpu_process_id = self.distribution_function(process_hash,python_process_id)
        return self.process_dict[process_hash][gpu_process_id]
        
    def distribution_function(self,process_hash,process_id):
        n_func_instances = len(self.process_dict[process_hash])
        return (process_id//settings.general.n_control_workers) % n_func_instances


    def release_shared_memory(self):
        #log.info('process handles {}, client_handles {}'.format(self.process_handles,self.client_handles))
        #log.info('trying to delete gpu process manager in P {}'.format(get_process_name()))
        for h_processes in self.process_handles.values():
            for h in h_processes.values():
                h.remove_shared_memory()
        for h in self.client_handles.values():
            h.remove_shared_memory()
    def remove_shared_memory(self):
        self.__del__()
        


