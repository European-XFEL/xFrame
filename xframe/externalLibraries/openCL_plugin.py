import numpy as np
import sys
import importlib
import logging
log=logging.getLogger('root')
import pyopencl as cl
import abc
from xframe.Multiprocessing_interfaces import OpenClInterface
from xframe.Multiprocessing import get_process_name,master_name
from xframe.library.pythonLibrary import hash_numpy_in_tuple

int_type = np.int32

################################################

def _get_platforms(allow_master_process=False):
    '''
    By default prevents access to get_platforms for master process. This is because calls to cl.get_platforms() on the master call would cause all subsequent worker/children calls to cl.get_platforms to fail.
    So only set allow_master_process if you dont want to use multiprocessing.
    '''
    is_master_process = (get_process_name()==master_name)
    if (not is_master_process) or allow_master_process:
        platforms = cl.get_platforms()
    elif is_master_process:
        raise AssertionError("Creation of OpenCL context not allowed in master process.")
    return platforms

class ContextHandler:
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.context_attached = False
    def attach_context(self,ctx,queue):
        try:
            assert (not self.context_attached), 'openCL context is already attached. Ignoring call to attach_context'
            self.ctx=ctx
            self.queue = queue
            self.context_attached = True
        except AssertionError as e:
            log.warning(e)
                    
class OpenClPlugin(OpenClInterface):
    cl_state = False
    contexts_created = False
    
    @classmethod
    def create_context(cls,allow_master_process=False):
        '''
        creates openCL context and queue for all gpus if they not already exist.        
        '''
        platform = _get_platforms(allow_master_process)
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        cls.cl_state = []
        for device in my_gpu_devices:
            ctx = cl.Context(devices=[device])
            queue = cl.CommandQueue(ctx)        
            state = {}
            state['context']=ctx
            state['queue']=queue
            cls.cl_state.append(state)
        cls.contexts_created = True
        
    class ClFunction(ContextHandler):
        def __init__(self,func_data : dict, kernel,context_data=False):            
            super().__init__()
            if isinstance(context_data,(tuple,list)):
                self.attach_context(*context_data)
            self.kernel = kernel
            d = func_data            
            self.dict = d 
            self.name = d['name']
            self.dtypes = d['dtypes']
            self.shapes = d['shapes']
            self.arg_roles = d['arg_roles']
            self.const_inputs = d['const_inputs']
            self.global_range = self.read_ranges(d.get('global_range',None))
            self.local_range = self.read_ranges(d.get('local_range',None))
    
            self.n_args = len(self.arg_roles)
            #log.info("there are {} gpu arguments".format(self.n_args))
            self.input_dtypes = [self.dtypes[arg_id] for arg_id in range(self.n_args) if self.arg_roles[arg_id] == 'input']
            self.input_shapes = [self.shapes[arg_id] for arg_id in range(self.n_args) if self.arg_roles[arg_id] == 'input']
            self.input_ids = [ i for i in range(self.n_args) if self.arg_roles[i]=='input']
            self.n_inputs = len(self.input_dtypes)
            self.output_dtypes = [self.dtypes[arg_id] for arg_id in range(self.n_args) if self.arg_roles[arg_id] == 'output']
            self.output_shapes = [self.shapes[arg_id] for arg_id in range(self.n_args) if self.arg_roles[arg_id] == 'output']
            self.output_ids = [ i for i in range(self.n_args) if self.arg_roles[i]=='output']
            self.n_outputs = len(self.output_dtypes)
            self.scalar_dtypes = [self.dtypes[i] if self.shapes[i]==None else None for i in range(self.n_args)]
            self.call = None

            self.cl_buffers = None
    
        @staticmethod
        def read_ranges(r):
            if not isinstance(r,(np.ndarray,tuple,list,int)):
                return None
            elif isinstance(r,np.ndarray):
                return r.astype(int_type)
            elif isinstance(r,int):
                return (int_type(r),)
            else:
                return tuple(int_type(i) for i in r)
 
        def assemble_buffers(self,input_buffers=False):
            assert self.context_attached, 'openCL context is not jet attached to this instance of CLFunction. You need to call attach_context befor calling assemble_buffers.'
            ctx = self.ctx
            mf = cl.mem_flags    
            cl_args = []            
            for arg_id in range(self.n_args):
                shape = self.shapes[arg_id]
                dtype = self.dtypes[arg_id]
                arg_role = self.arg_roles[arg_id]
                if arg_role in ['input','output'] :
                    if not isinstance(input_buffers,list):
                        if shape != None:
                            _buffer = cl.Buffer(ctx,mf.READ_WRITE,size=np.prod(shape)*np.dtype(dtype).itemsize)
                            cl_args.append(_buffer)
                        else:
                            cl_args.append(None)
                    else:
                        try:
                            cl_arg.append(input_buffers.pop(0))
                        except IndexError as e:
                            log.error('There are {} input buffers needed for CLFunction {} but not enough buffers have been provided to assemble_buffers.'.format(self.n_inputs,self.name))
                            raise e
                elif arg_role =='const_input':            
                    const_input = self.const_inputs[arg_id]
                    if shape != None:
                        _buffer = cl.Buffer(context = ctx,flags =  mf.READ_ONLY | mf.COPY_HOST_PTR,size = const_input.nbytes, hostbuf=np.ascontiguousarray(const_input))
                        cl_args.append(_buffer)
                    else:
                        cl_args.append(dtype(const_input))
                elif arg_role == 'local':
                    _buffer = cl.LocalMemory(np.prod(shape)*dtype().nbytes)
                    cl_args.append(_buffer)
                else:
                    raise AssertionError('Argument role {} is not known. Known roles are (input,output,const_input,local)'.format(arg_role))
                    
                if isinstance(input_buffers,list):
                    assert len(input_buffers)==0, 'There are exactely {} input buffers needed for CLFunction {} but more buffers have been provided to assemble_buffers.'.format(self.n_inputs,self.name)
            self.cl_buffers=cl_args
            #log.info(self.cl_buffers)
            return self

        def get_function(self,input_arrays = False,output_arrays = False):
            if (not isinstance(input_arrays,(list,tuple))) and (not isinstance(output_arrays,(list,tuple))):
                fct = self.get_function_unknown_IO_arrays()
            else:
                fct = self.get_function_known_IO_arrays(input_arrays,output_arrays)
            self.call =fct
            return fct
        
        def get_function_known_IO_arrays(self,input_arrays,output_arrays):
            assert self.context_attached, 'openCL context is not jet attached to this instance of CLFunction. You need to call attach_context befor calling assemble_function.'
            if not isinstance(self.cl_buffers,list):
                self.assemble_buffers()
            kernel = self.kernel
            if isinstance(kernel, str):
                kernel = cl.Program(self.ctx,kernel).build()
                self.kernel = kernel                
            cl_fct = kernel.__getattr__(self.name)
            cl_buffers = self.cl_buffers
            #log.info(cl_buffers)
            input_ids = self.input_ids
            output_ids = self.output_ids
            
            input_buffers = [cl_buffers[i] for i in input_ids]            
            output_buffers = [cl_buffers[i] for i in output_ids]
            cl_fct.set_scalar_arg_dtypes(self.scalar_dtypes)
        
            local_range = self.local_range
            global_range = self.global_range        
            n_inputs = self.n_inputs
            n_outputs = self.n_outputs
            enqueue_copy = cl.enqueue_copy
            ascontiguousarray = np.ascontiguousarray
            queue = self.queue
            if n_outputs==1:
                output_buffer = output_buffers[0]
                output_array = output_arrays[0]
                if n_inputs == 1:
                    input_buffer = input_buffers[0]
                    input_array= input_arrays[0]
                    def fct():
                        #pushing input arrays to GPU Memory
                        #log.info('input_buffer = {}'.format(input_buffer))
                        enqueue_copy(queue,input_buffer,ascontiguousarray(input_array))            
                        #executing kernel
                        #log.info('global_range {} '.format(global_range))
                        #log.info('local_range {} '.format(local_range))
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        enqueue_copy(queue,output_array,output_buffer)
                else:
                    def fct():
                        #pushing input arrays to GPU Memory
                        for i in range(n_inputs):
                            enqueue_copy(queue,input_buffers[i],ascontiguousarray(input_arrays[i]))
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        enqueue_copy(queue,output_array,output_buffer)
            else: 
                if n_inputs == 1:
                    input_buffer = input_buffers[0]
                    input_array= input_arrays[0]
                    def fct():
                        #pushing input arrays to GPU Memory
                        enqueue_copy(queue,input_buffer,ascontiguousarray(input_array))            
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        for i in range(n_outputs):
                            enqueue_copy(queue,output_arrays[i],output_buffers[i])
                else:
                    def fct():
                        #pushing input arrays to GPU Memory
                        for i in range(n_inputs):
                            enqueue_copy(queue,input_buffers[i],ascontiguousarray(input_arrays[i]))
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        for i in range(n_outputs):
                            enqueue_copy(queue,output_arrays[i],output_buffers[i])
            return fct
            
        def get_function_unknown_IO_arrays(self):
            assert self.context_attached, 'openCL context is not jet attached to this instance of CLFunction. You need to call attach_context befor calling assemble_function.'
            if not isinstance(self.cl_buffers,list):
                self.assemble_buffers()
            kernel = self.kernel
            if isinstance(kernel, str):
                kernel = cl.Program(self.ctx,kernel).build()
                self.kernel = kernel
            cl_fct = kernel.__getattr__(self.name)
            cl_buffers = self.cl_buffers
            input_ids = self.input_ids
            output_ids = self.output_ids
            output_arrays = [np.zeros(shape,dtype = dtype) for shape,dtype in zip(self.output_shapes,self.output_dtypes)]
            output_buffers = [cl_buffers[i] for i in output_ids]
            
            cl_fct.set_scalar_arg_dtypes(self.scalar_dtypes)
        
            local_range = self.local_range
            global_range = self.global_range        
            n_inputs = self.n_inputs
            n_outputs = self.n_outputs
            enqueue_copy = cl.enqueue_copy
            ascontiguousarray = np.ascontiguousarray
            queue = self.queue
            
            if n_outputs==1:
                output_arg = cl_buffers[output_ids[0]]
                output_array = output_arrays[0]
                if n_inputs == 1:
                    input_arg = cl_buffers[input_ids[0]]            
                    def fct(arg):
                        #pushing input arrays to GPU Memory
                        enqueue_copy(queue,input_arg,ascontiguousarray(arg))            
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        enqueue_copy(queue,output_array,output_arg)
                        return output_array
                else:
                    def fct(*args):
                        #pushing input arrays to GPU Memory
                        for i in range(n_inputs):
                            enqueue_copy(queue,cl_buffers[input_ids[i]],ascontiguousarray(args[i]))
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        enqueue_copy(queue,output_array,output_arg)
                        return output_array
            else: 
                if n_inputs == 1:
                    input_arg = cl_buffers[input_ids[0]]
                    def fct(arg):
                        #pushing input arrays to GPU Memory
                        enqueue_copy(queue,input_arg,ascontiguousarray(arg))            
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        for i in range(n_outputs):
                            enqueue_copy(queue,output_arrays[i],output_buffers[i])
                        return output_arrays
                else:
                    def fct(*args):
                        #pushing input arrays to GPU Memory
                        for i in range(n_inputs):
                            enqueue_copy(queue,cl_buffers[input_ids[i]],ascontiguousarray(args[i]))
                        #executing kernel
                        cl_fct(queue,global_range,local_range,*cl_buffers)
                        #extract output arrays from GPU
                        for i in range(n_outputs):
                            enqueue_copy(queue,output_arrays[i],output_buffers[i])
                        return output_arrays
            self.call = fct
            return fct
    
    class ClProcess(ContextHandler):
        def __init__(self, process_data : dict,context_data=False):
            super().__init__()
            context_data_available = isinstance(context_data,(tuple,list))
            if context_data_available:
                super().attach_context(*context_data)
            self.dict = process_data
            self.name = process_data['name']
            #self.hash = hash((process_data['kernel'], process_data['name']))
            self.hash = hash(
                (process_data['kernel'],
                 process_data['name'],
                 tuple( (fdat['shapes'],hash_numpy_in_tuple(fdat['const_inputs'])) for fdat in process_data['functions'])
            ))
            self.kernel = process_data['kernel']
            self.functions = [OpenClPlugin.ClFunction(func_data,self.kernel,context_data) for func_data in process_data['functions']]
            self.n_functions = len(self.functions)
            self.input_dtypes = self.functions[0].input_dtypes
            self.input_shapes = self.functions[0].input_shapes
            self.n_inputs = self.functions[0].n_inputs
            self.output_dtypes = self.functions[-1].output_dtypes
            self.output_shapes = self.functions[-1].output_shapes
            self.n_outputs = self.functions[0].n_outputs
            self.run = None
        def assemble_buffers(self):
            self.functions[0].assemble_buffers()
            out_ids = self.functions[0].output_ids
            out_buffers = [self.functions[0].cl_buffers[i] for i in out_ids]
            for cl_function in self.functions[1:]:
                cl_function.assemble_buffers(input_buffers=out_buffers)
                out_ids = cl_function.output_ids
                out_buffers = [ cl_function.cl_buffers[i] for i in out_ids]
            return self
                
        def assemble(self,input_arrays=False,output_arrays=False):
            assert self.context_attached, 'openCL context is not jet attached to this instance of CLProcess. You need to call attach_context befor calling assemble.'
            if self.n_functions == 1:
                self.run = self.assemble_1fct_process(input_arrays=input_arrays,output_arrays= output_arrays)
            else:
                self.run = self.assemble_nfct_process()
            return self
        def assemble_1fct_process(self,input_arrays = False,output_arrays = False):            
            return self.functions[0].get_function(input_arrays = input_arrays,output_arrays=output_arrays)
        def asssemble_nfct_process(self):
            cl_args = [self.functions[0].assemble_buffers()]
            initial_function = self.functions[0]
            output_ids = initial_function.output_ids
            for cl_function in self.functions[1:]:
                cl_args.append(cl_function.assemble_buffers(input_buffers=cl_args[-1][output_ids]))
                output_ids = cl_functions.output_ids
            raise NotImplementedError('GPU processes in which multiple GPU functions are called in succession are not implemented jet.')

        def attach_context(self,ctx,queue):
            super().attach_context(ctx,queue)
            for function in self.functions:
                function.attach_context(ctx,queue)
            
        
    @staticmethod
    def get_number_of_gpus(allow_master_process=False):
        platform = _get_platforms(allow_master_process)
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        n_gpus = len(my_gpu_devices)
        return n_gpus

    @classmethod
    def create_function(cls,kernel_data):
        fct_by_gpu = []
        for state in cls.cl_state:
            fct_per_gpu.append(cls.CLProcess(kernel_data,state['context'],state['queue']).assemble().run)    
        return fct_by_gpu
    @classmethod
    def create_process_on_all_gpus(cls, process_data:dict):
        process_by_gpu = []
        for state in cls.cl_state:
            process_by_gpu.append(cls.CLProcess(kernel_data,state['context'],state['queue']).assemble())    
        return process_by_gpu
    @classmethod
    def create_process_buffers_on_all_gpus(cls,process_data:dict):
        process_by_gpu = []
        for state in cls.cl_state:
            process_by_gpu.append(cls.ClProcess(process_data,context_data=[state['context'],state['queue']]).assemble_buffers())    
        return process_by_gpu

    

            
        
