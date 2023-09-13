import numpy as np
import logging

log=logging.getLogger('root')
from xframe.library import pythonLibrary as pyLib
xprint = pyLib.xprint
from xframe.interfaces import CommunicationInterface as ComInterfaceAnalysis
# in place for future upgrade to MPI
from xframe import Multiprocessing
from xframe.Multiprocessing import MPMode,MPMode_Queue
cpu_count = Multiprocessing.get_local_cpu_count


class SingleProcessCommunictionAnalysis(ComInterfaceAnalysis):
    def __init__(self,control_worker=False):
        self.control_worker=control_worker

    def get_data(self,opt):
        data=self.control_worker.processDataRequest_SingleProcess(opt)
        return data

    def get_experiment(self):
        return self.control_worker.experiment_worker
    
    def get_geometry(self,approximation='None',out_coord_sys='spherical'):
        geometry=self.control_worker.process_geometry_request_single_thread(approximation=approximation,out_coord_sys = out_coord_sys)
        return geometry
    def get_exp_info(self):
        return self.control_worker.process_exp_info_request_single_thread()

    def request_mp_evaluation(self,
                                func,
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
        result = self.control_worker.process_mp_request_single_node(func,
                                 mode = mode,
                                is_daemon=is_daemon,
                                return_process_objects = return_process_objects,
                                input_arrays=input_arrays,
                                const_inputs=const_inputs,
                                call_with_multiple_arguments=call_with_multiple_arguments,
                                split_mode=split_mode,
                                split_together=split_together,
                                n_processes = n_processes,
                                event_names=event_names,
                                create_queues = create_queues)
        return result
    def get_mp_lock(self):
        return self.control_worker.mp_lock

    def n_cpus(self):
        return cpu_count()

    def free_mem(self):
        return Multiprocessing.get_free_memory()

    def getGatheredResults():
        pass
    
    def sendResults():
        pass

    def put_string_to_controller(self,string):
        cpu_id = Multiprocessing.get_process_id()
        log.info('cpu id = {}'.format(cpu_id))
        self.control_worker.queue.put([cpu_id,string])
        Multiprocessing.wait_for_cpu_event(cpu_id)
        Multiprocessing.clear_cpu_event(cpu_id)        

    def add_gpu_process(self,gpu_process):
        control_worker = self.control_worker.select_control_worker()
        handle = self.control_worker.gpu_manager.request_gpu_handle_client(control_worker.name,gpu_process)
        return handle.run
    def restart_control_worker(self):
        self.control_worker.restart_working()


class MPICommunictionAnalysis(ComInterfaceAnalysis):
    def __init__(self):
        pass
    
    def getData(self):
        pass

    def getGeometry(self):
        pass

class MPICommunictionExperiment(ComInterfaceAnalysis):
    def __init__(self):        
        pass
    
    def getData(self):
        pass

    def getGeometry(self):
        pass
