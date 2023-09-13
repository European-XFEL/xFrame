import os
import logging
import traceback
import numpy as np

from itertools import repeat

from xframe.control import communicators as commModules
from xframe.interfaces import ProjectWorkerInterface,ExperimentWorkerInterface
from xframe.library import pythonLibrary as pyLib
from xframe.library.pythonLibrary import DictNamespace, xprint
from xframe.Multiprocessing import get_local_cpu_count as cpu_count
from xframe import settings
from xframe import database
from xframe.Multiprocessing import process_mp_request, MPMode
from xframe import Multiprocessing
#get_global_number_of_processes = Multiprocessing.get_global_number_of_processes

#import warnings
#warnings.filters.append(('ignore',None,UserWarning,None,0))

log=logging.getLogger('root')




class Controller:
    def __init__(self,project_worker=False,experiment_worker=False):       
        self._project_worker = project_worker
        self._experiment_worker = experiment_worker
        self.control_worker=ControlWorker(project_worker = project_worker,experiment_worker=experiment_worker)        
        Multiprocessing.comm_module = commModules.SingleProcessCommunictionAnalysis(control_worker = self.control_worker)
        self.job='not jet chosen'
        
    @property
    def project_worker(self):
        return self._project_worker
    @property
    def experiment_worker(self):
        return self._experiment_worker
    @project_worker.setter
    def project_worker(self,worker):
        try:
            #log.info('trying to set analysis worker in COntroller')
            #assert isinstance(worker,AnalysisWorker),'analysis_worker has to inherit from {} but the given object {} does not.'.format(AnalysisWorker,worker)
            self._project_worker = worker
            self.control_worker.project_worker = worker
            #log.info(self._analysis_worker)
        except AssertionError as e:
            log.error(e)            
    @experiment_worker.setter
    def experiment_worker(self, worker):
        try:
            #assert isinstance(worker,Experiment_WorkerWorkerInterface),'experiment_worker_worker has to inherit from {} but the given object {} does not.'.format(Experiment_WorkerWorkerInterface,worker)
            self._experiment_worker = worker
            self.control_worker.experiment_worker = worker
        except AssertionError as e:
            log.error(e)
            
            
            
    def run(self,oneshot=False,restart_control_worker=False):
        settings._save_settings_on_controller_run()
        if restart_control_worker:
            # restart allows the control_workers to reload settings and change from the default
            # ones to the settings provided by the analysis worker. (e.g for profiling)
            self.control_worker.restart_working()
        try:
            if not self.control_worker.are_workers_running:
                self.control_worker.start_working()
                
            job=self.choose_job()
            self.job = job
            xprint('\n ------- Start {} ------- \n'.format(job))
            result=job.run()
            xprint('\n ------- Finished {} ------- \n'.format(job))
        except Exception as e:
            traceback.print_exc()
            result = None
            log.error('The follwoing exception occured during, or directly before, job execution:\n {}'.format(e))
            
        if oneshot:
            del(self.control_worker.gpu_manager)
            self.control_worker.stop_working()
            
        if result != None:
            return result

    def choose_job(self):
        n_running_nodes = 1 #get_global_number_of_processes()
        if n_running_nodes==1:
            job=self.choose_job_single_node()
        else:
            job=self.choose_job_multi_node()
        return job
            
    def choose_job_single_node(self):
        return self.project_worker
    
    def choose_job_multi_node(self):
        return self.project_worker

    
class ControlWorker:
    def __init__(self,project_worker=False,experiment_worker=False,start_working=True):
        self.project_worker=project_worker
        self.experiment_worker=experiment_worker
        #self.dataSetsToAnalizeByID=dataSetsToAnalizeByID
        #self.remainingIDs=dataSetsToAnalizeByID.copy()
        #manager=Manager()       
        self.gpu_manager = Multiprocessing.GpuProcessManager()
        self.workers = []
        if start_working:
            self.start_working()

    def are_workers_running(self):
        some_workers_are_running = False
        for worker in self.workers:
            if worker.process.is_alive():
                some_workers_are_running = True
                break
        return some_workers_are_running
    def start_working(self):        
        if not self.are_workers_running():
            #log.info('run {} controll workers'.format(len(self.worker_processes)))
            if settings.general.n_control_workers!=0:
                _,workers = self.process_mp_request_single_node(self.work,is_daemon = True,return_process_objects = True,n_processes = settings.general.n_control_workers,create_queues = True)                
                self.workers = workers
        else:
            log.error('Some workers are still running can not start them again. Call stop_working or restart_working to chancle the old ones.')
    def stop_working(self):
        for worker in self.workers:
            worker.queue.put(("terminate",0,True))
            worker.process.join()
        self.workers = []
        
    def restart_working(self):
        if self.are_workers_running():
            self.stop_working()
        self.start_working()
        log.info('Run with {} control workers'.format(len(self.workers)))
        

    def work(self,**kwargs):
        q = kwargs['queue']
        #if not isinstance(self.gpu_manager,Multiprocessing.GpuProcessManager):
        #    print('Something went wrong, starting new GpuProcessManager!')
        #    self.gpu_manager = Multiprocessing.GpuProcessManager()
        run_profiling=settings.project.get('profiling',DictNamespace(enable=False)).enable
        if run_profiling:
            process_id = settings.project.profiling.gpu_worker_id
            #log.info('controler name = {}'.format(Multiprocessing.get_process_name()))
            if Multiprocessing.get_process_name() == -1*np.abs(process_id):
                try:
                    db = database.project
                    path = db.get_reconstruction_path()
                except Exception:                    
                    db = database.default
                    path = db.get_path('base',is_file=False)
                db.create_path_if_nonexistent(path)
                path += "control_worker_{}.stats".format(process_id)
                log.info('profile path = {}'.format(path))
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
                self.request_loop(q)
                profiler.disable()
                profiler.dump_stats(path)
            else:
                self.request_loop(q)
        else:
            self.request_loop(q)
            
    def request_loop(self,q):
        try:
            Multiprocessing.create_openCL_context()
            #log.info('context created in process {}'.format(Multiprocessing.get_process_id))
            gpu_process_request=self.gpu_manager.process_request
            #log.info('worker pid = {}'.format(os.getpid()))
            while True:
                request_type,cpu_id,data = q.get()            
                if request_type == "terminate":
                    #log.info('Request {} from process {} to {}'.format(request_type,cpu_id,Multiprocessing.get_process_name()))
                    break
                elif request_type == 'ping':
                    worker_id = Multiprocessing.get_process_name()
                    message = ('ping',worker_id)
                    Multiprocessing.put_to_process_queue(cpu_id,message)
                else:
                    gpu_process_request(request_type,cpu_id,data)
        finally:
            self.exit_request_loop()
            
        
    def exit_request_loop(self):
        #print('deleting gpu_manager in P {}'.format(Multiprocessing.get_process_name()))
        self.gpu_manager.release_shared_memory()
        self.gpu_manager = Multiprocessing.GpuProcessManager()
        
    def select_worker_queue(self):
        p_id = Multiprocessing.get_process_id()
        return self.queues[p_id%len(self.queues)]

    def select_control_worker(self):
        p_id = Multiprocessing.get_process_name()
        worker = self.workers[p_id%len(self.workers)]
        #log.info('reconstruction process {} selected control worker {}'.format(p_id,worker.name))
        return worker
    def processDataRequest_SingleProcess(self,opt):
        data_generator = self.experiment_worker.get_data(opt)
        return data_generator

    def process_geometry_request_single_thread(self,**kwargs):
        geometry=self.experiment_worker.get_geometry(**kwargs)
        return geometry

    def process_exp_info_request_single_thread(self):
        return self.experiment_worker.info

    def process_mp_request_single_node(self,*args,**kwargs):
        return process_mp_request(*args,**kwargs)
        
