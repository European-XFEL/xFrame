import pytest
import shutil
import os
import sys
import subprocess
import numpy as np
import importlib
from tlib import copy,save,load,create_path_if_nonexistent
import xframe
config_path = os.path.expanduser('~/.xframe/config.py')
config_backup_path = os.path.expanduser('~/.xframe/backup_config.py')

tmp_config_path = ''
_path = ''

@pytest.fixture(scope='class')
def tmp_path(tmpdir_factory):
    path = tmpdir_factory.mktemp('xtest'+str(np.random.rand()))
    return path

@pytest.fixture(scope='class',autouse=True)
def set_temp_home(tmp_path):    
    config_file_exists = os.path.exists(config_path)
    if config_file_exists:
        initial_config = load(config_path)
        print(f'initial config content = {initial_config}')
        copy(config_path,config_backup_path)
    global tmp_config_path
    tmp_config_path = f'{tmp_path}/.xframe/'
    global _path
    _path=tmp_path
    print(f'given path exists: {os.path.exists(tmp_path)}')
    yield
    if config_file_exists:
        copy(config_backup_path,config_path)
    print(f'config after test = {load(config_path)}')
    
def home_files_exist():
    paths=[
        'projects/fxs',
        'projects/tutorial',
        'experiments/SPB',
        'settings/projects/fxs/correlate/tutorial.yaml',
        'settings/projects/fxs/correlate/default_0.01.yaml',
        'settings/projects/fxs/simulate_ccd/tutorial.yaml',
        'settings/projects/fxs/simulate_ccd/default_0.01.yaml',
        'settings/projects/fxs/extract/tutorial.yaml',
        'settings/projects/fxs/extract/default_0.01.yaml',
        'settings/projects/fxs/reconstruct/tutorial.yaml',
        'settings/projects/fxs/reconstruct/default_0.01.yaml',
        'settings/projects/fxs/average/tutorial.yaml',
        'settings/projects/fxs/average/default_0.01.yaml',
        'settings/experiments/SPB/tutorial.yaml',
        'settings/experiments/SPB/default_0.01.yaml',
    ]
    global tmp_config_path
    for path in paths:
        _path= os.path.join(tmp_config_path,path)
        exists = os.path.exists(_path)
        if not exists:
            assert _path == None


class TestsCLI:
    def test_home_cmd(self):
        self.setup_home()
        home_files_exist()

    def test_project_creation(self):
        new_project = tmp_config_path+'projects/tmp/hello.py'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        print('Hello There!')\n",
                ]
        create_path_if_nonexistent(new_project)
        save(new_project,code)
        command = f"xframe tmp hello"
        p = subprocess.run(command,shell=True,capture_output=True)
        assert 'Hello There!' in str(p.stdout)

    def test_settings(self):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        settings_path_2 = tmp_config_path+'settings/projects/tmp/hello/opt2.yaml'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import settings\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        opt = settings.project\n",
                "        print(f'Hello {opt.int}!')\n",
                ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        create_path_if_nonexistent(settings_path_2)
        save(settings_path_2,settings)

        for i in range(1,3):
            cmd =  f"xframe tmp hello opt{i}"
            p = subprocess.run(cmd,shell=True,capture_output=True)
            assert 'Hello 42!' in str(p.stdout)

    def test_database(self):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        data_path = tmp_config_path+'data/tmp{val}.h5'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import database\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        db = database.project\n",
                "        test_data={'answer':42}\n",
                "        path_modifiers={'val':0}\n",
                "        db.save('my_file',test_data,path_modifiers=path_modifiers)\n",
                "        loaded_file = db.load('my_file',path_modifiers=path_modifiers)\n",
                "        answer = loaded_file['answer']\n",
                "        print(f'Answer = {answer}')\n",
                ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    'IO:\n',
                    '  folders:\n',
                    f'    data: "{os.path.dirname(data_path)}/"\n',
                    '  files:\n',
                    '    my_file:\n',
                    f'     name: "{os.path.basename(data_path)}"\n',
                    '     folder: data\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        cmd =  f"xframe tmp hello opt1"
        p = subprocess.run(cmd,shell=True,capture_output=True)
        assert os.path.exists(data_path.format(val=0))==True
        assert 'Answer = 42' in str(p.stdout)

    def test_custom_database(self):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        data_path = tmp_config_path+'data/tmp.h5'
        db_path = tmp_config_path+'/projects/tmp/_database_.py'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import database\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        db = database.project\n",
                "        test_data={'answer':42}\n",
                "        db.save('my_file',test_data)\n",
                "        loaded_file = db.load('my_file')\n",
                "        answer = loaded_file['answer']\n",
                "        yay = loaded_file['new_load']\n",
                "        nay = loaded_file['new_save']\n",
                "        print(f'Answer = {answer} {yay} {nay}')\n",
                ]
        db_code = [
            "from xframe.database.database import DefaultDB\n",
            "from xframe.interfaces import DatabaseInterface\n",
            "\n",
            "class ProjectDB(DefaultDB,DatabaseInterface):\n",
            "    def load_my_file(self,name,**kwargs):\n",
            "        data = self.load_direct(name,**kwargs)\n",
            "        data['new_load'] = 'yay'\n",
            "        return data\n",
            "\n",
            "    def save_my_file(self,name,data,**kwargs):\n",
            "        data['new_save'] = 'ney'\n",
            "        data = self.save_direct(name,data,**kwargs)\n",            
        ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    'IO:\n',
                    '  folders:\n',
                    f'    data: "{os.path.dirname(data_path)}/"\n',
                    '  files:\n',
                    '    my_file:\n',
                    f'     name: "{os.path.basename(data_path)}"\n',
                    '     folder: data\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        create_path_if_nonexistent(db_path)
        save(db_path,db_code)
        cmd =  f"xframe tmp hello opt1"
        p = subprocess.run(cmd,shell=True,capture_output=True)
        #assert os.path.exists(data_path)==True
        assert 'Answer = 42 yay ney' in str(p.stdout)
      
    def test_simple_multiprocessing(self):
        new_project = tmp_config_path+'projects/tmp/mp.py'
        code = [
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import Multiprocessing\n",
            "import numpy as np\n",
            "\n",
            "def multiply_matrix_with_vectors(vects,matrix,**kwargs):\n",
            "    vects= np.atleast_2d(vects)\n",
            "    new_vects = np.sum(matrix[None,:,:]*vects[:,None,:],axis=2)\n",
            "    return np.squeeze(new_vects)\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "        vectors = np.random.rand(200,10)\n",
            "        matrix = np.random.rand(10,10)\n",
            "\n",
            "        result = Multiprocessing.process_mp_request(multiply_matrix_with_vectors,input_arrays=[vectors],const_inputs = [matrix])\n",
            "\n",
            "        test_result = multiply_matrix_with_vectors(vectors,matrix)\n",
            "        if (result == test_result).all():\n",
            "            print('Test passed!')\n",            
        ]
        save(new_project,code)
        cmd =  f"xframe tmp mp"
        p = subprocess.run(cmd,shell=True,capture_output=True)
        assert 'Test passed!' in str(p.stdout)        
    def test_GPU(self):
        new_project = tmp_config_path+'projects/tmp/gpu.py'
        code = [
            "import numpy as np\n",
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import settings\n",
            "from xframe import Multiprocessing\n",
            "import xframe\n",
            "\n",
            "# Ensure a GPU worker is running\n",
            "settings.general.n_control_workers = 1\n",
            "xframe.controller.control_worker.restart_working()\n",
            "\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "\n",
            "        nq = 10 # vector length\n",
            "        nvec=5  # number of vectors\n",
            "        matrix = np.random.rand(nq,nq)\n",
            "        vects= np.random.rand(nq,nvec)\n",
            "        expected = matrix@vects # Expected result from module import symbol the gpu process\n",
            "\n",
            "        gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function\n",
            "\n",
            "        result = gpu_func(vects) # Evaluate kernel on given vectors\n",
            "        if (result==expected).all(): # Check if gpu version gives same result as numpy computation.\n",
            "            print('Test passed!') \n",
            "\n",
            "    def create_gpu_function(self,matrix,vects):\n",
            "        nq,nvec=vects.shape\n",
            "        # Define Kernel\n",
            '        kernel_str = """\n',
            "            __kernel void\n",
            "            apply_matrix(__global double* out, \n",
            "            __global double* matrix, \n",
            "            __global double* vect, \n",
            "            long nq,long nvec)\n",
            "            {\n",
            "            long i = get_global_id(0); \n",
            "            long j = get_global_id(1);\n",
            "\n",
            "            // Compute application of i'th matrix row on j'th vector\n",
            "            // Store result in value\n",
            "            double value = 0;\n",
            "            for (int q = 0; q < nq; ++q)\n",
            "            {\n",
            "            double matqq = matrix[i*nq + q];\n",
            "            double veciq = vect[q*nvec + j];\n",
            "            value += matqq * veciq;\n",
            "            }\n",
            "\n",
            "            // Write the result vector to device memory\n",
            "            out[i * nvec + j] = value;\n",
            "            }\n",
            '            """\n',
            "        # Define types and input constant arguments\n",
            "        kernel_dict_forward={\n",
            "            'kernel': kernel_str,\n",
            "            'name': 'gpu_func',\n",
            "            'functions': ({\n",
            "                'name': 'apply_matrix',\n",
            "                'dtypes' : (float,float,float,np.int64,np.int64),\n",
            "                'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),\n",
            "                'arg_roles' : ('output','const_input','input','const_input','const_input'),\n",
            "                'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),\n",
            "                'global_range' : (nq,nvec),\n",
            "                'local_range' : None\n",
            "                },)\n",
            "            }\n",
            "        # Create cl Process\n",
            "        cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)\n",
            "        # Register opencl function and get gpu client function\n",
            "        gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)\n",
            "        return gpu_func\n",            
        ]
        save(new_project,code)
        cmd =  f"xframe tmp gpu"
        p = subprocess.run(cmd,shell=True,capture_output=True)
        assert 'Test passed!' in str(p.stdout)
        pass
    def test_GPU_in_multiprocessing(self):
        new_project = tmp_config_path+'projects/tmp/gpu.py'
        n_processes = 10
        code = [
            "import numpy as np\n",
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import settings\n",
            "from xframe import Multiprocessing\n",
            "import xframe\n",
            "\n",
            "# Ensure a GPU worker is running\n",
            "settings.general.n_control_workers = 1\n",
            "xframe.controller.control_worker.restart_working()\n",
            "\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "        nq = 10 # vector length\n",
            "        nvec=5  # number of vectors\n",
            "        matrix = np.random.rand(nq,nq)\n",
            "        n_processes = 10\n",
            "        def run_parallel(**kwargs):\n",
            "            vects= np.random.rand(nq,nvec)\n",
            "            gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function\n",
            "            expected = matrix@vects # Expected result from the gpu process\n",
            "            result = gpu_func(vects) # Evaluate kernel on given vectors\n",
            "            if (result==expected).all(): # Check if gpu version gives same result as numpy computation.\n",
            "                print(f'Process {Multiprocessing.get_process_name()}: Test passed!')\n",
            f"        Multiprocessing.process_mp_request(run_parallel,n_processes = {n_processes})\n",
            "\n",
            "    def create_gpu_function(self,matrix,vects):\n",
            "        nq,nvec=vects.shape\n",
            "        # Define Kernel\n",
            '        kernel_str = """\n',
            "            __kernel void\n",
            "            apply_matrix(__global double* out, \n",
            "            __global double* matrix, \n",
            "            __global double* vect, \n",
            "            long nq,long nvec)\n",
            "            {\n",
            "            long i = get_global_id(0); \n",
            "            long j = get_global_id(1);\n",
            "\n",
            "            // Compute application of i'th matrix row on j'th vector\n",
            "            // Store result in value\n",
            "            double value = 0;\n",
            "            for (int q = 0; q < nq; ++q)\n",
            "            {\n",
            "            double matqq = matrix[i*nq + q];\n",
            "            double veciq = vect[q*nvec + j];\n",
            "            value += matqq * veciq;\n",
            "            }\n",
            "\n",
            "            // Write the result vector to device memory\n",
            "            out[i * nvec + j] = value;\n",
            "            }\n",
            '            """\n',
            "        # Define types and input constant arguments\n",
            "        kernel_dict_forward={\n",
            "            'kernel': kernel_str,\n",
            "            'name': 'gpu_func',\n",
            "            'functions': ({\n",
            "                'name': 'apply_matrix',\n",
            "                'dtypes' : (float,float,float,np.int64,np.int64),\n",
            "                'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),\n",
            "                'arg_roles' : ('output','const_input','input','const_input','const_input'),\n",
            "                'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),\n",
            "                'global_range' : (nq,nvec),\n",
            "                'local_range' : None\n",
            "                },)\n",
            "            }\n",
            "        # Create cl Process\n",
            "        cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)\n",
            "        # Register opencl function and get gpu client function\n",
            "        gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)\n",
            "        return gpu_func\n",            
        ]
        save(new_project,code)
        cmd =  f"xframe tmp gpu"
        p = subprocess.run(cmd,shell=True,capture_output=True)
        for process in np.arange(1,n_processes+1):
            assert f'Process {process}: Test passed!' in str(p.stdout)

    
    def setup_home(self):
        print(f'new home at : {tmp_config_path}')
        print(f'given path exists: {os.path.exists(_path)}')
        command = f"xframe --setup_home {tmp_config_path}"
        p = subprocess.Popen(command,shell=True,stdin=subprocess.PIPE)
        p.communicate('y'.encode())[0]

class TestsScripting:
    def test_home_cmd(self):
        xframe.change_home(tmp_config_path)
        home_files_exist()
    def test_project_creation(self,capsys):
        global tmp_config_path
        new_project = tmp_config_path+'projects/tmp/hello.py'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        print('Hello There!')\n",
                ]
        create_path_if_nonexistent(new_project)
        save(new_project,code)
        xframe.reload_home()
        
        xframe.select_project('tmp','hello')        
        xframe.run(update_worker=True)
        captured = capsys.readouterr()
        assert 'Hello There!' in captured.out
    def test_settings(self,capsys):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        settings_path_2 = tmp_config_path+'settings/projects/tmp/hello/opt2.yaml'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import settings\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        opt = settings.project\n",
                "        print(f'Hello {opt.int}!')\n",
                ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        create_path_if_nonexistent(settings_path_2)
        save(settings_path_2,settings)
        xframe.reload_home()
        for i in range(1,3):
            xframe.select_project('tmp','hello',f'opt{i}')
            xframe.run()
            captured = capsys.readouterr()
            assert 'Hello 42!' in str(captured.out)

    def test_database(self,capsys):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        data_path = tmp_config_path+'data/tmp{val}.h5'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import database\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        db = database.project\n",
                "        test_data={'answer':42}\n",
                "        path_modifiers={'val':0}\n",
                "        db.save('my_file',test_data,path_modifiers=path_modifiers)\n",
                "        loaded_file = db.load('my_file',path_modifiers=path_modifiers)\n",
                "        answer = loaded_file['answer']\n",
                "        print(f'Answer = {answer}')\n",
                ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    'IO:\n',
                    '  folders:\n',
                    f'    data: "{os.path.dirname(data_path)}/"\n',
                    '  files:\n',
                    '    my_file:\n',
                    f'     name: "{os.path.basename(data_path)}"\n',
                    '     folder: data\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        xframe.reload_home()
        xframe.select_project('tmp','hello',f'opt1')
        xframe.run()
        captured = capsys.readouterr()
        assert os.path.exists(data_path.format(val=0))==True
        assert 'Answer = 42' in str(captured.out)

    def test_custom_database(self,capsys):
        ## Has to be executed after test_project_creation
        
        new_project = tmp_config_path+'projects/tmp/hello.py'
        settings_path_1 = tmp_config_path+'projects/tmp/settings/hello/opt1.yaml'
        data_path = tmp_config_path+'data/tmp.h5'
        db_path = tmp_config_path+'/projects/tmp/_database_.py'
        code = ['from xframe.interfaces import ProjectWorkerInterface\n',
                'from xframe import database\n',
                '\n',
                'class ProjectWorker(ProjectWorkerInterface):\n',
                '    def run(self):\n',
                "        db = database.project\n",
                "        test_data={'answer':42}\n",
                "        db.save('my_file',test_data)\n",
                "        loaded_file = db.load('my_file')\n",
                "        answer = loaded_file['answer']\n",
                "        yay = loaded_file['new_load']\n",
                "        nay = loaded_file['new_save']\n",
                "        print(f'Answer = {answer} {yay} {nay}')\n",
                ]
        db_code = [
            "from xframe.database.database import DefaultDB\n",
            "from xframe.interfaces import DatabaseInterface\n",
            "\n",
            "class ProjectDB(DefaultDB,DatabaseInterface):\n",
            "    def load_my_file(self,name,**kwargs):\n",
            "        data = self.load_direct(name,**kwargs)\n",
            "        data['new_load'] = 'yay'\n",
            "        return data\n",
            "\n",
            "    def save_my_file(self,name,data,**kwargs):\n",
            "        data['new_save'] = 'ney'\n",
            "        data = self.save_direct(name,data,**kwargs)\n",            
        ]
        settings = ['int:\n',
                    '  command: "int(42)"\n',
                    'IO:\n',
                    '  folders:\n',
                    f'    data: "{os.path.dirname(data_path)}/"\n',
                    '  files:\n',
                    '    my_file:\n',
                    f'     name: "{os.path.basename(data_path)}"\n',
                    '     folder: data\n',
                    ]
        save(new_project,code)
        create_path_if_nonexistent(settings_path_1)
        save(settings_path_1,settings)
        create_path_if_nonexistent(db_path)
        save(db_path,db_code)
        xframe.reload_home()
        xframe.select_project('tmp','hello','opt1')
        xframe.run()
        captured = capsys.readouterr()
        assert 'Answer = 42 yay ney' in str(captured.out)
      
    def test_simple_multiprocessing(self,capsys):
        new_project = tmp_config_path+'projects/tmp/mp.py'
        code = [
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import Multiprocessing\n",
            "import numpy as np\n",
            "\n",
            "def multiply_matrix_with_vectors(vects,matrix,**kwargs):\n",
            "    vects= np.atleast_2d(vects)\n",
            "    new_vects = np.sum(matrix[None,:,:]*vects[:,None,:],axis=2)\n",
            "    return np.squeeze(new_vects)\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "        vectors = np.random.rand(200,10)\n",
            "        matrix = np.random.rand(10,10)\n",
            "\n",
            "        result = Multiprocessing.process_mp_request(multiply_matrix_with_vectors,input_arrays=[vectors],const_inputs = [matrix])\n",
            "\n",
            "        test_result = multiply_matrix_with_vectors(vectors,matrix)\n",
            "        if (result == test_result).all():\n",
            "            print('Test passed!')\n",            
        ]
        save(new_project,code)
        xframe.reload_home()
        xframe.select_project('tmp','mp')
        xframe.run()
        captured = capsys.readouterr()
        assert 'Test passed!' in str(captured.out)        
    def test_GPU(self,capsys):
        new_project = tmp_config_path+'projects/tmp/gpu.py'
        code = [
            "import numpy as np\n",
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import settings\n",
            "from xframe import Multiprocessing\n",
            "import xframe\n",
            "\n",
            "# Ensure a GPU worker is running\n",
            "settings.general.n_control_workers = 1\n",
            "xframe.controller.control_worker.restart_working()\n",
            "\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "\n",
            "        nq = 10 # vector length\n",
            "        nvec=5  # number of vectors\n",
            "        matrix = np.random.rand(nq,nq)\n",
            "        vects= np.random.rand(nq,nvec)\n",
            "        expected = matrix@vects # Expected result from module import symbol the gpu process\n",
            "\n",
            "        gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function\n",
            "\n",
            "        result = gpu_func(vects) # Evaluate kernel on given vectors\n",
            "        if (result==expected).all(): # Check if gpu version gives same result as numpy computation.\n",
            "            print('Test passed!') \n",
            "\n",
            "    def create_gpu_function(self,matrix,vects):\n",
            "        nq,nvec=vects.shape\n",
            "        # Define Kernel\n",
            '        kernel_str = """\n',
            "            __kernel void\n",
            "            apply_matrix(__global double* out, \n",
            "            __global double* matrix, \n",
            "            __global double* vect, \n",
            "            long nq,long nvec)\n",
            "            {\n",
            "            long i = get_global_id(0); \n",
            "            long j = get_global_id(1);\n",
            "\n",
            "            // Compute application of i'th matrix row on j'th vector\n",
            "            // Store result in value\n",
            "            double value = 0;\n",
            "            for (int q = 0; q < nq; ++q)\n",
            "            {\n",
            "            double matqq = matrix[i*nq + q];\n",
            "            double veciq = vect[q*nvec + j];\n",
            "            value += matqq * veciq;\n",
            "            }\n",
            "\n",
            "            // Write the result vector to device memory\n",
            "            out[i * nvec + j] = value;\n",
            "            }\n",
            '            """\n',
            "        # Define types and input constant arguments\n",
            "        kernel_dict_forward={\n",
            "            'kernel': kernel_str,\n",
            "            'name': 'gpu_func',\n",
            "            'functions': ({\n",
            "                'name': 'apply_matrix',\n",
            "                'dtypes' : (float,float,float,np.int64,np.int64),\n",
            "                'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),\n",
            "                'arg_roles' : ('output','const_input','input','const_input','const_input'),\n",
            "                'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),\n",
            "                'global_range' : (nq,nvec),\n",
            "                'local_range' : None\n",
            "                },)\n",
            "            }\n",
            "        # Create cl Process\n",
            "        cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)\n",
            "        # Register opencl function and get gpu client function\n",
            "        gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)\n",
            "        return gpu_func\n",            
        ]
        save(new_project,code)
        xframe.reload_home()
        xframe.select_project('tmp','gpu')
        xframe.run()
        captured = capsys.readouterr()
        assert 'Test passed!' in str(captured.out)
    def test_GPU_in_multiprocessing(self,capsys):
        new_project = tmp_config_path+'projects/tmp/gpu.py'
        n_processes = 5
        code = [
            "import numpy as np\n",
            "from xframe.interfaces import ProjectWorkerInterface\n",
            "from xframe import settings\n",
            "from xframe import Multiprocessing\n",
            "import xframe\n",
            "\n",
            "# Ensure a GPU worker is running\n",
            "settings.general.n_control_workers = 1\n",
            "xframe.controller.control_worker.restart_working()\n",
            "\n",
            "\n",
            "class ProjectWorker(ProjectWorkerInterface):\n",
            "    def run(self):\n",
            "        nq = 10 # vector length\n",
            "        nvec=5  # number of vectors\n",
            "        matrix = np.random.rand(nq,nq)\n",
            "        n_processes = 10\n",
            "        def run_parallel(**kwargs):\n",
            "            vects= np.random.rand(nq,nvec)\n",
            "            gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function\n",
            "            expected = matrix@vects # Expected result from the gpu process\n",
            "            result = gpu_func(vects) # Evaluate kernel on given vectors\n",
            "            if (result==expected).all(): # Check if gpu version gives same result as numpy computation.\n",            
            "                return True\n",
            "            else:\n",
            "                return False\n",
            f"        _out = Multiprocessing.process_mp_request(run_parallel,n_processes = {n_processes})\n",
            "        if all(_out):\n",
            "             print('Test passed!')\n",
            "\n",
            "    def create_gpu_function(self,matrix,vects):\n",
            "        nq,nvec=vects.shape\n",
            "        # Define Kernel\n",
            '        kernel_str = """\n',
            "            __kernel void\n",
            "            apply_matrix(__global double* out, \n",
            "            __global double* matrix, \n",
            "            __global double* vect, \n",
            "            long nq,long nvec)\n",
            "            {\n",
            "            long i = get_global_id(0); \n",
            "            long j = get_global_id(1);\n",
            "\n",
            "            // Compute application of i'th matrix row on j'th vector\n",
            "            // Store result in value\n",
            "            double value = 0;\n",
            "            for (int q = 0; q < nq; ++q)\n",
            "            {\n",
            "            double matqq = matrix[i*nq + q];\n",
            "            double veciq = vect[q*nvec + j];\n",
            "            value += matqq * veciq;\n",
            "            }\n",
            "\n",
            "            // Write the result vector to device memory\n",
            "            out[i * nvec + j] = value;\n",
            "            }\n",
            '            """\n',
            "        # Define types and input constant arguments\n",
            "        kernel_dict_forward={\n",
            "            'kernel': kernel_str,\n",
            "            'name': 'gpu_func',\n",
            "            'functions': ({\n",
            "                'name': 'apply_matrix',\n",
            "                'dtypes' : (float,float,float,np.int64,np.int64),\n",
            "                'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),\n",
            "                'arg_roles' : ('output','const_input','input','const_input','const_input'),\n",
            "                'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),\n",
            "                'global_range' : (nq,nvec),\n",
            "                'local_range' : None\n",
            "                },)\n",
            "            }\n",
            "        # Create cl Process\n",
            "        cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)\n",
            "        # Register opencl function and get gpu client function\n",
            "        gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)\n",
            "        return gpu_func\n",            
        ]
        save(new_project,code)
        xframe.reload_home()
        xframe.select_project('tmp','gpu')
        #with capsys.disabled():
        xframe.run()
        captured = capsys.readouterr()
        assert f'Test passed!' in str(captured.out)


