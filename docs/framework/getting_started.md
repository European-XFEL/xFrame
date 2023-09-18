# Getting Started
The following tutorials assume you are using a Linux operating system.  
First lets create the home folder for *xFrame* by calling
```console
$ xframe --setup_home HOME_PATH
```
and substituting `HOME_PATH` with where ever you want *xFrame* to store files and lookup projects by default. If no value for `HOME_PATH` is given `~/.xframe` will be used. 


## Project
To create a project simply create the folder
```
HOME_PATH/projects/tmp
```
Every sub folder of the 'projects' folder will be recognized as a *project* by xFrame.
You can also configure *xFrame* to search other locations for *projects* more on that can be found in [xFrame Settings](#xframe-settings).
Now, let us bring life to our first project by creating the file 
```
HOME_PATH/projects/tmp/hello.py
```
with the following content
```py linenums="1" 
from xframe.interfaces import ProjectWorkerInterface	
	
class ProjectWorker(ProjectWorkerInterface):
def run(self):
	print('Hello There!')
```
Going back to your command line we can now do the following
```console
$ xframe tmp hello
	
 ------- Start <tutorial.hello.Worker object at 0x7f54c34d5050> ------- 
	
Hello There!

 ------- Finished <tutorial.hello.Worker object at 0x7f54c34d5050> ------- 
```	
## Settings
Now lets add some settings to our project. For that create the following file (and its sub folders.)
```	
HOME_PATH/projects/tmp/settings/hello/set123.yaml
```	
with contents
```yaml linenums="1" 
name: Pi
random_number:
	command: 'np.random.rand()'
```	
Note that the whatever string is placed behind a `command:` field will be executed and stored in the settings name above it, in this case `random_number:`.

??? info "Other locations for settings files"
	If you like to keep your settings separate from your actual project you can also create the settings file at
	```	
	HOME_PATH/settings/projects/tmp/hello/set123.yaml
	```	
	In case both files exist the one in `HOME_PATH/settings` is used preferentially.
	

We can now use these settings by importing settings.analysis from xframe as follows.

```py linenums="1" 
from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
	
class ProjectWorker(ProjectWorkerInterface):
	def run(self):
		opt = settings.project
		print(f'Hello {opt.name}, your random number is: {opt.random_number}')
```

When executing the xframe command we can tell the project which settings file to use by simply appending the name of the settings file as follows:
```console
$ xframe tmp hello set123

 ------- Start <tutorial.hello.Worker object at 0x7fc352b05090> ------- 
	
Hello Pi, your random number is: 0.9360428946014102

 ------- Finished <tutorial.hello.Worker object at 0x7fc352b05090> -------
```
xFrame also allows for the creation of default settings for this and further details view [Settings](#settings_1).

## Data Access
Lets assume we want to save or load some data.
In this case we can use the xframe.database.project module.
To see how this might be done consider the following change to our `hello.py` file:

```py linenums="1" 
from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
from xframe import database	
import numpy as np

class ProjectWorker(ProjectWorkerInterface):
	def run(self):
		opt = settings.project
		db  = database.project
		print(f'Hello {opt.name}, your random number is: {opt.random_number}')
		
		data = {'name':opt.name,'random_number':opt.random_number,'data':np.arange(10,dtype=float)}			
		path = '~/.xframe/data/hello_data.h5'
		db.save(path,data)
		data2 = db.load(path)
		print(f'Loaded data = {data2}')
```

Upon execution this gives:

```console
$ xframe tutorial.hello set123
 ------- Start <tutorial.hello.Worker object at 0x7fb101b4efd0> ------- 
 
Hello Pi, your random number is: 0.5233796142886044
Loaded data = {'data': array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]), 'name': 'Pi', 'random_number': 0.5233796142886044}

 ------- Finished <tutorial.hello.Worker object at 0x7fb101b4efd0> -------
```	

The database class infers the file type from its path. Currently supported file types are:

 * '.h5' (*h5py*),
 * '.vts' (spherical/polar grid only saving, *vtk*)
 * '.vtr' (cartesian grid only saving, *vtk*)
 * '.npy' (*numpy*)
 * '.raw'
 * '.txt'
 * '.yaml' (*ruamel.yaml*)
 * '.matplotlib' (will be saved as '.png', *matplotlib*)
 * '.cv' (will be saved as png and loaded as numpy array, *pyopencv*)
 * '.py' (treated as text files)


### Integration with settings file

Always having to keep track of your file paths in code, as above, is cumbersome and does not scale well when trying to manage a whole suite of different files and folders.
*xFrame* provides a solution using the sittings file.  You are able to add files and folders by modifying  our `set123.yaml` file as follows :

```yaml linenums="1" 
name: Pi
random_number:
	command: 'np.random.rand()'
IO:
	folders:
		base: ~/.xframe/
		data:
			base: data/
	files:
		my_data:
			name: hello_data.h5
			folder: data				
```

This allows us to save our dataset using the alias `'my_data'` as follows:

```py linenums="1" 
from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
from xframe import database	
import numpy as np

class ProjectWorker(ProjectWorkerInterface):
	def run(self):
		opt = settings.project
		db  = database.project
		print(f'Hello {opt.name}, your random number is: {opt.random_number}')	
		
		data = {'name':opt.name,'random_number':opt.random_number,'data':np.arange(10,dtype=float)}			
		db.save('my_data',data)
		data2 = db.load('my_data')
		print(f'Loaded data = {data2}')
```

Note that, changing the base location of our system of folders and potentially files can now simply be done by changing path stored in the `base:` folder of the settings file. All other deriving folders will be automatically updated.
### Dynamic paths (i.e. *path_modifiers*)
A way to allow paths to adjust dynamically during runtime is to use *path_modifiers*, which is a fancy way of saying that the database class supports string formating of file paths. Consider the following settings file

```yaml linenums="1" 
name: Pi
random_number:
	command: 'np.random.rand()'

IO:
	folders:
		base: ~/.xframe/
		data:
			base: data_{data_number}/
	files:
		my_data:
			name: hello_dat_{data_number}.h5
			folder: data
```
The location and file name of 'my_data' can now be dynamically adjusted by specifying the optional argument path_modifiers as follows:

```py
db.save('my_data',data,path_modifiers={'data_number':2})
```	
which would save `data` to `"~/.xframe/data_2/hello_dat_2.h5"`.
 
	
### Custom behavior
*xFrame* allows to easily customize the behaviors of the databases `load` and `save` routines fore each specified file. To do so lets create the file:

```	
HOME_PATH/projects/temp/_database_.py
```	

with contents:

```py linenums="1"
from xframe.database.database import DefaultDB
from xframe.interfaces import DatabaseInterface 

class ProjectDB(DefaultDB,DatabaseInterface):
	def load_my_data(self,name,**kwargs):
		data = self.load_direct(name,**kwargs)			
		data['name'] = 'Phi'			
		return data
		
	def save_my_data(self,name,data,**kwargs):			
		if data['name'] == 'Phi':
			data['name'] = 'Pi'			
		data = self.save_direct(name,data,**kwargs)
```
			
Upon project creation xFrame will search for a `_database_.py` file in our project folder that contains an `ProjectDB` class. If it is found xFrame replaces the default database (`DefaultDB`) class with the one provided by the project.
Letting our custom `ProjectDB` inherit from `DefaultDB` keeps all aforementioned properties of the db class.
Additionally we can now make use of the feature, that whenever `db.load('my_data')` or `db.save('my_data',data)` is called, the `db` class first searches for methods named `db.load_my_data` or `db.save_my_data`. If it finds those methods further processing is redirected to these methods.

The above example shows how these functions might be implemented.
Here we use the custom loader to change our name from 'Pi' to 'Phi' and vise versa in the saving routine.
Note that inside of these functions calls to `super().load` or `super().save` should be avoided since these would cause infinite recursion. Instead use `super().load_direct` and `super().save_direct` which skips the search for custom methods. 

## Multiprocessing
Currently *xFrame* supports multiprocessing via the multiprocessing library of python.
The logic handling multiprocessing is contained in `xframe.Multiprocessing`.
In order to make monitoring of running processes easy the following restriction is enforced.

* Only the master process is allowed to spawn child processes.

As an example create the file `~/.xframe/projects/tutorial/mp.py` with contents:

```py linenums="1"
    from xframe.interfaces import ProjectWorkerInterface
    from xframe import Multiprocessing
    import numpy as np
    
    def multiply_matrix_with_vectors(vects,matrix,**kwargs):
        vects= np.atleast_2d(vects)
        new_vects = np.sum(matrix[None,:,:]*vects[:,None,:],axis=2)
        return np.squeeze(new_vects)
    	
    class ProjectWorker(ProjectWorkerInterface):
        def run(self):
            vectors = np.random.rand(200,10)
            matrix = np.random.rand(10,10)
			
            result = Multiprocessing.process_mp_request(multiply_matrix_with_vectors,input_arrays=[vectors],const_inputs = [matrix])
			
            test_result = multiply_matrix_with_vectors(vectors,matrix)
            if (result == test_result).all():
                print('Test passed!')
```

??? info "Why not just use `multiprocessing.pool`"
	Under the hood `Multiprocessing.process_mp_request` uses `multiprocessing.Process` because of its increased flexibility. Our custom method additionally it supports
	
	* starting of daemon processing, 
	* automatic creation of communication queues, 
	* a method for syncing workers, estimation of available memory for each worker, 
	* different modes of splitting input data, 
	* automatic assembly of output data into numpy arrays, 
	* usage of posix shared memory using the package SharedMemory etc. 
		
	For more details see [Multiprocessing](../multiprocessing)

## GPU Access

*xFrame* allows tries to define GPU kernels as intuitive as possible wile at the same time allowing their execution in a multiprocess environment. The central limitation which prevents context corruption in child processes is that:

* Only child processes are allowed to load opencl Platforms.

In the following we demonstrate the creation of a GPU kernel that applies a matrix to a set of vectors.

```py linenums="1"
import numpy as np
from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
from xframe import Multiprocessing
import xframe

# Ensure a GPU worker is running
settings.general.n_control_workers = 1
xframe.controller.control_worker.restart_working()


class ProjectWorker(ProjectWorkerInterface):
	def run(self):
				
		nq = 10 # vector length
		nvec=5  # number of vectors
		matrix = np.random.rand(nq,nq)
		vects= np.random.rand(nq,nvec)
		expected = matrix@vects # Expected result from the gpu process
		
		gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function
		
		result = gpu_func(vects) # Evaluate kernel on given vectors
		if (result==expected).all(): # Check if gpu version gives same result as numpy computation.
			print('Test passed!') 
			
	def create_gpu_function(self,matrix,vects):
		nq,nvec=vects.shape
		# Define Kernel
		kernel_str = """
			__kernel void
			apply_matrix(__global double* out, 
			__global double* matrix, 
			__global double* vect, 
			long nq,long nvec)
			{
			long i = get_global_id(0); 
			long j = get_global_id(1);
				
			// Compute application of i'th matrix row on j'th vector
			// Store result in value
			double value = 0;
			for (int q = 0; q < nq; ++q)
			{
			double matqq = matrix[i*nq + q];
			double veciq = vect[q*nvec + j];
			value += matqq * veciq;
			}
		
			// Write the result vector to device memory
			out[i * nvec + j] = value;
			}
			"""
		# Define types and input constant arguments
		kernel_dict_forward={
			'kernel': kernel_str,
			'name': 'gpu_func',
			'functions': ({
				'name': 'apply_matrix',
				'dtypes' : (float,float,float,np.int64,np.int64),
				'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),
				'arg_roles' : ('output','const_input','input','const_input','const_input'),
				'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),
				'global_range' : (nq,nvec),
				'local_range' : None
				},)
			}
		# Create cl Process
		cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
		# Register opencl function and get gpu client function
		gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)
		return gpu_func
```

Last but not least we may use the gpu_function from above in a multiprocess environment as follows.

```py linenums="1"
import numpy as np
from xframe.interfaces import ProjectWorkerInterface
from xframe import settings
from xframe import Multiprocessing
import xframe

# Ensure a GPU worker is running
settings.general.n_control_workers = 1
xframe.controller.control_worker.restart_working()


class ProjectWorker(ProjectWorkerInterface):
	def run(self):
				
		nq = 10 # vector length
		nvec=5  # number of vectors
		matrix = np.random.rand(nq,nq)
		n_processes = 10
		def run_parallel(**kwargs):
			vects= np.random.rand(nq,nvec)                    
			gpu_func = self.create_gpu_function(matrix,vects) # Create GPU Kernel function
			expected = matrix@vects # Expected result from the gpu process
			result = gpu_func(vects) # Evaluate kernel on given vectors
			if (result==expected).all(): # Check if gpu version gives same result as numpy computation.			
				print(f'Process {Multiprocessing.get_process_name()}: Test passed!')
		Multiprocessing.process_mp_request(run_parallel,n_processes = 10)

	
	def create_gpu_function(self,matrix,vects):
		nq,nvec=vects.shape
		# Define Kernel
		kernel_str = """
			__kernel void
			apply_matrix(__global double* out, 
			__global double* matrix, 
			__global double* vect, 
			long nq,long nvec)
			{
			long i = get_global_id(0); 
			long j = get_global_id(1);
				
			// Compute application of i'th matrix row on j'th vector
			// Store result in value
			double value = 0;
			for (int q = 0; q < nq; ++q)
			{
			double matqq = matrix[i*nq + q];
			double veciq = vect[q*nvec + j];
			value += matqq * veciq;
			}
		
			// Write the result vector to device memory
			out[i * nvec + j] = value;
			}
			"""
		# Define types and input constant arguments
		kernel_dict_forward={
			'kernel': kernel_str,
			'name': 'gpu_func',
			'functions': ({
				'name': 'apply_matrix',
				'dtypes' : (float,float,float,np.int64,np.int64),
				'shapes' : ((nq,nvec),matrix.shape,(nq,nvec),None,None,None),
				'arg_roles' : ('output','const_input','input','const_input','const_input'),
				'const_inputs' : (None,matrix,None,np.int64(nq),np.int64(nvec)),
				'global_range' : (nq,nvec),
				'local_range' : None
				},)
			}
		# Create cl Process
		cl_process =  Multiprocessing.openCL_plugin.ClProcess(kernel_dict_forward)
		# Register opencl process and get gpu client function
		gpu_func = Multiprocessing.comm_module.add_gpu_process(cl_process)
		return gpu_func

```
