# Getting Started
The follwing tutorials assume you are using a Linux operating system.  
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
Every subfolder of the 'projects' folder will be recognized as a *project* by xFrame.
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
	If you like to keep your settings separate from your actual project you can also create the settingsfile at
	```	
	HOME_PATH/settings/projects/tmp/hello/set123.yaml
	```	
	In case both files exist the one in `HOME_PATH/settings` is usesed preferentially.
	

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
# Note: Contents from here on are outdated and will be updated shortly !! 
## Data Access
Let say we want to save or load some data for this we can use xframe.database.analysis module.
To see how this might be done consider the following change to our `hallo.py` file:

	from xframe.analysis.analysisWorker import AnalysisWorker
	from xframe.settings import analysis as opt
	from xframe.database import analysis
	import numpy as np
	
	
	class Worker(AnalysisWorker):
		def start_working(self):
			print(f'Hello {opt.name}, your random number is: {opt.random_number}')
			
			data = {'name':opt.name,'random_number':opt.random_number,'data':np.arange(10,dtype=float)}			
			path = '~/.xframe/data/hello_dat.h5'
			db.save(path,data)
			data2 = db.load(path)
			print(f'Loaded data = {data2}')

			
Upon execution this gives:

	$ xframe tutorial.hello set123
	 ------- Start <tutorial.hello.Worker object at 0x7fb101b4efd0> ------- 

	Hello Pi, your random number is: 0.5233796142886044
	Loaded data = {'data': array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]), 'name': 'Pi', 'random_number': 0.5233796142886044}
	
	 ------- Finished <tutorial.hello.Worker object at 0x7fb101b4efd0> -------

#### Integration with settings file
Always having to keep track of your file paths in code, as above, is cumbersome and does not scale well when trying to manage a whole suite of different files and folders.
A solution is to let your settings file handle the file and folder locations. 
To achive this lets modify our `set123.yaml` :

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
				name: hello_dat.h5
				folder: data
				

This allows us to save our dataset using the alias `'my_data'` using `db.save('my_data',data)`, i.e.:

	from xframe.analysis.analysisWorker import AnalysisWorker
	from xframe.settings import analysis as opt
	from xframe.database import analysis
	import numpy as np
	
	
	class Worker(AnalysisWorker):
		def start_working(self):
			print(f'Hello {opt.name}, your random number is: {opt.random_number}')

	        data = {'name':opt.name,'random_number':opt.random_number,'data':np.arange(10,dtype=float)}	
			db.save('my_data',data)
			data2 = db.load('my_data')
			print(f'Loaded data = {data2}')

Note that, changing the base location of our system of folders and potentially files can now simply be done by changing path stored in the `base:` folder of the settings file. All other deriving folders will be automatically updated.
### Dynamic paths (i.e. *path_modifiers*)
A way to allow paths to adjust dynamicall during runtime is to use *path_modifiers*, which is a fancy way of saying thet the database class supports string formating of filepaths. Consider the following settings file

```yaml
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

The location and file name of 'my_data' can now be dynamically ajusted by specifying the optional argument path_modifyers as follows:

```py
db.save('my_data',data,path_modifiers={'data_number':1})
```	
which would save `data` to `"~/.xframe/data_2/hello_dat_2.h5"`.
 
	
### Custom behaviour
To allow for easy customization of the data access xFrame provides the possibility to easily write your own database routines and intercept calls to `db.load` and `db.save`.

To do so lets create the file:

	~/.xframe/projects/tutorial/database.py
	
with contents:

```py
from xframe.database.database import default_DB
from xframe.analysis.interfaces import DatabaseInterface as DatabaseInterfaceAnalysis

class Analysis_DB(default_DB,DatabaseInterfaceAnalysis):
	def load_my_data(self,name,**kwargs):
		data = self.load_direct(name,**kwargs)			
		data['name'] = 'Phi'			
		return data
		
	def save_my_data(self,name,data,**kwargs):			
		if data['name'] == 'Phi':
			data['name'] = 'Pi'			
		data = self.save_direct(name,data,**kwargs)
```
			
Upon project creation xFrame will search for a `database.py` file in our project folder that contains an `Analysis_DB` class. If it is found xFrame replaces the default database (`default_DB`) module with the one provided by the project.
Letting our custom `Analysis_DB` inherit from `default_DB` keeps all aformentioned proerties of the db class.
Additionaly we can now make use of the feature, that whenever `db.load('my_data')` or `db.save('my_data',data)` is called, the `db` class first searches for methods named `db.load_my_data` or `db.save_my_data`. If it finds these methods further processing of the request is redirected to these methods.

The above example shows how these functions might be implemented.
Here we use the custom loader to change our name from 'Pi' to 'Phi' and vise versa in the saving routine.
Note that inside of these functions calls to `super().load` or `super().save` should be avoided since these would cause infinite recursion. Instead use `super().load_direct` and `super().save_direct` which skip the search for custom methods. 

## Multiprocessing
Currently xFrame only supports multiprocessing on a single node using the multiprocessing library of python.
xFrame handels multiprocessing via the `xframe.Multiprocessing` module.

As an example create the file `~/.xframe/projects/tutorial/mp.py` with contents:

    from xframe.analysis.analysisWorker import AnalysisWorker
    from xframe import Multiprocessing
    import numpy as np
    
    def multiply_matrix_with_vectors(vects,matrix,**kwargs):
        vects= np.atleast_2d(vects)
        new_vects = np.sum(matrix[None,:,:]*vects[:,None,:],axis=2)
        return np.squeeze(new_vects)
    	
    class Worker(AnalysisWorker):
        def start_working(self):
            vectors = np.random.rand(200,10)
            matrix = np.random.rand(10,10)
            result = Multiprocessing.process_mp_request(multiply_matrix_with_vectors,input_arrays=[vectors],const_inputs = [matrix])
            test_result = multiply_matrix_with_vectors(vectors,matrix)
            if (result == test_result).all():
                print('Test passed!')
