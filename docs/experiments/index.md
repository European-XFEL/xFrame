# Experiments


For accessing data from the SPB experiment at European XFEL see [SPB at EuXFEL](SPB/getting_started.md).  

!!! warning "Unfinished Feature" 
	Note that everything mentioned on this site is still very preliminary and not fully fleshed out.
	
??? info "For developers"
	The purpose of *experiments* in *xFrame is to provide access to XFEL scattering data.
	Conceptually an *experiment* together with its settings file should contain all information necessary to create scattering patterns on a grid in reciprocal space for a given real experimental setup.  
	Such data includes, e.g.:

	- X-ray wavelength, detector geometry, distance between sample and detector, detector mask, background to subtract,  normalization range, ...

	A new *experiment* can be created as follows.  

	1. Choose a `HOME_FOLDER` as described in the [Framework](../framework/getting_started) section.
	2. Create a custom named folder
	```console
	HOME_PATH/experiments/my_experiment
	```
	3. place an `experiments.py` file in this folder with the following contents:

	```py linenums='1'
	from xframe.interfaces import ExperimentWorkerInterface
	from typing import Generator
	
	class DataSelection:
		def __init__(_dict:dict):
			pass

	class ExperimentWorker(ExperimentWorkerInterface):
		DataSelection = DataSelection
		def get_data(self,selection : DataSelection) -> Generator[int,None,None]:
			pass
		def get_reciprocal_pixel_grid(self) -> dict:
			pass
		def get_geometry(self) -> dict:
			pass
		def run(self):
			pass
	```

	`DataSelection` functions as a datastructure that contains all information needed to select a pice of scattering patterns (i.e. pulse_numbers, train_ids ....).  
	`get_data` takes a `DataSelection` object and returns a generator that retrieves chunks of processed scattering patterns.  
	`get_reciprocal_pixel_grid` returns the detector pixel coordinates in reciprocal space.  
	`get_geometry` returns real space information about the detector geometry.  

	__Connection to *projects*__  
	If an experiment has been given, either by calling 
	```py linenums='1'
	import xframe
	xframe.select_experiment('my_experiment','default')
	xframe.import_selected_experiment()
	```
	or by specifying the following command line arguments when a project is started,
	```console
	$ xframe -e my_experiment -eopt default
	```
	one can access an instance of the above defined ExperimentWorker via
	```
	xframe.experiment
	```
	In the above expamle `default` is the name of a settings file placed in either of the following paths  
	`HOME_PATH/settings/experiments/my_experiment/default.yaml`  
	`HOME_PATH/experiments/my_experiment/settings/default.yaml`

