!!! note "Features"
	* Access processed chunks of scattering patterns as (16,512,128) numpy arrays
	* Select patterns by indices or slices of train_ids, cell_ids, pulse_ids and/or frame_ids
	* Get metadata for each pattern (mask, gain, baseline_shift)
	* Filter patterns or mask pixels by gain,lit pixels,mean,max,... 
	* Normalize patterns
	* Specify regions of interest.
	* Not implemented yet! Direct data input to the FXS *project*

!!! warning "Assumes xFrame is running on a Maxwell node" 
## Install 
If you have not installed *xFrame* yet and only want to use it to acces SPB data do the following.
	
1. Create a conda environment using the commands
```
module load maxwell mamba
. mamba-init
mamba create --name xframe python=3.8
mamba activate xframe
```
You can alternatively call the environment by whatever name you like instead of "xframe"
3. Install xframe
```
pip install 'xframe[all]'
```
	
If you already have installed *xFrame* note that the *SPB* experiment requires an additional dependency. In the mamba environment in which *xFrame* is installed you can either call
```console
$ pip install SharedArray
```
or 
```console
$ pip install `xframe[all]`
```
	
## Access data
To get started set a home path for *xFrame*, if you have not done so already, by calling:
```console
$ xframe --setup_home HOME_PATH
```
where `HOME_PATH` is a custom path to where ever you want *xFrame* to store files by default, if no value for `HOME_PATH` is given `~/.xframe` will be used. 

You should now be able to find the tutorial settings file at 
```
HOME_PATH/settings/experiments/SPB/tutorial.yaml
```
it has the following contents:
```yaml linenums="1"
bad_cells:
  command: '[0]' # also accepts slice or numpy arrays e.g. 'np.arange(0,1)'

sample_distance: 285 # in mm
detector_origin: [0,0,217] # In laboratory coordinates [mm]
x_ray_energy: 8000 # In ev

IO:
  folders:
    exp_data_base: '/gpfs/exfel/exp/SPB/202202/p003046/'
```
To continue change the `exp_data_base` value to the path of an experiment you have access to (don't forget the  '/' at the end on the path).  
With this setup you can do the following in python.
```py linenums="1"
import xframe
xframe.select_experiment('SPB','tutorial')
xframe.import_selected_experiment()

exp = xframe.experiment
run=20
selection = exp.DataSelection(20,n_frames=2000)
data_generator = exp.get_data(selection)

for chunk in data_generator:
	print(chunk.keys())
	del(chunk)
```
After calling the `select_experiment` method the experiments settings and database modules can be accessed via
```py
xframe.settings.experiment
xframe.database.experiment
```
As can be seen the result of calling `get_data` is a generator that yields chunks of scattering patterns according to the provided `DataSelection` instance.
??? info "`del(chunk)`"
	Deleting the current chunk at the end of the for loop is important, otherwise you will probably run out of memory.

Metadata such as a list of all `train_ids` in a run can be accessed by calling 
```py linenums="1"
run = 20
meta = xframe.experiment.get_metadata(run)
train_ids = meta['frame_ids']
```

If for example you only want frames from the trains `1474660273` and `1474660274` you may change the above `selection` variable to one of the following:
```py linenums="7"
selection = exp.DataSelection(20,frame_ids=[1474660273,1474660274],frames_mode='exact')
selection = exp.DataSelection(20,frame_ids=np.arange(1474660273,1474660275),frames_mode='exact')
```
Let say you don't care about the precise frame ids but want to select every 2'nd train which lies between the 50'th and 80'th train, in this case your selection could be one of the following.
```py linenums="7"
selection = exp.DataSelection(20,frame_ids=slice(50,80,2),frames_mode='relative')
selection = exp.DataSelection(20,frame_ids=np.arange(50,80,2),frames_mode='relative')
```

## Filters
Filters allow you to calculate a bad frames/paterns mask or modify each pattern on-the-fly. If you want to write custom filters see [Filters](../filters).

If you want to filter you data by number of lit pixels you can add the following to your settings file 
```yaml linenums="1"
```
## Region of interest
To get access to 