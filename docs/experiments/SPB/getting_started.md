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
If you have not installed *xFrame* yet and only want to use it to access SPB data do the following.
	
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
x_ray_energy: 8000 # In eV

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
filter_sequence: [lit_pixels]
filters:
  lit_pixels:
    class: LitPixels
    lit_trheshold: 1 # Pixel value above which it is considered to bit lit
    limits:
      command: [0.1,None] # Lower and upper limit for lit fractions 0.1 = 10%	
```
A mask specifying the patterns which passed the filter can be accessed via the attribute `chunk['filtered_frames_mask']`.

Lets say you want to normalize each pattern before the lit pixel filter is applied, this can be done as follows.
```yaml linenums="1"
filter_sequence: [norm,lit_pixels]
filters:
  lit_pixels:
    class: LitPixels
    lit_trheshold: 1 # Pixel value above which it is considered to bit lit
    limits:
      command: [0.1,None] # Lower and upper limit for lit fractions 0.1 = 10%
  norm:
    class: NormalizationFilter #Normalization by mean value	
```
### Regions of interest
In case you only want to use certain regions within each pattern to compute the filters one can specify regions of interest for each filter by adding an ROIs option as follows

```yaml linenums="1"
filter_sequence: [norm,lit_pixels]
filters:
  lit_pixels:
    class: LitPixels
    lit_trheshold: 1 # Pixel value above which it is considered to bit lit
    limits:
      command: [0.1,None] # Lower and upper limit for lit fractions 0.1 = 10%
  norm:
    class: NormalizationFilter #Normalization by mean value
	ROIs: [all]
```
If more than one names are given the corresponding regions or interest are combined.
ROI's can be assigned in the settings file e.g. as follows
```yaml linenums="1"
ROIs:
  rect1:
    class: Rectangle
	parameters:
	  center: [0.3,0.02]
	  x_len: 0.2
	  y_len: 0.2
  donut:
    class: Annulus
	parameters:
	  center: [0,0]
	  inner_radius: 0.07
	  outer_radius: 0.12
  asic070:
    class: Asic
	parameters:
	  asics: [[0,7,0]] # each entry consists of module_id,asic_id_x [0,..,7],asic_id_y[0,1] 
```
This specifies the ROIs `rect1`,`donut` and `asic070`.
All length values are specified in reciprocal space i.e. in $[\text{Å}^{-1}]$.

## Geometry
Crystfel geometry files (.geom) can be specified in the settings via

```yaml linenums="1"
IO:
  files:
    geometry:
	  name: 'your_file.geom'
	  folder: 'geometry'
```
By default the geometry points to
```
HOME_PATH/data/SPB/geometry/
```
where `HOME PATH` is *xFrames* home directory.
You can also interactively change the detector position by using `xframe.experiment.detector`.

```py linenums="1"
import xframe
xframe.select_experiment('SPB','tutorial')
xframe.import_selected_experiment()

agipd = xframe.experiment.detector
print(f'Detector origin {agipd.origin}[mm])
agipd.pixel_grid
```
The pixel_grid is automatically updated if `xframe.origin` is changed.  
You can also access the individual module plains by
```py linenums="1"
module0 = agipd.modules[0]
print(f'Module 0 origin = {module0.base} [mm]')
print(f'Module 0 x direction = {module0.x_direction} [mm]')
print(f'Module 0 y direction = {module0.x_direction} [mm]')
module0.base+=np.array([2,0,0]) #shift by 2mm in x direction
agipd.assemblePixelGrid()
```
The call to `agipd.assemblePixelGrid` is necessary since the class currently can not auto detect changes to its modules. 
