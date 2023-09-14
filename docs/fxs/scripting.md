# Scripting
Apart from the command line interface all routines implemented in the `fxs` toolkit can also be accessed interactively. 

In fact executing the command 
```console
$ xframe fxs reconstruct tutorial
```
is equivalent to the following python code
```python
import xframe
xframe.select_project('fxs','reconstruct','tutorial')
xframe.run()
```
Calling `xframe.select_project('fxs','reconstruct','tutorial')` will load the settings file and setup the database module. This allows you to change settings and look at data that is used by `fxs` in a convenient way.  

## Modify Settings
As an example lets take another look at the tutorial settings file for the `correlate` worker.


```yaml linenums="1"
structure_name: 'tutorial'

image_dimensions: [512,512]

n_processes: 32

max_n_patterns: 2000


compute: ['is_good','waxs_aver','ccf_q1q2']
pixel_size: 200.0                 #Detector pixel size in microns
sample_distance: 800.0            #Detector to sample distance in [mm]
wavelength: 1.23984               #X-ray wavelength in Å
detector_origin: [255.2,255.5]    #2D detector origin in pixel coordinates

interpolation_order: 2            #Cartesian to Polar spline interpolation order

phi_range:                        #Azimutal angle grid via (start,stop,npoints,???)
    command: '(0.0, 2*np.pi, 1024, "exact")'
fc_n_max: 70                      #maximal resolved harmonic order


IO:
  folders:
    home:
      command: 'xframe_opt.IO.folders.home'
    in_base:
      home: 'data/fxs/input_files/'
  files:
    input_file_list:
      name: patterns_list.txt
      folder: in_base
```

In python you have access to these settings via the `xframe.settings.project` attribute.
All settings files in *xFrame* support `.` notation to access its attributes, as well as most of the features 
of a python dictionary, here is an example.
```python
import xframe
xframe.setup_project('fxs',correlate','tutorial')
 
opt = xframe.settings.project

# Looking up properties in the settings file
n_processes_1 = opt.n_processes
n_processes_2 = opt['n_processes']
non_existing = opt.get('non_existing_settings',False)

#Changing settings on the fly
# via . notation
opt.n_processes = 10
# or like you would use a dictionary
opt['n_processes'] = 10

# Running the script with modified settings
xframe.run()
```
More over the `xfrmae.settings.project` contains the parsed version of the sittings file, that is all `command:` lines are replace with the output of `eval()` applied to their string and all default settings have been included.

The settings file to be saved alongside the routine results will correspond to the state of `xframe.settings.project` at the moment `xframe.run()` is called.

<!--??? info "Settings objects are instances `SimpleNamespace` + modifications"
	Under the hood settings objects are instances of a custom class `DictNamespace` that inherits from the standard `SimpleNamespace` with custom implementations of the usal methods available for dictionaries. (The class`s implementation is in `xframe.libraries.pythonLibrary.DictNamespace`).
	If you want a pure dictionary copy of the settings object just call `xframe.settings.project.dict()`, however changing this copy will have no effect on the used settings.
	Another asside is that '.' notation is particularly usefull in interactive mode since there it supports autocompletion.-->

## Accessing project files 
You can also us the `fxs` database module stored in `xframe.database.project` after the call to `select_project`.
This allows you easy access to all files defined in the `IO:` section of your settings file.

If, for example, you want to look at the `patterns_list.txt` file from the above settings you can use its name `input_file_list` to get access to it, e.g.

```python
import xframe
xframe.setup_project('fxs','correlate','tutorial')
db = xframe.database.project

# loading the patterns_list.txt 
file_list = db.load('input_file_list')

# Overriding the patterns_list.txt
db.save('input_file_list',file_list)
```

In fact, all file access in `fxs`'s routines is managed via the entries in the `IO:` portion of the settings files, they are not explicitly listed in the `tutorial.yaml` of the [getting started](../getting_started/) page files to keep the tutorials simple. 

More information on xFrame`s handling of settings and data access can be found under [Settings](../../framework/#settings) 







