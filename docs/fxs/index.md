# A workflow for fluctuation x-ray scattering data analysis

The xFrame's `fxs` toolkit provides the following capabilities:

 1. **Calculating cross-correlation functions** from scattering patterns.
 2. **Extracting  invariants** from cross-correlations.
 3. **Reconstruction** of the electron density based on rotational invarants.
 4. **Alignment and Averaging** of reconstructions.

## Installation	
The fxs toolkit comes bundled as part of xFrame. The additional dependencies needed by `fxs` can be automatically installed via:

	pip install 'xframe[fxs]'

The dependencies are as follows:

* essential:
    * [numpy(BSD-3)](https://numpy.org/) 
	* [scipy(BSD-3)](https://scipy.org/) 
	* [h5py(BSD-3)](https://www.h5py.org/)
	* [ruamel.yaml(MIT)](http://yaml.readthedocs.io)
	* [pysofft(GPL3.0)](https://github.com/TimBerberich/pysofft) : 3D rotational alignment
	* [shtns(CeCILL2.1)](https://bitbucket.org/nschaeff/shtns/src/master/) : Spherical harmonic Transforms
	* [pygsl(GPL2.0)](https://github.com/pygsl/pygsl) : Invariant extraction
	* [click(BSD-3)](https://click.palletsprojects.com/en/8.1.x/): Commandline interface
   
* optional:
	* [pyopencl(MIT/X)](https://documen.tician.de/pyopencl/) : GPU Support
    * [vtk(BSD-3)](https://vtk.org/) : 3D visualization
	* [matplotlib(BSD compatible)](https://matplotlib.org/) : 2D visualization
	* [opencv-python(MIT)](https://github.com/opencv/opencv-python) : 2D visualization
	* [flt(MIT)](https://github.com/ntessore/flt) : Invariant extraction
	* [psutil(BSD-3)](https://github.com/giampaolo/psutil): Hardware info

Note that the python packages [pygsl](https://bitbucket.org/nschaeff/shtns/src/master/) and [pyopencl](https://documen.tician.de/pyopencl/) require the libraries [gsl](https://www.gnu.org/software/gsl/) and [opencl](https://www.khronos.org/opencl/) to be installed on your system to function properly.

In order to view the opututed 3d vtk datasets we recomend to install a vtk viewer such as [paraview](https://www.paraview.org/).

## Setup on [Maxwell cluster](https://confluence.desy.de/display/MXW/Maxwell+Cluster/) at Desy
In order to setup xframe on Maxwell you need to follow a few steps (time of writing 9.2023)  

 1. Create a conda environment using the commands
	 ```
	 module load maxwell mamba
	 . mamba-init
	 mamba create --name xframe python=3.8
	 mamba activate xframe
	 ```
	 You can alternatively call the environment by whatever name you like instead of "xframe"
 2. Install GSL
	```
	mamba install -c anaconda gsl
	```
 3. Install xframe
	```
	pip install 'xframe[fxs]'
	```
	
This should have done the trick :)  
If you are still facing issues feel free to create an issue at github.
