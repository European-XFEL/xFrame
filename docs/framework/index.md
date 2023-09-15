# xFrame Handling Details
The idea behind xFrame is to make the creation of scientific algorithms easier by providing a framework that takes care of non-scientific details such as [settings](./settings), [data access](./data_access), [multiprocessing](./multiprocessing) and [GPU access](./gpu_access).
## Installation
xFrame is made available vie PyPi and a basic installation should be as simple as calling:

	pip install xframe
	
The dependencies installed alongside xframe:

* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [h5py](https://www.h5py.org/)
* [ruamel.yaml](http://yaml.readthedocs.io)
* [click](https://click.palletsprojects.com/en/8.1.x/): Commandline interface
