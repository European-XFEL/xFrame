![](https://raw.githubusercontent.com/European-XFEL/xframe/main/docs/images/xFrame_logo_title.svg "xFrame logo")
[![Lates Release](https://img.shields.io/github/v/release/European-XFEL/xframe)](https://github.com/European-XFEL/xframe/releases)
![License](https://img.shields.io/github/license/European-XFEL/xframe)
[![Documentation](https://img.shields.io/readthedocs/extra-foam)](https://xframe-fxs.readthedocs.io/en/latest/)
[![Documentation](https://img.shields.io/badge/documentation-online-blue)](https://xframe-fxs.readthedocs.io/en/latest/)
![Language](https://img.shields.io/badge/language-python-blue)
# Introduction
The main objective of __xFrame__ is to provide an easy and flexible framwork that handles details such as
- Data storage / File access
- Settings management
- Multiprocessing
- GPU access

Allowing users to focus on the "scientific" part of their algorithm.  
*xFrame* has been created during our quest of writing a toolkit for fluctuation X-ray scattering.
To install *xFrame* simply call
```
pip install xframe
```
For more information and tutorials visit the documentaiton at [xframe-fxs.readthedocs.io](https://xframe-fxs.readthedocs.io)

# Fluctuation X-Ray Scattering Toolkit [fxs]
The project __fxs__ comes bundled with *xFrame* and provides a workflow for 3D structure determination from fluctuation X-ray scattering data.

Additional dependencies needed by *fxs* can be installed via
```
pip install 'xframe[fxs]'
```
please make sure that the following libraries are installed on your system before calling `pip install` 
- [GNU Scientific Library](https://www.gnu.org/software/gsl/)
- [OpenCL](https://www.khronos.org/opencl/)

Further information as well as tutorials can be found at [xframe-fxs.readthedocs.io/en/latest/fxs](https://xframe-fxs.readthedocs.io/en/latest/fxs)

# Citation
If you find this piece of software usefull for your scientific project consider citing the paper  
Still in review ... :)
