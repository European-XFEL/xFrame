import ctypes as C 
from ctypes import c_uint,c_double,c_int,c_size_t,c_char,c_void_p
import logging

log=logging.getLogger('root')

#error handler#
gsl_error_handler_t = C.CFUNCTYPE(c_void_p, c_char,c_char,c_int,c_int)

#general Complex#
gsl_complex_data=c_double*2
class gsl_complex(C.Structure):
    _fields_ = [("dat",gsl_complex_data)]



gsl_complex_packed_array = C.POINTER(c_double)


#FFT#
class gsl_fft_real_wavetable(C.Structure):
    _fields_=[
        ("n",c_size_t),
        ("nf",c_size_t),
        ("factor",c_size_t*64),
        ("twiddle",C.POINTER(gsl_complex)*64),
        ("trig",C.POINTER(gsl_complex))
    ]

class gsl_fft_complex_wavetable(C.Structure):
    _fields_=[
        ("n",c_size_t),
        ("nf",c_size_t),
        ("factor",c_size_t*64),
        ("twiddle",C.POINTER(gsl_complex)*64),
        ("trig",C.POINTER(gsl_complex))
    ]

class gsl_fft_halfcomplex_wavetable(C.Structure):
    _fields_=[
        ("n",c_size_t),
        ("nf",c_size_t),
        ("factor",c_size_t*64),
        ("twiddle",C.POINTER(gsl_complex)*64),
        ("trig",C.POINTER(gsl_complex))
    ]

class gsl_fft_real_workspace(C.Structure):
    _fields_=[
        ("n",c_size_t),
        ("scratch",C.POINTER(c_double)),
    ]

class gsl_fft_complex_workspace(C.Structure):
    _fields_=[
        ("n",c_size_t),
        ("scratch",C.POINTER(c_double)),
    ]
    

#DHT#
class gsl_dht_struct(C.Structure):
    _fields_=[
        ('size',c_size_t), #size of sample arrays to be transformed
        ('nu',c_double), #bessel funciton order
        ('xmax',c_double), # realspace cutoff        
        ('kmax',c_double), # reciprocal space cutoff
        ('j',C.POINTER(c_double)), # array of computed j_nu zeros
        ('Jjj',C.POINTER(c_double)), # transform numerator
        ('J2',C.POINTER(c_double))  # transform denominator
    ]
