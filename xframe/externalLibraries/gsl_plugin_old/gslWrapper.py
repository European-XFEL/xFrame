import os
file_path = os.path.realpath(__file__)
externalLibraries_dir = os.path.dirname(file_path)
import numpy as np
import ctypes as C 
from ctypes import c_uint,c_double,c_int,c_size_t,c_void_p
import logging

from xframe.library import pythonLibrary as pyLib
from xframe.externalLibraries.gsl_plugin import gslStructures as gslStruct

log=logging.getLogger('root')

cblas = pyLib.load_clib('gslcblas',mode=C.RTLD_GLOBAL)
print("gsl file path = {}".format(file_path))
gsl = pyLib.load_clib('gsl')

try:
    dht = pyLib.load_clib(externalLibraries_dir+'/dht.so')
    dht_found = True
except OSError as e:
    dht_found = False
    log.info('Could not load custom discrete hankel transform Library "dht.so" . Skipping dht')
    
class Wraper:
    def __init__(self):
        def setTypes(function,restype,argtype):
            function.restype=restype
            if isinstance(argtype,list):
                function.argtypes=argtype
            else:
                function.argtype=argtype
            return function


        #error handling#
        gsl_set_error_handler_off=setTypes(gsl.gsl_set_error_handler_off,gslStruct.gsl_error_handler_t,c_void_p)
        
        #defining input and output types for all used gsl functions#
        #FFT mixed Radix#
        #complex
        gsl_fft_complex_wavetable_alloc=gsl.gsl_fft_complex_wavetable_alloc
        gsl_fft_complex_wavetable_alloc.restype=C.POINTER(gslStruct.gsl_fft_complex_wavetable)
        gsl_fft_complex_wavetable_alloc.argtype=c_size_t

        gsl_fft_complex_wavetable_free=gsl.gsl_fft_complex_wavetable_free
        gsl_fft_complex_wavetable_free.restype=None
        gsl_fft_complex_wavetable_free.argtype=C.POINTER(gslStruct.gsl_fft_complex_wavetable)

        gsl_fft_complex_workspace_alloc=gsl.gsl_fft_complex_workspace_alloc
        gsl_fft_complex_workspace_alloc.restype=C.POINTER(gslStruct.gsl_fft_complex_workspace)
        gsl_fft_complex_workspace_alloc.argtype=c_size_t

        gsl_fft_complex_workspace_free=gsl.gsl_fft_complex_workspace_free
        gsl_fft_complex_workspace_free.restype=None
        gsl_fft_complex_workspace_free.argtype=C.POINTER(gslStruct.gsl_fft_complex_workspace)

        gsl_fft_complex_forward=gsl.gsl_fft_complex_forward
        gsl_fft_complex_forward.restype=c_int
        gsl_fft_complex_forward.argtype=[gslStruct.gsl_complex_packed_array, c_size_t, c_size_t, C.POINTER(gslStruct.gsl_fft_complex_wavetable),C.POINTER(gslStruct.gsl_fft_complex_workspace)]

        gsl_fft_complex_inverse=gsl.gsl_fft_complex_inverse
        gsl_fft_complex_inverse.restype=c_int
        gsl_fft_complex_inverse.argtype=[gslStruct.gsl_complex_packed_array, c_size_t, c_size_t, C.POINTER(gslStruct.gsl_fft_complex_wavetable),C.POINTER(gslStruct.gsl_fft_complex_workspace)]

        #real
        gsl_fft_real_wavetable_alloc=gsl.gsl_fft_real_wavetable_alloc
        gsl_fft_real_wavetable_alloc.restype=C.POINTER(gslStruct.gsl_fft_real_wavetable)
        gsl_fft_real_wavetable_alloc.argtype=c_size_t

        gsl_fft_real_wavetable_free=gsl.gsl_fft_real_wavetable_free
        gsl_fft_real_wavetable_free.restype=None
        gsl_fft_real_wavetable_free.argtype=C.POINTER(gslStruct.gsl_fft_real_wavetable)

        gsl_fft_real_workspace_alloc=gsl.gsl_fft_real_workspace_alloc
        gsl_fft_real_workspace_alloc.restype=C.POINTER(gslStruct.gsl_fft_real_workspace)
        gsl_fft_real_workspace_alloc.argtype=c_size_t

        gsl_fft_real_workspace_free=gsl.gsl_fft_real_workspace_free
        gsl_fft_real_workspace_free.restype=None
        gsl_fft_real_workspace_free.argtype=C.POINTER(gslStruct.gsl_fft_real_workspace)

        gsl_fft_real_unpack=gsl.gsl_fft_real_unpack
        gsl_fft_real_unpack.restype=c_int
        gsl_fft_real_unpack.argtype=[C.POINTER(c_double),gslStruct.gsl_complex_packed_array,c_size_t, c_size_t]

        gsl_fft_real_forward=gsl.gsl_fft_real_transform
        gsl_fft_real_forward.restype=c_int
        gsl_fft_real_forward.argtype=[C.POINTER(c_double), c_size_t, c_size_t, C.POINTER(gslStruct.gsl_fft_real_wavetable),C.POINTER(gslStruct.gsl_fft_real_workspace)]
        #halfcomplex
        gsl_fft_halfcomplex_wavetable_alloc=gsl.gsl_fft_halfcomplex_wavetable_alloc
        gsl_fft_halfcomplex_wavetable_alloc.restype=C.POINTER(gslStruct.gsl_fft_halfcomplex_wavetable)
        gsl_fft_halfcomplex_wavetable_alloc.argtype=c_size_t

        gsl_fft_halfcomplex_wavetable_free=gsl.gsl_fft_halfcomplex_wavetable_free
        gsl_fft_halfcomplex_wavetable_free.restype=None
        gsl_fft_halfcomplex_wavetable_free.argtype=C.POINTER(gslStruct.gsl_fft_halfcomplex_wavetable)

        gsl_fft_halfcomplex_unpack=gsl.gsl_fft_halfcomplex_unpack
        gsl_fft_halfcomplex_unpack.restype=c_int
        gsl_fft_halfcomplex_unpack.argtype=[C.POINTER(c_double),C.POINTER(c_double),c_size_t,c_size_t]
        
        gsl_fft_halfcomplex_inverse=gsl.gsl_fft_halfcomplex_inverse
        gsl_fft_halfcomplex_inverse.restype=c_int
        gsl_fft_halfcomplex_inverse.argtype=[C.POINTER(c_double), c_size_t, c_size_t, C.POINTER(gslStruct.gsl_fft_halfcomplex_wavetable),C.POINTER(gslStruct.gsl_fft_real_workspace)]

        
        #BesselZero#
        gsl_sf_bessel_zero_Jnu=gsl.gsl_sf_bessel_zero_Jnu
        gsl_sf_bessel_zero_Jnu.restype=c_double
        gsl_sf_bessel_zero_Jnu.argtyp=[c_double,c_uint]

        #Legendre Polynomials#
        gsl_sf_legendre_Pl=gsl.gsl_sf_legendre_Pl
        gsl_sf_legendre_Pl.restype=c_double
        gsl_sf_legendre_Pl.argtyp=[c_int,c_double]


        #DHT#
        if dht_found:
            gsl_dht_alloc=setTypes(gsl.gsl_dht_alloc,C.POINTER(gslStruct.gsl_dht_struct),c_size_t)
            gsl_dht_init=setTypes(gsl.gsl_dht_init,c_int,[C.POINTER(gslStruct.gsl_dht_struct),c_double,c_double])
            gsl_dht_new=setTypes(gsl.gsl_dht_new,C.POINTER(gslStruct.gsl_dht_struct),[c_size_t,c_double,c_double])
            gsl_dht_free=setTypes(gsl.gsl_dht_free,None,C.POINTER(gslStruct.gsl_dht_struct))
            gsl_dht_apply=setTypes(gsl.gsl_dht_apply,c_int,[C.POINTER(gslStruct.gsl_dht_struct),C.POINTER(c_double),C.POINTER(c_double)])        
            dht_2D=setTypes(dht.dht_2D,c_int,[c_size_t,C.POINTER(C.POINTER(gslStruct.gsl_dht_struct)),np.ctypeslib.ndpointer(np.complex, ndim=2, flags='C'),np.ctypeslib.ndpointer(np.complex, ndim=2, flags='C'),np.ctypeslib.ndpointer(np.complex, ndim=1, flags='C')])
        
            dht_bare=setTypes(dht.dht_bare,c_int,[C.POINTER(C.POINTER(gslStruct.gsl_dht_struct)),np.ctypeslib.ndpointer(np.complex, ndim=1, flags='C'),np.ctypeslib.ndpointer(np.complex, ndim=1, flags='C'),gslStruct.gsl_complex])
            dht_test=setTypes(dht.test,c_int,[c_size_t,c_size_t,np.ctypeslib.ndpointer(np.complex, ndim=2, flags='C')])
        
        
        #error handling#
        self.gsl_set_error_handler_off=gsl_set_error_handler_off
        
        #fft#
        #MixedRadix#
        #complex
        self.gsl_fft_complex_wavetable_alloc=gsl_fft_complex_wavetable_alloc
        self.gsl_fft_complex_wavetable_free=gsl_fft_complex_wavetable_free
        self.gsl_fft_complex_workspace_alloc=gsl_fft_complex_workspace_alloc
        self.gsl_fft_complex_workspace_free=gsl_fft_complex_workspace_free

        self.gsl_fft_complex_forward=gsl_fft_complex_forward
        self.gsl_fft_complex_inverse=gsl_fft_complex_inverse

        #real
        self.gsl_fft_real_wavetable_alloc=gsl_fft_real_wavetable_alloc
        self.gsl_fft_real_wavetable_free=gsl_fft_real_wavetable_free
        self.gsl_fft_real_workspace_alloc=gsl_fft_real_workspace_alloc
        self.gsl_fft_real_workspace_free=gsl_fft_real_workspace_free
        self.gsl_fft_real_unpack=gsl_fft_real_unpack
        
        self.gsl_fft_real_forward=gsl_fft_real_forward
        
        #halfcomplex
        self.gsl_fft_halfcomplex_wavetable_alloc=gsl_fft_halfcomplex_wavetable_alloc
        self.gsl_fft_halfcomplex_wavetable_free=gsl_fft_halfcomplex_wavetable_free
        self.gsl_fft_halfcomplex_unpack=gsl_fft_halfcomplex_unpack
        
        self.gsl_fft_halfcomplex_inverse=gsl_fft_halfcomplex_inverse


        #BesselZero#
        self.gsl_sf_bessel_zero_Jnu=gsl_sf_bessel_zero_Jnu
        #Legendre Polynomials
        self.gsl_sf_legendre_Pl=gsl_sf_legendre_Pl
        if dht_found:
            #DHT#
            self.gsl_dht_alloc=gsl_dht_alloc     #creates DHT object of given <size>
            self.gsl_dht_init=gsl_dht_init         #initalizes the DHT for a given <DHT object> and value of the <Bessel order> and <real cutoff>
            self.gsl_dht_new=gsl_dht_new      # (the two above at the same time) Creates DHT object of given <size> and initializes it for a griven <Bessel order> and  <real cutoff>  
            self.gsl_dht_free=gsl_dht_free      #frees the allocated memory for a DHT object
            self.gsl_dht_apply=gsl_dht_apply  # Applies a given <DHT object> trasform to an given <input array> and stores it in a given <output array>

            #my ajusted versions of DHT#
            self.dht_2D=dht_2D # computes dht of 2D array along the second axis
            self.dht_bare=dht_bare # modified version of gsl_dht_apply without ajustable prefactor
            self.dht_test=dht_test
