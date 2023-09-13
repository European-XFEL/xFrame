import logging
log = logging.getLogger('root')
from flint import arb,arb_mat
from flint import acb,acb_mat
from flint import good
import numpy as np
from xframe.library.interfaces import GSLInterface
import time
_jacobi_p = arb.jacobi_p
_bessel_J = arb.bessel_j
_sqrt = arb.sqrt
_pi = arb.pi
half = arb('1/2')

class ARBPlugin(GSLInterface):
    lib = arb
    good = good    
    @staticmethod
    def jacobi_p(x,n,a,b):        
        def generate_functions(x,n,a,b):
            def f():
                return _jacobi_p(arb(float(x)),int(n),float(a),float(b))
            return f
        f_list=tuple(map(generate_functions,x,n,a,b))
        
        results = np.array(tuple(float(good(f,dps=15,maxprec=100000).str(radius=False)) for f in f_list))
        return results

    @staticmethod
    def spherical_bessel_arb(x,n):
        value = _sqrt(_pi()/(2*x))*_bessel_J(x,n+half)
        if x==0:
            if n==0:
                value=arb(1)
            else:
                value=arb(0)
        return value
    @staticmethod
    def zernike_ND_arb(m,n,x,D=2):
        nm2 = int((n-m)/2)
        return (-1)**(nm2)*x**m*_jacobi_p(1-2*x**2,nm2,m+D/2-1,0)
    @staticmethod
    def zernike_ND_arb_no_sign(m,n,x,D=2):
        return x**m*_jacobi_p(1-2*x**2,int((n-m)/2),m+D/2-1,0)
    
    @staticmethod
    def zernike_ND(m,n,x,D=2):
        D = arb(D)
        #((-1)**((n-m)/2))*(points**m)*eval_jacobi((n-m)/2,m+D/2-1,0,1-2*(points**2))
        def generate_functions(m,n,x_val):
            def f():
                x = arb(float(x_val))
                nm2 = int((n-m)/2)
                return (-1)**(nm2)*x**m*_jacobi_p(1-2*x**2,nm2,int(m)+D/2-1,0)
            return f
        f_list=tuple(map(generate_functions,m,n,x))
        
        results = np.array(tuple(float(good(f,dps=15,maxprec=100000).str(radius=False)) for f in f_list))
        return results
    @staticmethod
    def zernike_ND2(m,n_max,x,D=2):
        print(D)
        D = arb(D)
        n_points = len(x)
        ms = [m]*n_points
        result=[]
        zernike_ND = ARBPlugin.zernike_ND
        for n in range(m,n_max,2):
            print(n)
            result.append(zernike_ND(ms,[n]*n_points,x))
        return np.array(result)

    @staticmethod
    def calc_spherical_zernike_wights_fixed_ord_pi_worker(k,p,order,n_radial_points,expansion_limit,**kwargs):
        ''' k >= 1 , p>=0'''
        R_ND = ARBPlugin.zernike_ND_arb_no_sign
        jnu = ARBPlugin.spherical_bessel_arb
        #radial_points=opt['n_radial_points']+1        
        if p == 0:
            k2p = k**2
        else:
            k2p = k**2/p
        #k2p = p_not_zero_case*(k**2/p)+p_zero_case*(k**2) 
        expansion_range=range(order,expansion_limit+1,2)
        #print('k2p={}'.format(k2p))
        #print('R_ND={}'.format(tuple(R_ND(order,n,k/n_radial_points) for n in expansion_range)))

        summands = tuple((2*n+3)*R_ND(0,n,k/n_radial_points,D=3)*jnu(p*_pi(),n+1) for n in expansion_range)
        if (p==0) and (order==0):
            summands= ((2*0+3)*R_ND(order,0,k/n_radial_points,D=3),) + summands[1:]
        weight = sum(summands)
        return weight*k2p*_sqrt(2/_pi())
    
    @staticmethod
    def calc_spherical_zernike_wights_fixed_ord_pi(order,n_radial_points,expansion_limit,**kwargs):
        worker = ARBPlugin.calc_spherical_zernike_wights_fixed_ord_pi_worker
        ks = tuple(arb(int(k)) for k in np.arange(1,n_radial_points))
        ps = tuple(arb(int(p)) for p in np.arange(n_radial_points))
        start = time.time()
        def gen_functions(k,p):
            if p==0:
                print( time.time() - start )
            def f():
                return worker(k,p,order,n_radial_points,expansion_limit)
            return f
        
        weights=np.array(
            tuple(
                tuple(
                    float(good(gen_functions(k,p),dps=15,maxprec=100000).str(radius=False)) for p in ps) for k in ks)
        )
        return weights

        
        
