import pygsl.testing.sf as sf
import numpy as np
from xframe.library.interfaces import GSLInterface

class GSLPlugin(GSLInterface):
    lib = sf
    @staticmethod
    def legendre_sphPlm_array(l_max,m_max,xs,return_orders = False,sorted_by_l = False):
        '''
        Returns array of all normalized associated Legendre coefficients $\overline{P}^m_l$ with orders $l\leql_{max}$.
        :param l_max: Maximum Order to consider
        :type l_max: int
        :param xs: Array or value of points at which to evaluate the associated legendre polynomials
        :type xs: numpy.ndarray or float
        :param return_orders: Wether or not to return the order arrays l and m for which values where calculated. 
        :type return_orders: bool
        :param sort_by_l: If true output is ordered like ($P^0_0,P^0_1,P^1_1,P^0_2,P^1_2,P^2_2,\ldots$) and otherwise the output is ordered by m (i.e.$ P^0_0,\ldots,P^0_{l_max},P^1_1,\ldots,P^0_{l_max},\ldots$)
        :type sort_by_l: bool
        :return: Array with coefficients. If return_orders is true, then the order arrays are returned as well.
        :rtype: numpy.ndarray or tuple
        '''
        if sorted_by_l:
            return GSLPlugin.legendre_sphPlm_array_l(l_max,m_max,xs,return_orders = return_orders)
        else:
            return GSLPlugin.legendre_sphPlm_array_m(l_max,m_max,xs,return_orders = return_orders)
    @staticmethod
    def legendre_sphPlm_array_l(l_max,m_max,xs,return_orders = False):
        ls = np.arange(l_max+1)
        ms = np.concatenate(tuple(np.arange(l+1)[:m_max+1] for l in ls))
        repeated_ls = np.concatenate(tuple(np.full(min(l+1,m_max+1),l) for l in ls))
        xs = np.atleast_1d(xs)
        values = sf.legendre_sphPlm(repeated_ls[:,None],ms[:,None],xs[None,:])
        if return_orders:
            return (np.squeeze(values),repeated_ls,ms)
        else:
            return np.squeeze(values)
    @staticmethod
    def legendre_sphPlm_array_m(l_max,m_max,xs,return_orders = False):        
        ms = np.arange(m_max+1)
        ls = np.concatenate(tuple(np.arange(m,l_max+1) for m in ms))
        repeated_ms = np.concatenate(tuple(np.full(max(0,l_max+1-m),m) for m in ms))
        xs = np.atleast_1d(xs)
        values = sf.legendre_sphPlm(ls[:,None],repeated_ms[:,None],xs[None,:])
        if return_orders:
            return (np.squeeze(values),ls,repeated_ms)
        else:
            return np.squeeze(values)

    @staticmethod
    def legendre_sphPlm_array_single_m(l_max,m,xs,return_orders = False):
        ls = np.arange(l_max+1)
        ms = np.full(len(ls),m)
        xs = np.atleast_1d(xs)
        values = sf.legendre_sphPlm(ls[:,None],ms[:,None],xs[None,:])
        if return_orders:
            return (np.squeeze(values),ls,ms)
        else:
            return np.squeeze(values)
    @staticmethod
    def legendre_sphPlm_array_single_l(l,l_max,xs,return_orders = False):
        ms = np.arange(0,l+1)
        ls = np.full(len(ms),l)
        xs = np.atleast_1d(xs)
        values = sf.legendre_sphPlm(l,ms[:,None],xs[None,:])
        
        if return_orders:
            return (np.squeeze(values),ls,ms)
        else:
            return np.squeeze(values)
        
    @staticmethod
    def bessel_jl(ls,xs):
        return sf.bessel_jl(ls,xs)

    @staticmethod
    def hyperg_2F1(a,b,c,z):        
        return sf.hyperg_2F1(a,b,c,z)


    @staticmethod
    def coupling_3j_e(two_ja, two_jb, two_jc, two_ma, two_mb, two_mc):        
        return sf.coupling_3j(two_ja, two_jb, two_jc, two_ma, two_mb, two_mc)

    @staticmethod
    def clebsch_gordan(J, j1, j2, m1, m2, M):        
        return (-1)**(j2-j1-M)*np.sqrt(2*J+1)*sf.coupling_3j(int(2*j1), int(2*j2), int(2*J), int(2*m1), int(2*m2), int(-2*M))

    @staticmethod
    def clebsch_gordan_2(J,j1, j2, m1, m2):
        M=m1+m2
        return (-1)**(np.abs(j2-j1-M))*np.sqrt(2*J+1)*sf.coupling_3j(2*j1, 2*j2, 2*J, 2*m1, 2*m2, -2*M)
