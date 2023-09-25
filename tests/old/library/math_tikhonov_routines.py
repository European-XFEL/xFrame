import numpy as np
from xframe.library import mathLibrary as mLib
from xframe import log
import scipy as sp
log=log.setup_custom_logger('root','INFO')

#A = np.random.rand(3*20*10).reshape(3,20,10)
size = 100
n_systems = 3
A = np.diag(np.exp(-np.arange(0,size)*np.log(1e+5)/size))
#A[0,-1]=100
x_exact = np.arange(0.9,1,0.1/size)[::-1]
b_free = A.dot(x_exact)
b = b_free[None,:] + np.random.normal(scale = 0.1,size = n_systems*size).reshape(n_systems,size)

A = np.stack((A,)*n_systems)


'''
    Finds x that minimizes $|| A x -B ||^2 - ||\lambda Id x ||^2 $
'''
if A.ndim < 3:
    A = A[None,...]
    b = b[None,...]
try:
    #lambd = mLib.approximate_tikhonov_parameters(AA,bb)
    lambd = mLib.approximate_tikhonov_parameters(A,b)
    #lambd = np.array([0.0])
except Exception:
    print('fuu')
    lambd = np.array([0.0])

allow_offset = True
#from sklearn import linear_model
#reg2 = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13),gcv_mode='svd',fit_intercept=False,normalize = False)
#reg = linear_model.Ridge(alpha=lambd[0],solver='svd',fit_intercept=allow_offset,normalize = False)
#reg.fit(A[0],b[0])
#reg2.fit(A[0],b[0])
#x1 = np.copy(reg.coef_)
#x2 = reg2.coef_
#new_b = A[0].dot(x1) + reg.intercept_
#
#lambd2 = mLib.approximate_tikhonov_parameters(A[0],new_b)
#reg3 = linear_model.Ridge(alpha=lambd2[0],solver='svd',fit_intercept=allow_offset,normalize = False)
#x3 = reg3.fit(A[0],new_b).coef_

#x,o= mLib.tikhonov_regularization(A,b,lambd,allow_offset=allow_offset)
xx,oo= mLib.optimal_tikhonov_regularization(A,b,allow_offset=allow_offset,iterations = 1)
xxx,ooo= mLib.optimal_tikhonov_regularization(A,b,allow_offset=allow_offset,iterations = 1)

new_b2 = A[0].dot(xx[0])+oo[0]
new_bs = np.sum(A*xx[:,None,:],axis = -1)+oo[:,None]
