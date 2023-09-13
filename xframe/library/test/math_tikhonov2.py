import numpy as np
from xframe.library import mathLibrary as mLib
from xframe import log
import scipy as sp
log=log.setup_custom_logger('root','INFO')

#A = np.random.rand(3*20*10).reshape(3,20,10)
size = 100
A = np.diag(np.exp(-np.arange(0,size)*np.log(1e+5)/size))
A[0,-1]=100
x_exact = np.arange(0.9,1,0.1/size)[::-1]
b_free = A.dot(x_exact)
b = b_free + np.random.normal(scale = 0.001,size = size)
b = np.stack((b))



'''
    Finds x that minimizes $|| A x -B ||^2 - ||\lambda Id x ||^2 $
    '''
lambd = 0.617
n_orders = A.shape[-1]
lambd = np.atleast_1d(lambd)
if A.ndim < 3:
    A = A[None,...]
    b = b[None,...]
#n_qs = A.shape[0]
#log.info('n_ord = {} n_qs = {}'.format(n_orders,n_qs))
#eyes = np.asarray(lambd)[:,None,None]*np.eye(n_orders)[None,...]
#log.info('A shape = {} eyes_shape = {}'.format(A.shape,eyes.shape))
#new_A = np.concatenate((A,eyes),axis = 1)
#new_b = np.concatenate((b,np.zeros((n_qs,n_orders))),axis = 1)

AA = np.copy(A)
bb = np.copy(b)
#x = np.sum(vh*d_UT_y[:,:,None],axis = -2)
#preprocess data:
A_offset = np.average(A[0], axis=0,weights=None)
A_offset = A_offset.astype(A[0].dtype, copy=False)
AA[0] -= A_offset

A_scale = np.ones(A[0].shape[1], dtype=A[0].dtype)
b_offset = np.average(b[0], axis=0,weights=None)
bb[0] -= b_offset

try:
    #lambd = mLib.approximate_tikhonov_parameters(AA,bb)
    lambd = mLib.approximate_tikhonov_parameters(A,b)
    #lambd = np.array([0.0])
except Exception:
    print('fuu')
    lambd = np.array([0.0])
U, s, Vt = sp.linalg.svd(A[0], full_matrices=False)
idx = s > 1e-15  # same default value as scipy.linalg.pinv
s_nnz = s[idx][:,None]
UTy = np.dot(U.T, b[0])
d = np.zeros((s.size, lambd.size), dtype=A[0].dtype)
d[idx] = s_nnz / (s_nnz**2 + lambd)
d_UT_y = d * UTy[:,None]
xx = np.dot(Vt.T, d_UT_y).T
x = mLib.tikhonov_solver_svd(A,b,lambd)

intercept = b_offset - np.dot(A_offset, x.T)

bx = A[0].dot(x[0])#+b_offset#A[0].dot(x[0])+intercept
#bx = A[0].dot(x[0])+intercept


from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13),gcv_mode='svd',fit_intercept=False,normalize = False)
reg2 = linear_model.Ridge(alpha=0,solver='svd',fit_intercept=False,normalize = False)
reg.fit(A[0],b[0])
reg2.fit(A[0],b[0])
x1 = reg.coef_
x2 = reg2.coef_
bbx = A[0].dot(reg2.coef_)+reg2.intercept_

err = np.mean(np.linalg.norm(bx-b_free)/np.linalg.norm(b_free))
err2 = np.linalg.norm(np.abs(bbx-b_free)/np.linalg.norm(b_free))
xerr = np.mean(np.linalg.norm(x-x_exact)/np.linalg.norm(x_exact))
xerr2 = np.linalg.norm(np.abs(x1-x_exact)/np.linalg.norm(x_exact))
h=np.linalg.lstsq(A[0],b[0])[0]
xerr3 = np.linalg.norm(np.abs(h-x_exact)/np.linalg.norm(x_exact))
print('own',err)
print(err2)
print(reg.alpha_)
print(lambd)
print(xerr)
print(xerr2)
print(xerr3)
#x = np.array(tuple(np.linalg.lstsq(A_part,b_part,rcond = None)[0] for A_part,b_part in zip(new_A,new_b)))

    
