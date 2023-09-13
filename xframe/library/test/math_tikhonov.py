import numpy as np
from xframe.library import mathLibrary as mLib
from xframe import log
log=log.setup_custom_logger('root','INFO')

#A = np.random.rand(3*20*10).reshape(3,20,10)
A = np.diag(np.exp(-np.arange(0,200)*np.log(1e+6)/100))
b_free = np.arange(0.9,1,0.1/200)[::-1]
b = b_free + np.random.normal(scale = 0.001,size = 200)
b = np.stack((b))


#locals().update(mLib.approximate_tikhonov_parameters(A,b))
lambd = mLib.approximate_tikhonov_parameters(A,b)
x = mLib.optimal_tikhonov_regularization(A,b)
x2 = np.linalg.lstsq(A,b,rcond = None)[0][None,...]
x3 = mLib.tikhonov_regularization(A,b,[0.617])

r = np.sum(A*x[:,None,:],axis = -1)
r2= np.sum(A*x2[:,None,:],axis = -1)
r3= np.sum(A*x3[:,None,:],axis = -1)                     

print(np.mean(np.abs(r-b_free)/np.abs(b_free)))
print(np.mean(np.abs(r2-b_free)/np.abs(b_free)))
print(np.mean(np.abs(r3-b_free)/np.abs(b_free)))
