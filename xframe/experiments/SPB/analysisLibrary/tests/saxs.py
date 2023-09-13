if os.getcwd()[-6:] != "source":
    #print(os.getcwd())
    os.chdir('../')
    
import numpy as np
from analysisLibrary import saxs as saxs_lib


data = np.arange(100).reshape(10,10)
mask = (np.random.rand(100).reshape(10,10)+0.5).astype(int).astype(bool)
mask[0] = False

saxs = saxs_lib.calc_saxs(data,mask)
saxs_slow = saxs_lib.calc_saxs_slow(data,mask)

print( 'both saxs versions give the same answer ={}'.format((saxs == saxs_slow).all())) 


data=np.array([[0,1,0,1,1,0]])
mask= data.astype(bool)
saxs = saxs_lib.calc_saxs(data,mask)
expected = np.array([1]) 
print('saxs is as expected = {}'.format((saxs == expected).all()))
