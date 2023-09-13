import numpy as np
from xframe.library.gridLibrary import NestedArray,ReGrider
from xframe.presenters.matplolibPresenter import plot1D

data = np.arange(100)*np.pi/50

g_a = NestedArray((np.arange(100)/4)[:,None],1)
g_b = NestedArray((np.arange(300)/2)[:,None],1)

new_data = ReGrider.regrid(data,g_a,'cartesian',g_b,'cartesian',options = {'interpolation':'linear','fill_value':0.0})
