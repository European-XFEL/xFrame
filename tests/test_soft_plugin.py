import pytest
import os
import sys
import subprocess
import numpy as np
import importlib
import numpy as np
from xframe.externalLibraries.soft_plugin import Soft
bw = 16

@pytest.fixture(scope='module')
def soft():
    return Soft(bw)

def test_wigner_small_d(soft):
    d = soft.wigners_small
    betas = d.betas
    
    exact_d = {}
    exact_d[(0,0,0)] = np.full(len(betas),1)
    exact_d[(1,0,0)] = np.cos(betas)
    exact_d[(1,1,0)] = -1/np.sqrt(2)*np.sin(betas)
    exact_d[(1,-1,0)] = -1/np.sqrt(2)*np.sin(np.pi+betas) # 
    exact_d[(1,0,1)] = 1/np.sqrt(2)*np.sin(betas) #
    exact_d[(1,0,-1)] = 1/np.sqrt(2)*np.sin(np.pi+betas) #
    exact_d[(1,1,1)] = 1/2*(1+np.cos(betas))
    exact_d[(1,1,-1)] = 1/2*(1-np.cos(betas))
    exact_d[(1,-1,1)] = 1/2*(1-np.cos(betas)) #
    exact_d[(2,0,0)] = 1/2*(3*np.cos(betas)**2-1)
    exact_d[(2,1,0)] = -np.sqrt(3/8)*np.sin(2*betas)
    exact_d[(2,1,1)] = 1/2*(2*np.cos(betas)**2+np.cos(betas)-1)
    exact_d[(2,1,-1)] = 1/2*(-2*np.cos(betas)**2+np.cos(betas)+1)
    exact_d[(2,2,0)] = np.sqrt(3/8)*np.sin(betas)**2
    exact_d[(2,2,1)] = -1/2*np.sin(betas)*(1+np.cos(betas))
    exact_d[(2,2,-1)] = -1/2*np.sin(betas)*(1-np.cos(betas))
    exact_d[(2,2,2)] = 1/4*(1+np.cos(betas))**2
    exact_d[(2,2,-2)] = 1/4*(1-np.cos(betas))**2

    for key,vals in exact_d.items():
        assert np.isclose(vals,d.lnk[key],rtol=1e-16).all(),f'Given small d at {key} does not match the exact small d.'

    
