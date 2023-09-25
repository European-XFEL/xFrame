import numpy as np
from xframe.externalLibraries import soft_plugin,shtns_plugin

def test_rotate_coeff():
    bw=4
    shtns = shtns_plugin.sh(bw-1)
    soft = soft_plugin.Soft(bw)
    
    coeff = np.zeros(shtns.n_coeff,np.complex128)
    #coeff[4:9]=1
    coeff[4:9]=np.array([1.1, 0, 2, 3, 1])
    split_ids = shtns.cplx_l_split_indices
    euler_angles = np.array([0.,np.pi/2,0.])
    rotated_coeff = soft.rotate_coeff_single(coeff,split_ids,euler_angles)
    
    expected = np.array([0.249745, 1.55, 0.285982, -1.45, 3.24974])
    
    is_close = np.isclose(rotated_coeff[4:9],expected,atol = 1e-5)
    if (~is_close).any():
        print("TEST FAILED!")
    else:
        print('test passed')
    return locals()

def test_calc_C():
    pi = np.pi
    bw=100
    shtns = shtns_plugin.sh(bw-1)
    soft = soft_plugin.Soft(bw)
    
    
    split_ids = shtns.cplx_l_split_indices
    ml_split_ids = shtns.cplx_m_split_indices
    lm_split_ids = shtns.cplx_l_split_indices
    
    sphere_grid = shtns.grid[:]
    half_sphere = np.zeros((128,)+sphere_grid.shape[:-1])
    #half_sphere[:]=1
    for i in range(99):
        half_sphere[:,i,i] = 1
        half_sphere[:,i,i+1] = 1
        half_sphere[:,i,i+2] = 1
        
    coeff = np.concatenate(shtns.forward_l(half_sphere),axis = 1)
    euler_angles = np.array([0.15,0.9625,1.9])*np.pi
    a,b,g = euler_angles/np.pi
    euler_angles2 = np.array([int(g)+1-(g-int(g)),b,int(a)+1-(a-int(a))])*np.pi
    coeff2 = soft.rotate_coeff(coeff,lm_split_ids,euler_angles)
    #coeff3 = soft.rotate_coeff_single(coeff2[0],lm_split_ids,euler_angles2)[None,:]
    #coeff31 = soft.rotate_coeff_single(coeff2[0],lm_split_ids,euler_angles3)[None,:]
    #coeff2 = np.concatenate(shtns.forward_l(i_half_sphere))[None,:]
    #coeff = np.zeros(shtns.n_coeff,np.complex128)
    #coeff[4:9]=1
    #coeff[4:9]=np.array([1.1, 0, 2, 3, 1])
    #coeff = np.random.rand(shtns.n_coeff) + 1j*np.random.rand(shtns.n_coeff)
    #coeff=coeff[None,:]
    
    #coeff2_lm = shtns.m_to_l_ordering(coeff2)
    #signal = np.random.rand(*shtns.grid.shape)
    #signal = signal[None,:]
    #coeff = shtns.forward_l(signal)
    #ignal = shtns.inverse_l(np.split(coeff,lm_split_ids,axis = 1))    

    
    #ml_coeff = np.concatenate([coeff[:,ids] for ids in shtns.cplx_m_indices],axis = 1)
    print('max_real coeff = {}'.format(coeff.real.max()))
    print('max_imax coeff = {}'.format(coeff.imag.max()))
    #C1 = soft.calc_mean_C(coeff2_lm[None,:],coeff2_lm[None,:],1,lm_split_ids)
    C2 = soft.calc_mean_C(coeff,coeff2,2,lm_split_ids)
    maxarg = np.unravel_index(np.argmax(C2),(2*bw,)*3)
    angles = soft.grid[maxarg[1],maxarg[0],maxarg[2]]/np.pi
    angles[0] = 2 - angles[0]
    angles[2] = 2 - angles[2]
    print('angles/pi = {}'.format(angles))
    return locals()



if __name__=='__main__':
    #locals().update(test_rotate_coeff())
    locals().update(test_calc_C())
    
