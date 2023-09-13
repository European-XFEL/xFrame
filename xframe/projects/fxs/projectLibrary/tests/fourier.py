import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.plugins.MTIP.analysisLibrary.fourier_transforms import generate_ft
from xframe.plugins.MTIP.analysisLibrary.harmonic_transforms import HarmonicTransform
from xframe.plugins.MTIP.analysisLibrary.hankel_transforms import generate_weightDict
from xframe.plugins.MTIP.analysisLibrary.ft_grid_pairs import get_grid
from xframe.plugins.MTIP.analysisLibrary.fxs_invariant_tools import density_to_deg2_invariants
from xframe.library.mathLibrary import SampleShapeFunctions
from xframe.startup_routines import load_recipes
from xframe.control.Control import Controller
analysis_sketch = load_recipes(ana_name='MTIP.reconstruct', ana_settings='3d_pentagon_1p_00')[0]
controller = Controller(analysis_sketch)
from xframe.settings import analysis as opt
from xframe.database import analysis as db

opt.dimensions = 3
opt.grid.n_phis = 2*opt.grid.max_order+1
#opt.grid.n_angular_points = 100
pi_in_q = opt.fourier_transform.pi_in_q
if pi_in_q:
    name_postfix='N'+str(opt.grid.n_radial_points)+'mO'+str(opt.grid.max_order)+'nO'+str(len(opt.fourier_transform.pos_orders))+'_pi'
else:
    name_postfix='N'+str(opt.grid.n_radial_points)+'mO'+str(opt.grid.max_order)+'nO'+str(len(opt.fourier_transform.pos_orders))

#try:
#    weights = db.load('ft_weights',path_modifiers={'postfix':name_postfix,'type':opt.fourier_transform.type+'_'+str(opt.dimensions)+'D'})
#except FileNotFoundError as e:
#    weights = generate_weightDict_zernike(opt.grid.max_order,opt.grid.n_radial_points,dimensions=opt.dimensions)
mode = 'trapz'
weights = generate_weightDict(opt.grid.max_order,opt.grid.n_radial_points,dimensions=opt.dimensions,mode=mode)
    
data = db.load('reciprocal_proj_data',path_modifiers={'name':opt.name})
max_q= data['data_radial_points'].max()
max_r = np.pi*opt.grid.n_radial_points/max_q


ht_opt={
            'dimensions':opt.dimensions,
            **opt.grid,            
        }

harm_trf = HarmonicTransform('complex',ht_opt)
ft,ift =  generate_ft(max_r,weights,harm_trf,opt.dimensions,use_gpu=True,pi_in_q=pi_in_q,mode=mode)


grid_pair=get_grid({**opt.fourier_transform,**ht_opt,**harm_trf.grid_param,'max_q':max_q,'n_radial_points_from_data':256})
real_grid = grid_pair.realGrid

if opt.dimensions == 2:
    f = SampleShapeFunctions.get_polygon_function(600,6,coordSys = 'polar')
    density = f(real_grid[:]).astype(complex)
    density[:20,:19]=-1
if opt.dimensions ==3 :
    f = SampleShapeFunctions.get_tetrahedral_function(600)
    density = f(real_grid[:]).astype(complex)
    density[:20,:19,-10:]=-1
    
B1 = density_to_deg2_invariants(density,harm_trf,ft,opt.dimensions)
rd = ft(density)

d2 = ift(rd)
B2 = density_to_deg2_invariants(d2,harm_trf,ft,opt.dimensions)
diff = (density.real - d2.real)**2

if opt.dimensions == 3:
    db.save('fourier_test_{}_3D_2.vts'.format(mode),[density.real, d2.real , diff],grid = real_grid[:],dset_names=["original",'transformed', 'diff'],grid_type='spherical')
else:
    db.save('fourier_test_{}_2D.vts'.format(mode),[density.real, d2.real , diff],grid = real_grid[:],dset_names=["original",'transformed', 'diff'],grid_type='polar')
#    

qs=grid_pair.reciprocalGrid[:].__getitem__((slice(None),)+(0,)*opt.dimensions) # expression is just grid[:,0,0,0] in 3D case ans grid[:,0,0] in 2D
qqs=qs[None,:,None]*qs[None,None,:]
if opt.dimensions == 3:
    qqs *= qqs


diff_norm = np.sum((qqs*(B1-B2))**2,axis = (1,2))
norm = np.sum((qqs*B1)**2,axis = (1,2))
errors = np.full(len(norm),-1,dtype = float)
non_zero_mask = norm!=0.0
errors[non_zero_mask]=diff_norm[non_zero_mask]/norm[non_zero_mask]
print(errors)



orders = np.arange(71)
signs = [ (-1)**m for l in orders for m in range(1,l+1)]
pos_ids =  [ l*(l+1)+m for l in orders for m in range(1,l+1)]
neg_ids =  [ l*(l+1)+m for l in orders for m in range(-1,-l-1,-1)]
zero_ids = [ l*(l+1) for l in orders]
L = orders[-1]
pos_mask = np.zeros((L+1)**2,dtype = bool)
neg_mask = np.zeros((L+1)**2,dtype = bool)
pos_mask[pos_ids] = True
neg_mask[:] = ~pos_mask
neg_mask[zero_ids] = False
harm_rd = harm_trf.forward(rd)

herrors = [np.max(np.abs(harm_rd[l][19] - (-1)**l * (-1.0)**np.arange(-l,l+1)*harm_rd[l][19][::-1].conj())) for l in orders]
error = np.max(herrors)


