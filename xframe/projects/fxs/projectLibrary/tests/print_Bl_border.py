import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.startup_routines import dependency_injection
dependency_injection()
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi

db = default_DB()

r_base_path = '/gpfs/exfel/theory_group/user/berberic/MTIP/reconstructions/20_8_2022/run_0/'
opt = db.load_settings('analysis',path = r_base_path + 'analysis.yaml')
base_path = '/gpfs/exfel/theory_group/user/berberic/MTIP/reciprocal_proj_data/'
path = base_path + '3d_covid_1p_10k_mO70_02_psd.h5'
data = db.load(path)

Bl = data['deg_2_invariant']
qs = data['data_radial_points']
options = {'plot_range':[1e8,1e17]}
out_path = '/gpfs/exfel/theory_group/user/berberic/MTIP/test/'

point_1=np.array((4,qs[3]))
point_2=np.array((70,0.34))
diff = point_2 - point_1
rot=np.array([[0,1],[-1,0]])
diff_rot = -(rot@diff)
grid = GridFactory.construct_grid('uniform',[np.arange(0,71),qs])
grid_2 = grid[:]-point_1
dot = np.sum(grid_2[:]*diff_rot[None,None,:],axis = -1)
mask = dot<0
mask = mask[:,:,None] + mask[:,None,:]
#Bl[mask] = 0

#print_bl = np.moveaxis(Bl,0,2)
#print_bl[:,mask.T]=0


def _save_first_invariants(bls,radial_points,base_path,options,name='diag_'):
    max_value = np.max(tuple(np.abs(bl).max() for bl in bls))
    vmin,vmax = options.get('plot_range',[max_value*1e-12,max_value])
    if [vmin,vmax] == [None,None]:
        vmin,vmax = [max_value*1e-12,max_value]
        
    grid = GridFactory.construct_grid('uniform',[radial_points,np.arange(bls.shape[2])])
    order_ids = [6,12,18,24,30]
    #order_ids = [20,40,60,80,127]
    
    shape = [2,len(order_ids)]
    layouts = []
    for i in range(shape[0]):
        layout_part = []
        orders = np.arange(10*i,10*i+len(order_ids)*2,2)
        for o in orders:
            layout = {'title':'$B_{'+'{}'.format(o)+'}$',
                      'x_label':'$q_1$',
                      'y_label':'$q_2$'
                      }
            layout_part.append(layout)
        layouts.append(layout_part)
                                                
    fig_bl_masks = heat2D_multi.get_fig([[np.abs(bls[n]) for n in order_ids],[np.abs(bls[10+n]) for n in order_ids] ],scale = 'log',layout = layouts,grid =grid[:],shape = shape,size = (30,10),vmin= vmin, vmax = vmax,cmap='plasma')
    bl_path = base_path +name+'Bl.matplotlib'
    db.save(bl_path,fig_bl_masks,dpi = 300)
    

#_save_first_invariants(print_bl[...,::2],qs,out_path,options,'diag_')
#_save_first_invariants(Bl,qs,out_path,options,'diag_2')
