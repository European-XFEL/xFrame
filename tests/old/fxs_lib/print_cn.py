import numpy as np
import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
from xframe.database.euxfel import default_DB
from xframe.library.gridLibrary import GridFactory
from xframe.startup_routines import dependency_injection
dependency_injection()
from xframe.presenters.matplolibPresenter import plot1D,heat2D_multi
from xframe.library.mathLibrary import circularHarmonicTransform_real_forward
from xframe.library.mathLibrary import circularHarmonicTransform_complex_forward

db = default_DB()

ccd_1 = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_covid_1p_100k.h5')
#ccd_1 = db.load('/gpfs/exfel/theory_group/user/berberic/MTIP/ccd/3d_covid_1p_100k_small.h5')

cc= ccd_1['intra']['ccf_2p_q1q2']
ccm = np.moveaxis(circularHarmonicTransform_real_forward(cc),2,0)
radial_points = ccd_1['q1']

base_path = '/gpfs/exfel/theory_group/user/berberic/MTIP/test/ccn/'
options = {'plot_range':[1e7,1e16]}

def _save_first_invariants(bls,radial_points,base_path,options,name='model_0_'):
    max_value = np.max(tuple(np.abs(bl).max() for bl in bls))
    vmin,vmax = options.get('plot_range',[max_value*1e-12,max_value])
    if [vmin,vmax] == [None,None]:
        vmin,vmax = [max_value*1e-12,max_value]
        
    grid = GridFactory.construct_grid('uniform',[radial_points,np.arange(bls.shape[2])])
    order_ids = [[2,4,16,24,34],[36,38,40,42,44]]
    #order_ids = [20,40,60,80,127]
    
    shape = [2,len(order_ids[0])]
    layouts = []
    for i in range(shape[0]):
        layout_part = []
        orders = np.array(order_ids[i])
        #orders = np.arange(10*i,10*i+len(order_ids)*2,2)
        for o in orders:
            layout = {'title':'$C_{'+'{}'.format(o)+'}$',
                      'x_label':'$q_1$',
                      'y_label':'$q_2$'
                      }
            layout_part.append(layout)
        layouts.append(layout_part)
                                                
    fig_bl_masks = heat2D_multi.get_fig([[np.abs(bls[n]) for n in order_ids[0]],[np.abs(bls[n]) for n in order_ids[1]] ],scale = 'log',layout = layouts,grid =grid[:],shape = shape,size = (30,10),vmin= vmin, vmax = vmax,cmap='plasma')
    bl_path = base_path +name+'ccn.matplotlib'
    db.save(bl_path,fig_bl_masks,dpi = 300)
    
_save_first_invariants(ccm,radial_points,base_path,options,'covid_100k_2')
