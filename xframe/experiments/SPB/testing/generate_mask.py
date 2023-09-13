import os
os.chdir('/gpfs/exfel/theory_group/user/berberic/Programs/xframe')
import numpy as np
from xframe.detectors.agipd import AGIPD
from xframe.library import physicsLibrary as pLib
from xframe.library.pythonLibrary import grow_mask
from xframe.presenters.matplolibPresenter import agipd_heat
from xframe.presenters.matplolibPresenter import heat2D
from xframe.startup_routines import dependency_injection,load_recipes
from xframe.library.mathLibrary import cartesian_to_spherical

init = False
if init:
    dependency_injection()
    from xframe.control.Control import Controller
    analysis_sketch,exp_sketch = load_recipes(ana_name='fxs3046_online.generate_plots_2', ana_settings='mean_test',exp_name='fxs3046_online',exp_settings = 'default_proc')
    controller = Controller(analysis_sketch,experiment_sketch=exp_sketch)
    controller.chooseJob()
    from scipy import ndimage
    from xframe import settings
    from xframe import Multiprocessing
    from xframe import database

db = database.analysis
ex_db = database.experiment
comm = Multiprocessing.comm_module

opt = settings.analysis
ex_opt = settings.experiment

def setup_locals():    
    agipd = AGIPD(ex_db,load_geometry_file=True)
    agipd.origin= ex_opt.detector_origin
    energy = ex_opt.x_ray_energy
    wavelength = pLib.energy_to_wavelength(energy)
    geometry = comm.get_geometry()
    return locals()


#locals().update(setup_locals())


def feature_selection(mask,asic_border_mask):
    mask = (mask.astype(int) - asic_border_mask.astype(int)).astype(bool)
    main_mask = np.full(mask.shape,0).astype(int)
    #main_mask[:]=mask
    small_features = np.full(mask.shape,0)
    for m in range(16):
        m_labels,m_nf = ndimage.label(mask[m])
        for l in range(1,m_nf):
            feature_mask = (m_labels == l)
            n_masked_pixels = np.sum(m_labels==l)        
            if n_masked_pixels >3 and n_masked_pixels <1e4  :
                tmp_mask = main_mask[m]
                tmp_mask += 2*l*feature_mask
                #main_mask[m]=tmp_mask
            elif  n_masked_pixels <=3:
                tmp_mask = small_features[m]
                tmp_mask += l*feature_mask
                #small_features[m]=tmp_mask
    return main_mask, small_features

def select_bad_pixels(fig_dict,limits={'std':[5,None],'max':[60,300]}):
    globals().update(setup_locals())
    std_dict = fig_dict['std_2d_generic']
    std_data = np.mean([std_dict[key] for key in std_dict],axis =0)
    max_data = fig_dict['maximum_2d']


    tmp_mask = ((std_data*max_data)!=0)
    bare_mask = np.array(agipd.sensitive_pixel_mask)
    bare_mask[bare_mask] = tmp_mask.flatten()
    bare_mask = ~bare_mask

    #exclude points with max lower than on all modules:
    if isinstance(limits['max'][0],(int,float)):
        max_data[max_data<limits['max'][0]]=0
        
    #exclude points with max higher than on some modules:
    for m in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        tmp_data = max_data[m]
        if isinstance(limits['max'][1],(int,float)):            
            tmp_data[tmp_data>limits['max'][1]]=0
        max_data[m] = tmp_data

    #mask things that are zero
    tmp_mask = (max_data!=0).astype(float)
    mask_max = np.array(agipd.sensitive_pixel_mask)
    mask_max[mask_max] = tmp_mask.flatten()
    mask_max = ~mask_max
        
    #exclude points with std higher/lower than on some modules:
    for m in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        tmp_data = std_data[m]
        if isinstance(limits['std'][0],(int,float)):
            tmp_data[tmp_data<limits['std'][0]]=0
        if isinstance(limits['std'][1],(int,float)):
            tmp_data[tmp_data>limits['std'][1]]=0
        std_data[m] = tmp_data

    #mask things that are zero
    tmp_mask = (max_data!=0).astype(float)
    mask_std = np.array(agipd.sensitive_pixel_mask)
    mask_std[mask_std] = tmp_mask.flatten()
    mask_std = ~mask_std

    return mask_max,mask_std


def get_border_mask(data_shape):
    border_mask = np.full(data_shape,False)
    border_mask[:,:2]=True
    border_mask[:,62:67]=True
    border_mask[:,127:132]=True
    border_mask[:,192:197]=True
    #border_mask[:,257:264]=True
    border_mask[:,257:262]=True 
    border_mask[:,322:327]=True
    border_mask[:,387:392]=True
    border_mask[:,452:457]=True
    border_mask[:,-2:]=True
    border_mask[:,:,-1:]=True
    border_mask[:,:,:1]=True
    return border_mask


def get_manual_mask(data_shape):
     # bad_asics_in module 15
    m_15_mask = np.full(data_shape,False)
    m_15_mask[15,:197]=True

    # bad_asic_in module 5
    m_5_mask = np.full(data_shape,False)
    m_5_mask[5,262:322,:64]=True

    # bad_asic_in module 10
    m_10_mask = np.full(data_shape,False)
    m_10_mask[10,262:322,64:]=True

    # bad_asic_in module 13
    m_13_mask = np.full(data_shape,False)
    m_13_mask[13,262:322,:64]=True
    
    # bad_asics_in module 1
    m_0_mask = np.full(data_shape,False)
    m_0_mask[0,:67,30:50]=True
    m_0_mask[0,457:,64:]=True

    # bad_asics_in module 3
    m_3_mask = np.full(data_shape,False)
    m_3_mask[3,:67,62:82]=True

    # bad_asics_in module 11
    m_11_mask = np.full(data_shape,False)
    m_11_mask[11,327:392,16:32]=True

    combined = m_15_mask | m_0_mask | m_3_mask | m_11_mask | m_5_mask | m_10_mask | m_13_mask

    return {'combined':combined,'module_15':m_15_mask,'module_0':m_0_mask,'module_3':m_3_mask,'module_11':m_11_mask,'module_5':m_5_mask,'module_10':m_10_mask,'module_13':m_13_mask}

def reformat_masks(masks,sensitive_pixel_mask,data_shape):
    ags=sensitive_pixel_mask
    ds = data_shape
    for key in masks:
        if isinstance(masks[key],dict):
            masks[key]=reformat_masks(masks[key],sensitive_pixel_mask,data_shape)
        else:
            masks[key] = ~masks[key].astype(bool)[ags].reshape(ds)
    return masks

def combine_masks(high_masks,medium_masks):
    globals().update(setup_locals())
    data_shape = high_masks[0].shape

    high_gain_mask = high_masks[0] * high_masks[1]
    medium_gain_mask = medium_masks[0] * medium_masks[1]

    max_mask = high_masks[0] | medium_masks[0]
    std_mask = high_masks[1] | medium_masks[1]
    
    border_mask = get_border_mask(data_shape)

    mask_c = max_mask*std_mask 
    main_mask,small_features = feature_selection(mask_c.copy(),border_mask)
    #main_mask_std,small_std = feature_selection(mask_std,border_mask)

    #small_features = small_max & small_std
    #main_mask = main_mask_max & main_mask_std

    manual_masks = get_manual_mask(data_shape)
    combined_manual = manual_masks.pop('combined')
    enlarged_main_mask = grow_mask(main_mask,2)

    combined_mask= (main_mask.astype(bool) | small_features.astype(bool) | border_mask.astype(bool) | combined_manual.astype(bool) | enlarged_main_mask.astype(bool))

    masks = {'combined':combined_mask,'main_fatures':main_mask,'small_features':small_features,'enlarged_main_features':enlarged_main_mask,'asic_border_mask':border_mask,'manual_masks':manual_masks,'high_gain_mask':high_gain_mask,'medium_gain_mask':medium_gain_mask}

    ags=agipd.sensitive_pixel_mask
    ds = (16,512,128)
    masks = reformat_masks(masks,ags,ds)
    return masks
    

def generation_routine(medium_dict,high_dict,limit={'std':[5,None],'max':[60,300]}):
    globals().update(setup_locals())
    
    
    detected_values = (mask_max*mask_std).copy()
    #mask[:] = 1


            #mask = ~mask#mask = ~mask

    border_mask = np.full(mask_max.shape,False)
    border_mask[:,:2]=True
    border_mask[:,62:67]=True
    border_mask[:,127:132]=True
    border_mask[:,192:197]=True
    #border_mask[:,257:264]=True
    border_mask[:,257:262]=True 
    border_mask[:,322:327]=True
    border_mask[:,387:392]=True
    border_mask[:,452:457]=True
    border_mask[:,-2:]=True
    border_mask[:,:,-1:]=True
    border_mask[:,:,:1]=True


    # bad_asics_in module 15
    m_15_mask = np.full(mask_max.shape,False)
    m_15_mask[15,:197]=True

    # bad_asics_in module 1
    m_0_mask = np.full(mask_max.shape,False)
    m_0_mask[0,:67,30:50]=True
    m_0_mask[0,457:,64:]=True

    # bad_asics_in module 3
    m_3_mask = np.full(mask_max.shape,False)
    m_3_mask[3,:67,62:82]=True

    # bad_asics_in module 11
    m_11_mask = np.full(mask_max.shape,False)
    m_11_mask[11,327:392,16:32]=True
    

    #bar_plus_mask = np.full(mask.shape,False)
    #bar_plus_mask[4:8,255:257]=True


    mask_std = (mask_std.astype(int) - border_mask.astype(int)).astype(bool)



    main_mask_max,small_max = feature_selection(mask_max)
    main_mask_std,small_std = feature_selection(mask_std)

    small_features = small_max & small_std
    main_mask = main_mask_max & main_mask_std
    
    main_mask[border_mask] = 0
    #border_mask -= agipd_heat.sensitive_pixel_mask 
    #grads= ndimage.sobel(main_mask[0].astype(int))
    main_mask_int = main_mask.copy()
    main_mask = main_mask
    grads1= np.sum(np.abs(np.gradient(main_mask.astype(int),edge_order = 1)[1:]),axis = 0)
    grads2= np.sum(np.abs(np.gradient((grads1!=0).astype(int),edge_order = 1)[1:]),axis = 0)
    new_mask = (grads1!=0) | (grads2!=0)

    #xy_grads = np.sum(np.array(grads[1:]),axis = 0)

    #show_mask = m_labels
    ms=15
    #heat2D.show(main_mask[0]+2*new_mask,agipd.pixel_grid[ms],scale='lin')
    #agipd_heat.show(np.abs(2*final_mask-new_mask)[agipd.sensitive_pixel_mask],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin',cmap_name='viridis_r')
    
    ags=agipd.sensitive_pixel_mask
    ds = (16,512,128)
    main_mask = main_mask.astype(bool)[ags].reshape(ds)
    main_mask_int = main_mask_int[ags].reshape(ds)
    small_features = small_features.astype(bool)[ags].reshape(ds)
    new_mask = new_mask.astype(bool)[ags].reshape(ds)
    border_mask = border_mask.astype(bool)[ags].reshape(ds)
    #bar_mask = bar_mask.astype(bool)[ags].reshape(ds)
    #bar_plus_mask =  bar_plus_mask.astype(bool)[ags].reshape(ds)
    m_15_mask =  m_15_mask.astype(bool)[ags].reshape(ds)
    m_0_mask =  m_0_mask.astype(bool)[ags].reshape(ds)
    m_3_mask =  m_3_mask.astype(bool)[ags].reshape(ds)
    m_11_mask =  m_11_mask.astype(bool)[ags].reshape(ds)
    detected_values = detected_values.astype(bool)[ags].reshape(ds)
    

    combined_mask = ~(main_mask | small_features | new_mask |border_mask | bar_mask | bar_plus_mask)

    mask_dict = {'mask':combined_mask,'main_features':main_mask,'small_features':small_features,'enlagement_main':new_mask,'asic_borders':border_mask,'bars':bar_mask,'bar_asic_gap':bar_plus_mask,'main_features_int':main_mask_int,'module_15_bad_asics':m_15_mask,'module_0_bad_asics':m_0_mask,'module_3_bad_asic_area':m_3_mask,'module_11_bad_asic_area':m_11_mask,'detected_values':detected_values}
    globals().update(locals())
    return mask_dict

#stupid_limits = {'std':[66,60],'max':[310,300]} 

#fig_dict_low = db.load('figure_data',path_modifiers={'name':'bg_data_proc','run':3,'time':'2022-08-31_20:47:52'})
#low_limits = {'std':[100,1e4],'max':[4000,1e5]}
#mask_dict_low = generation_routine(fig_dict_low,limits = low_limits)#low_limits)

fig_dict_high = db.load('figure_data',path_modifiers={'name':'bg_data_proc','run':1,'time':'2022-08-31_20:45:39'})
fig_dict_medium = db.load('figure_data',path_modifiers={'name':'bg_data_proc','run':2,'time':'2022-08-31_20:46:53'})
medium_limits = {'std':[200,2000],'max':[2000,2e4]}
high_limits = {'std':[6,60],'max':[25,300]}

medium_masks = select_bad_pixels(fig_dict_medium,limits = medium_limits)
high_masks = select_bad_pixels(fig_dict_high,limits = high_limits)

mask_dict = combine_masks(high_masks,medium_masks)

#mask_dict = database.experiment.load('custom_mask',path_modifiers={'name':'agipd_mask'})
fig = agipd_heat.get_fig(mask_dict['combined'],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'Agipd Mask p3046','y_label':'y_direction [mm]','x_label':'x_direction [mm]'},scale='lin')
db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/combined.matplotlib',fig,dpi=800)
db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/new_mask.h5',mask_dict)
#fig = agipd_heat.get_fig(~mask_dict['high_gain_mask'],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/high_gain.matplotlib',fig,dpi=800)
#fig = agipd_heat.get_fig(~mask_dict['medium_gain_mask'],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/medium_gain.matplotlib',fig,dpi=800)
#fig = agipd_heat.get_fig(~mask_dict['enlarged_main_features'],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/test.matplotlib',fig,dpi=800)





#mask_dict = db_ana.load_default({'name':'/gpfs/exfel/exp/SPB/202102/p002510/scratch/berberich/mask.h5'},use_name_as_path = True)

#streak_mask = np.full((16,512,128),True)
#streak_mask[4,:,:94]=False
#streak_mask[-1,:,:94]=False

#mask_low = ~ (mask_dict_low['asic_borders'] |  mask_dict_low['enlagement_main'] | mask_dict_low['main_features'] | mask_dict_low['small_features'])
#mask_low = ~ (mask_dict_low['main_features'])
#mask_medium = ~ (mask_dict_medium['asic_borders'] | mask_dict_medium['enlagement_main'] | mask_dict_medium['main_features'] | mask_dict_medium['small_features'])
#mask_medium = ~ (mask_dict_medium['detected_values'])
#mask_high = ~ (mask_dict_high['asic_borders'] | mask_dict_high['enlagement_main'] | mask_dict_high['main_features'] | mask_dict_high['small_features'] | mask_dict_high['module_15_bad_asics'] | mask_dict_high['module_0_bad_asics'] | mask_dict_high['module_3_bad_asic_area']| mask_dict_high['module_11_bad_asic_area'])
#mask_high = ~ (mask_dict_high['detected_values'] )

#mask_or =  mask_high | mask_medium
#mask_and =  mask_high & mask_medium
#
#slope = 128/(512+100)
#for x in range(512):
#    y=128-int(slope *x)
#    streak_mask[4,x,:y]=False
#    streak_mask[-1,x,:y]=False
#    
#mask =  mask & streak_mask
#
#mask_dict['crude_steak_mask']=streak_mask
#mask_dict['mask'] = mask




#db_ana.save_default(mask_dict,{'name':'/gpfs/exfel/exp/SPB/202102/p002510/scratch/berberich/mask.h5'},use_name_as_path = True)


#plot_mask = base
#plot_mask[8] += normal_mask[8]==new_mask[8]
#plot_mask= (new_mask)
#fig = agipd_heat.get_fig(mask_high,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/high_mask.matplotlib',fig,dpi=800)
#fig = agipd_heat.get_fig(mask_medium,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/medium_mask.matplotlib',fig,dpi=800)
##fig = agipd_heat.get_fig(mask_low,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
##db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/low_mask.matplotlib',fig,dpi=800)
#fig = agipd_heat.get_fig(mask_or,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/or_mask.matplotlib',fig,dpi=800)
#fig = agipd_heat.get_fig(mask_and,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin')
#db.save('/gpfs/exfel/theory_group/user/berberic/p3046/masks/and_mask.matplotlib',fig,dpi=800)

#fig = agipd_heat.get_fig(bar_mask[agipd.sensitive_pixel_mask],agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'},scale='lin',cmap_name='viridis_r')


#fig.savefig('/gpfs/exfel/exp/SPB/202102/p002510/scratch/berberich/figures/tmp_mask.png',dpi=800)

#fig.show()

# fig = agipd_heat.get_fig(std_data,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'std'})
# fig.savefig('/gpfs/exfel/theory_group/user/berberic/gain_test/figures/Standard_Deviation_2D/run_164/std.png',dpi = 800)
# fig = agipd_heat.get_fig(max_data,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'max'})
# fig.savefig('/gpfs/exfel/theory_group/user/berberic/gain_test/figures/Maximum 2D/run_164/max.png',dpi = 800)
# fig = agipd_heat.get_fig(std_data*max_data,agipd.pixel_grid,agipd.sensitive_pixel_mask,layout={'title':'combined'})
# fig.savefig('/gpfs/exfel/theory_group/user/berberic/gain_test/figures/Standard_Deviation_2D/run_164/combined.png',dpi = 800)


#heat2D.show(d['mean'],grid = regridder.regrid_data['cart']['lab_small']['grid'],scale='log')

#custom_mask = np.full(agipd.data_shape,True).astype(bool)
#db.save('custom_mask',{'mask':custom_mask},path_modifiers = {'name':'custom_mask'})

#background =  np.full(agipd.data_shape,0)
#background[4] = -1000.4
#db.save('background',{'background':background},path_modifiers = {'name':'background'})

