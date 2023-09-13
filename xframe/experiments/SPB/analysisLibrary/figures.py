from xframe.presenters import matplotlibPresenter as mpres
import numpy as np
from xframe.library.mathLibrary import spherical_to_cartesian



def Intensity_histogramm(values,n_patterns,run,caption):
    layout = {
        'title': 'Run:{} Intensity Histogramm \n {} considered patterns'.format(run,n_patterns),
        'x_label': 'Mean Intensity',
        'y_label': 'Frequency',
        'caption': caption
    }
    return mpres.hist1D.get_fig(values,y_scale = 'log',layout = layout)

    
def mean_2d(mean,run,caption,geometry,lab_grid=False):
    if lab_grid:
        print_grid = geometry['framed_lab_pixel_grid']
    else:
        print_grid = spherical_to_cartesian(geometry['framed_pixel_grid'])
    print_mask = geometry['framed_mask']
    layout = {
        'title': 'Run:{} Pixelwise Mean'.format(run),
        'x_label': 'x',
        'y_label': 'y',
        'caption': caption
    }
    return mpres.agipd_heat.get_fig(np.abs(mean),print_grid,print_mask,scale = 'log',layout = layout,vmin=1)

def max_2d(maximum,run,caption,geometry,lab_grid=False):
    if lab_grid:
        print_grid = geometry['framed_lab_pixel_grid']
    else:
        print_grid = spherical_to_cartesian(geometry['framed_pixel_grid'])
    print_mask = geometry['framed_mask']
    layout = {
        'title': 'Run:{} Pixelwise Maximum'.format(run),
        'x_label': 'x',
        'y_label': 'y',
        'caption': caption
    }
    return mpres.agipd_heat.get_fig(maximum,print_grid,print_mask,scale = 'log',layout = layout,vmin = 1)

def std_2d(maximum,run,caption,geometry,lab_grid=False):
    if lab_grid:
        print_grid = geometry['framed_lab_pixel_grid']
    else:
        print_grid = spherical_to_cartesian(geometry['framed_pixel_grid'])
    print_mask = geometry['framed_mask']
    layout = {
        'title': 'Run:{} Pixelwise Standard Deviation'.format(run),
        'x_label': 'x',
        'y_label': 'y',
        'caption': caption
    }
    return mpres.agipd_heat.get_fig(maximum,print_grid,print_mask,scale = 'log',layout = layout,vmin=0.6)



