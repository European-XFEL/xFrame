import logging
import os
import re
import traceback
import numpy as np


from xframe.interfaces import DetectorInterface
from xframe.library.mathLibrary import plane3D
from xframe.library.gridLibrary import GridFactory
from xframe import database

log=logging.getLogger('root')

class AGIPD:
    dimensions=3
    number_of_modules=16    
    groups = np.array([[[12,13,14,15],[8,9,10,11]],[[0,1,2,3],[4,5,6,7]]],dtype = int)
    modules_per_group=4
    module_width_in_pixel = 512#+7
    module_height_in_pixel = 128
    data_shape = (number_of_modules,module_width_in_pixel,module_height_in_pixel)
    framed_pixel_mask = np.full((16,module_width_in_pixel+2,module_height_in_pixel+2),False)
    framed_pixel_mask[:,1:-1,1:-1]=True
    asic_slices = [
        [
            [slice(i * 64, i*64+64),slice(0,64)],
            [slice(i * 64, i*64+64),slice(64,128)],
        ]
        for i in range(8)]
    
    def __init__(self,geometry_path = None,super_sampling=1):
        self.super_sampling = super_sampling
        if super_sampling >1:
            self.module_width_in_pixel*=super_sampling
            self.module_height_in_pixel*=super_sampling
            self.data_shape = (self.number_of_modules,self.module_width_in_pixel,self.module_height_in_pixel)
            self.framed_pixel_mask = np.full((self.number_of_modules,self.module_width_in_pixel+2,self.module_height_in_pixel+2),False)
            self.framed_pixel_mask[:,1:-1,1:-1]=True
        
        self.database=database.experiment           
        self._origin = np.zeros(3,dtype = float)
        self.quadrants=np.zeros([2,2])
        self.pixel_corners = np.zeros([self.number_of_modules,self.module_width_in_pixel+1, self.module_height_in_pixel+1,self.dimensions])
        self.framed_pixel_centers=np.zeros([self.number_of_modules,self.module_width_in_pixel+2, self.module_height_in_pixel+2,self.dimensions])
        
        modules=[]
        if super_sampling > 1:
            modules.append(AGIPDmodule2(0,super_sampling=super_sampling))
        else:
            modules.append(AGIPDmodule(0))
        self.wide_pixel_mask = np.full((16,self.module_width_in_pixel,self.module_height_in_pixel),False)
        self.wide_pixel_mask[0,...]=modules[0].horz_wide_pixel_mask[:,None]
        for id in np.arange(1,self.number_of_modules,1):
            if super_sampling > 1:
                newModule=AGIPDmodule2(id,pixel_corners=modules[0].pixel_corners.copy(),super_sampling=super_sampling)
            else:
                newModule=AGIPDmodule(id,pixel_corners=modules[0].pixel_corners.copy())
            modules.append(newModule)
            self.wide_pixel_mask[id,...]=newModule.horz_wide_pixel_mask[:,None]
        modules=np.array(modules)
        self.modules=modules
        
        quadrant0=AGIPDmoduleGroup(modules[self.groups[0,0]])
        quadrant1=AGIPDmoduleGroup(modules[self.groups[0,1]])
        quadrant2=AGIPDmoduleGroup(modules[self.groups[1,0]])
        quadrant3=AGIPDmoduleGroup(modules[self.groups[1,1]])
        self.quadrants=np.array([[quadrant0,quadrant2],[quadrant1,quadrant3]])
        self.assemble_pixel_grids()
        if not isinstance(geometry_path,str):
            path = database.default.get_path('install_experiments',is_file=False)
            geometry_path = os.path.join(path,'SPB/default.geom')
        self.load_geometry(geometry_path)
        self._pixel_corner_index = False
        
        
    @property
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self,origin:np.ndarray):
        transitition_vect = origin - self._origin 
        for module in self.modules:
            plane = module.detection_plane
            plane.base = plane.base + transitition_vect
            module.detection_plane = plane
            self.assemble_pixel_grids()
        self._origin = origin

    
    def assemble_pixel_grids(self):
        for module in self.modules:
            module._update_pixel_corners()
            self.pixel_corners[module.id] = module.pixel_corners
            self.framed_pixel_centers[module.id] = module.framed_pixel_centers
            
    def moveModuleTo(self,moduleId,plane):
        self.modules[moduleId].detection_plane = plane
        self.pixel_corners[moduleId] = modules[moduleId].pixel_corners
        self.framed_pixel_centers[moduleId] = modules[moduleId].framed_pixel_centers

    def load_geometry(self,data_path='geometry'):
        full_path = self.database.get_path(data_path)
        file_type = os.path.splitext(full_path)[1]
        if file_type == '.h5':
            planes_dict = self.database.load(full_path)
            planes = tuple( plane3D.from_array(plane_dict[str(module)]) for module in modules)
        elif file_type == '.geom':
            planes = self.load_crystfel_geom(full_path)
        else:
            raise IOError('Geometry file type ({}) is not .h5 nor .geom'.format(file_type))
        for module in self.modules:
            module.detection_plane = planes[module.id]
            self.pixel_corners[module.id] = module.pixel_corners
            self.framed_pixel_centers[module.id] = module.framed_pixel_centers
            
    def load_crystfel_geom(self,data_path='geometry'):
        geom_path='geom_from_motors.geom'
        search_phrases=tuple('p{}a0/fs = '.format(module) for module in range(16))        
        lines = self.database.load(data_path)
        r_float = '[\-]*[0-9]+\.[0-9]*'
        r_module = '[0-9]+[a-z]'
        
        data=[None]*16
        for id,line in enumerate(lines):
            if any(tuple(phrase in line for phrase in search_phrases)):
                module = re.search(r_module,line)[0][:-1]
                y_direction = np.array(re.findall(r_float,line)+['0.0']).astype(float)
                x_direction = np.array(re.findall(r_float,lines[id+1])+['0.0']).astype(float)
                base = np.zeros(3,float)
                temp  = np.array(re.findall(r_float,lines[id+2])+re.findall(r_float,lines[id+3])+re.findall(r_float,lines[id+4])).astype(float)
                base[:len(temp)] = temp 
                base[:2]*=0.2*1e-3#0.2mm pixelwidth | geom file etries are measured in pixel width
                #print(f'{line[:3]} xvec {x_direction}\t\t yvec ={y_direction} corner = {base*5}')
                data[int(module)]=plane3D(base=base,x_direction =x_direction,y_direction = y_direction)
        planes= tuple(data)
        return planes
    
    @property
    def pixel_corner_index(self):
        if not isinstance(self._pixel_corner_index,np.ndarray):
            self._pixel_corner_index = self.generate_pixel_corner_index()
        return self._pixel_corner_index

    def generate_pixel_corner_index(self):
        pixel_corners = self.pixel_corners
        indices = np.arange(np.prod(pixel_corners.shape[:-1])).reshape(pixel_corners.shape[:-1])
        pixel_corner_index=[]
        for module in indices:
            module_indices = np.concatenate((module[:-1,1:,None],module[:-1,:-1,None],module[1:,:-1,None],module[1:,1:,None]),axis = -1)
            pixel_corner_index.append(module_indices)
        pixel_corner_index = np.array(pixel_corner_index).astype(np.int32)
        return pixel_corner_index
    
    def plot_data(self,data,scale='log',vmin=None,vmax=None,cmap='inferno',figsize=(10,10),bad_color='black'):
        from xframe.presenters import matplotlibPresenter
        layout = {'x_label':r'$x \quad [m] $','y_label':r'$y \quad [m]$'}
        fig = matplotlibPresenter.agipd_heatmap(data,self.pixel_corners,layout = layout,scale=scale,vmin=vmin,vmax=vmax,cmap=cmap,figsize=figsize,bad_color= bad_color)
        return fig
    
class AGIPDmoduleGroup:
    def __init__(self,modules):
        self.modules={}
        
        for module in modules:
            self.modules['{id}'.format(id=module.id)]=module
        

    def moveTo(self, quadrantPlain):
        print('Work In progress')


def construct_horz_wide_pixel_mask(_width_in_pixel,_wide_pixel_column_separation):
    horz_wide_pixel_mask = np.zeros(_width_in_pixel,bool)
    horz_wide_pixel_mask[_wide_pixel_column_separation-1:-1:_wide_pixel_column_separation] = True
    horz_wide_pixel_mask[_wide_pixel_column_separation::_wide_pixel_column_separation] = True
    return horz_wide_pixel_mask
def construct_local_hv_pixel_corners(_height_in_pixel,_pixel_size,_wide_pixel_size,wide_pixel_mask):
    h_pixel_corners=np.zeros(len(wide_pixel_mask)+1,float)
    for i,is_wide in zip(range(len(wide_pixel_mask)),wide_pixel_mask):
        h_pixel_corners[i+1]=h_pixel_corners[i]+(~is_wide)*_pixel_size[0]+is_wide*_wide_pixel_size[0]
    v_pixel_corners= np.arange(_height_in_pixel+1)*_pixel_size[1]
    return h_pixel_corners,v_pixel_corners
def construct_local_pixel_corners(h_corners,v_corners):
    pixel_corners_2d = np.stack(np.meshgrid(h_corners,v_corners,indexing = 'ij'),2)
    pixel_corners_3d = np.insert(pixel_corners_2d,2,0,axis=2)
    return pixel_corners_3d
def construct_local_framed_pixel_centers(h_corners,v_corners,_pixel_size):
    horizontal_centers = np.zeros(len(h_corners)+1,dtype=float)
    horizontal_centers[1:-1] = np.diff(h_corners)/2+h_corners[:-1]
    horizontal_centers[0] = horizontal_centers[1]-_pixel_size[0]
    horizontal_centers[-1] = horizontal_centers[-2]+_pixel_size[0]
    
    vertical_centers = np.zeros(len(v_corners)+1,dtype=float)
    vertical_centers[1:-1] = np.diff(v_corners)/2+v_corners[:-1]
    vertical_centers[0] = vertical_centers[1]-_pixel_size[1]
    vertical_centers[-1] = vertical_centers[-2]+_pixel_size[1]
    
    framed_pixel_centers_2d = np.stack(np.meshgrid(horizontal_centers,vertical_centers,indexing = 'ij'),2)
    framed_pixel_centers_3d = np.insert(framed_pixel_centers_2d,2,0,axis=2) 
    return framed_pixel_centers_3d
    
class AGIPDmodule:
    ''' 
    Each module consists of a rigid sensor of 128x512+7 pixels. Standard pixels are squares of width 0.2 mm. 
    Each 64'th and 64'th+1 pixel is of doubled width, i.e. 0.4x0.2 mm (exept the last pixel row).
    These are pixels horizontally in between (64,64) asics, the start and end of moduels are normal pixels:
    
    module:
     a00 || a10 || a20 || a30 || a40 || a50 || a60 || a70 
     a01 || a11 || a21 || a31 || a41 || a51 || a61 || a71
    each | stands for one of the doubled with pixels.
    
    
    For specifics about the pixel structure in each module look at the paper from Allagholi et al. (ISSN 16005775) 
    The Adaptive Gain Integrating Pixel Detector at the European XFEL.
    Section 4.2
    '''    
    _spaceDim=3
    _number_of_gain_stages=3
    _number_of_memory_cells=352
    _pixel_size=np.array([.2,.2])*1e-3 # pixel size in meters
    _wide_pixel_size= np.array([.4,.2])*1e-3 # pixel size in meters
    _wide_pixel_column_separation = 64 #65 no! the doubled pixels are not dead (insensitive)
    _width_in_pixel=512 #+7 no! the doubled pixels are not dead (insensitive)
    _height_in_pixel=128
    
    horz_wide_pixel_mask = construct_horz_wide_pixel_mask(_width_in_pixel,_wide_pixel_column_separation)
    _local_horz_pixel_corners,_local_vert_pixel_corners = construct_local_hv_pixel_corners(_height_in_pixel,_pixel_size,_wide_pixel_size,horz_wide_pixel_mask)
    local_pixel_corners = construct_local_pixel_corners(_local_horz_pixel_corners,_local_vert_pixel_corners)
    local_framed_pixel_centers = construct_local_framed_pixel_centers(_local_horz_pixel_corners,_local_vert_pixel_corners,_pixel_size)
        
    
    def __init__(self,id,detection_plane = False, pixel_corners = False):
        self.pixel_corners=np.zeros([self._width_in_pixel+1,self._height_in_pixel+1,self._spaceDim])
        self.framed_pixel_centers=np.zeros([self._width_in_pixel+2,self._height_in_pixel+2,self._spaceDim])
        if isinstance(detection_plane,plane3D):
            self._detection_plane = detection_plane
        else:
            self._detection_plane = plane3D()#detection_plane
            
        self.id=id
        if isinstance(pixel_corners,np.ndarray):
            self.pixel_corners=pixel_corners
        else:
            self._update_pixel_corners()
    @property
    def detection_plane(self):
        return self._detection_plane
    @detection_plane.setter
    def detection_plane(self, plane:plane3D):
        self._detection_plane = plane
        self._update_pixel_corners()

    def _update_pixel_corners(self):
        plane = self._detection_plane
        base = plane.standardForm['base']
        x_direction = plane.standardForm['x_direction']
        y_direction = plane.standardForm['y_direction']
        
        transformation_matrix = np.array([x_direction,y_direction,np.zeros(3)]).T
        shape = self.pixel_corners.shape
        framed_center_shape = self.framed_pixel_centers.shape
        self.pixel_corners = base + transformation_matrix.dot(self.local_pixel_corners.reshape(-1,3).T).T.reshape(shape)
        self.framed_pixel_centers = base + transformation_matrix.dot(self.local_framed_pixel_centers.reshape(-1,3).T).T.reshape(framed_center_shape)       


class AGIPDmodule2:
    ''' 
    Each module consists of a rigid sensor of 128x512+7 pixels. Standard pixels are squares of width 0.2 mm. 
    Each 64'th and 64'th+1 pixel is of doubled width, i.e. 0.4x0.2 mm (exept the last pixel row).
    These are pixels horizontally in between (64,64) asics, the start and end of moduels are normal pixels:
    
    module:
     a00 || a10 || a20 || a30 || a40 || a50 || a60 || a70 
     a01 || a11 || a21 || a31 || a41 || a51 || a61 || a71
    each | stands for one of the doubled with pixels.
    
    
    For specifics about the pixel structure in each module look at the paper from Allagholi et al. (ISSN 16005775) 
    The Adaptive Gain Integrating Pixel Detector at the European XFEL.
    Section 4.2
    '''    
    _spaceDim=3
    _number_of_gain_stages=3
    _number_of_memory_cells=352
    _pixel_size=np.array([.2,.2])*1e-3 # pixel size in meters
    _wide_pixel_size= np.array([.4,.2])*1e-3 # pixel size in meters
    _wide_pixel_column_separation = 64 #65 no! the doubled pixels are not dead (insensitive)
    _width_in_pixel=512 #+7 no! the doubled pixels are not dead (insensitive)
    _height_in_pixel=128
    
    def __init__(self,id,detection_plane = False, pixel_corners = False,super_sampling=1):
        self.super_sampling=super_sampling
        self._pixel_size=self._pixel_size.copy()/super_sampling
        self._wide_pixel_size=self._wide_pixel_size.copy()/super_sampling
        self._wide_pixel_column_separation*=super_sampling
        self._width_in_pixel*=super_sampling
        self._height_in_pixel*=super_sampling

        self.horz_wide_pixel_mask = construct_horz_wide_pixel_mask(self._width_in_pixel,self._wide_pixel_column_separation)
        self._local_horz_pixel_corners,self._local_vert_pixel_corners = construct_local_hv_pixel_corners(self._height_in_pixel,self._pixel_size,self._wide_pixel_size,self.horz_wide_pixel_mask)
        self.local_pixel_corners = construct_local_pixel_corners(self._local_horz_pixel_corners,self._local_vert_pixel_corners)
        self.local_framed_pixel_centers = construct_local_framed_pixel_centers(self._local_horz_pixel_corners,self._local_vert_pixel_corners,self._pixel_size)
        
        self.pixel_corners=np.zeros([self._width_in_pixel+1,self._height_in_pixel+1,self._spaceDim])
        self.framed_pixel_centers=np.zeros([self._width_in_pixel+2,self._height_in_pixel+2,self._spaceDim])
        if isinstance(detection_plane,plane3D):
            self._detection_plane = detection_plane
        else:
            self._detection_plane = plane3D()#detection_plane
            
        self.id=id
        if isinstance(pixel_corners,np.ndarray):
            self.pixel_corners=pixel_corners
        else:
            self._update_pixel_corners()
    @property
    def detection_plane(self):
        return self._detection_plane
    @detection_plane.setter
    def detection_plane(self, plane:plane3D):
        self._detection_plane = plane
        self._update_pixel_corners()

    def _update_pixel_corners(self):
        plane = self._detection_plane
        base = plane.standardForm['base']
        x_direction = plane.standardForm['x_direction']
        y_direction = plane.standardForm['y_direction']
        
        transformation_matrix = np.array([x_direction,y_direction,np.zeros(3)]).T
        shape = self.pixel_corners.shape
        framed_center_shape = self.framed_pixel_centers.shape
        self.pixel_corners = base + transformation_matrix.dot(self.local_pixel_corners.reshape(-1,3).T).T.reshape(shape)
        self.framed_pixel_centers = base + transformation_matrix.dot(self.local_framed_pixel_centers.reshape(-1,3).T).T.reshape(framed_center_shape)       
