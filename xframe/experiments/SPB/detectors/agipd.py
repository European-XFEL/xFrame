import logging
import numpy as np


from xframe.experiment.interfaces import DetectorInterface
from xframe.simulators.interfaces import DetectorInterfaceSimulation
from xframe.detectors.interfaces import DatabaseInterfaceDetector
from xframe.library.mathLibrary import plane3D
from xframe.library.gridLibrary import GridFactory

log=logging.getLogger('root')

class AGIPD(DetectorInterface,DetectorInterfaceSimulation):
    dimensions=3
    number_of_modules=16    
    groups = np.array([[[12,13,14,15],[8,9,10,11]],[[0,1,2,3],[4,5,6,7]]],dtype = int)
    modules_per_group=4
    module_width_in_pixel = 512+7
    module_height_in_pixel = 128
    data_shape = (16,512,128)
    sensitive_pixel_mask = np.full((16,519,128),True)
    framed_sensitive_pixel_mask = np.full((16,521,130),False)
    sensitive_pixel_mask[:,64:][:,::65]=False
    framed_sensitive_pixel_mask[:,1:-1,1:-1]=sensitive_pixel_mask
    asic_slices = [
        [
            [slice(i * 64, i*64+64),slice(0,64)],
            [slice(i * 64, i*64+64),slice(64,128)],
        ]
        for i in range(8)]
    
    def __init__(self,database,load_geometry_file=False,**kwargs):
        try:
            assert isinstance(database,DatabaseInterfaceDetector)
            self.database=database
        except AssertionError:
            log.error('database is not instance of DatabaseInterfaceDetector')
        
        self._origin = np.zeros(3,dtype = float)
        self.quadrants=np.zeros([2,2])
        self.pixel_grid=np.zeros([self.number_of_modules,self.module_width_in_pixel+1, self.module_height_in_pixel+1,self.dimensions])
        self.framed_pixel_grid=np.zeros([self.number_of_modules,self.module_width_in_pixel+1+2, self.module_height_in_pixel+1+2,self.dimensions])
        self.framed_pixel_centers=np.zeros([self.number_of_modules,self.module_width_in_pixel+2, self.module_height_in_pixel+2,self.dimensions])
        
        modules=[]
        modules.append(AGIPDmodule(0))
        for id in np.arange(1,self.number_of_modules,1):
            newModule=AGIPDmodule(id,pixel_grid=modules[0].pixel_grid)
            modules.append(newModule)
        modules=np.array(modules)
        self.modules=modules
        
        quadrant0=AGIPDmoduleGroup(modules[self.groups[0,0]])
        quadrant1=AGIPDmoduleGroup(modules[self.groups[0,1]])
        quadrant2=AGIPDmoduleGroup(modules[self.groups[1,0]])
        quadrant3=AGIPDmoduleGroup(modules[self.groups[1,1]])
        self.quadrants=np.array([[quadrant0,quadrant2],[quadrant1,quadrant3]])
        self.assemblePixelGrid()
        if load_geometry_file:
            self.loadGeometryFile()
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
        self.assemblePixelGrid()
        self._origin = origin

    
    def assemblePixelGrid(self):
        for module in self.modules:            
            self.pixel_grid[module.id]=module.pixel_grid
            self.framed_pixel_grid[module.id]=module.framed_pixel_grid
            self.framed_pixel_centers[module.id]=module.framed_pixel_centers
            
    def moveModuleTo(self,moduleId,plane):
        self.modules[moduleId].detection_plane = plane
        self.pixel_grid[moduleId] = modules[moduleId].pixel_grid
        self.framed_pixel_grid[moduleId] = modules[moduleId].framed_pixel_grid
        self.framed_pixel_centers[moduleId]=modules[moduleId].framed_pixel_centers
                    
    def loadGeometryFile(self):
        modulePlains=self.database.load('geometry')
        listOfQuadrants=self.quadrants.flatten()
        for module in self.modules:
            module.detection_plane = modulePlains[module.id]
        self.assemblePixelGrid()
                
    def get_geometry(self):
        return self.pixel_grid        
    @property
    def pixel_corner_index(self):
        if not isinstance(self._pixel_corner_index,np.ndarray):
            self._pixel_corner_index = self.generate_pixel_corner_index()
        return self._pixel_corner_index

    def generate_pixel_corner_index(self):
        pixel_grid = self.pixel_grid
        indices = np.arange(np.prod(pixel_grid.shape[:-1])).reshape(pixel_grid.shape[:-1])
        pixel_corner_index=[]
        for module in indices:
            module_indices = np.concatenate((module[:-1,1:,None],module[:-1,:-1,None],module[1:,:-1,None],module[1:,1:,None]),axis = -1)
            pixel_corner_index.append(module_indices)
        pixel_corner_index = np.array(pixel_corner_index).astype(np.int32)
        return pixel_corner_index

    
class AGIPDmoduleGroup:
    def __init__(self,modules):
        self.modules={}
        
        for module in modules:
            self.modules['{id}'.format(id=module.id)]=module
        

    def moveTo(self, quadrantPlain):
        print('Work In progress')

    
class AGIPDmodule:
    ''' 
    Each module consists of a rigid sensor of 128x512+7 pixels. Standard pixels are squares of width 0.2 mm. 
    Each 65'th column consists of insensitive pixels with doubled width, i.e. 0.4x0.2 mm (exept the last pixel row).

    For specifics about the pixel structure in each module look at the paper from Allagholi et al. (ISSN 16005775) 
    The Adaptive Gain Integrating Pixel Detector at the European XFEL.
    Section 4.2
    '''    
    _spaceDim=3
    _numberOfGainStages=3
    _numberOfMemoryCells=352
    _standardPixelSize=np.array([.2,.2]) # pixel size in millimeter 
    _widePixelSize= np.array([.4,.2]) # pixel size in millimeter
    _standardPixelSize3D=np.array([.2,.2,0]) # pixel size in millimeter 
    _widePixelSize3D= np.array([.4,.2,0]) # pixel size in millimeter
    _widePixelColumnSeparation=65
    _widthInPixel=512+7
    _heightInPixel=128
    def _construct_local_pixel_grid(_widthInPixel,_heightInPixel,_widePixelColumnSeparation,_widePixelSize,_standardPixelSize):       
        localPixelGridInPixels=np.stack(np.meshgrid(np.arange(_widthInPixel+1), np.arange(_heightInPixel+1),indexing='ij' ),2)
        widePixelContribution=np.zeros(localPixelGridInPixels.shape)
        for widePixelColumn in np.arange(_widePixelColumnSeparation,_widthInPixel,_widePixelColumnSeparation):
            widePixelContribution[widePixelColumn:,:]+=np.array([_widePixelSize[0]-_standardPixelSize[0],0])
        localPixelGrid2D=localPixelGridInPixels*_standardPixelSize+widePixelContribution
        localPixelGrid3D=np.insert(localPixelGrid2D,2,0,axis=2)        
        return localPixelGrid3D
    def _construct_local_framed_pixel_centers(local_framed_pixel_grid,_standardPixelSize3D,_widePixelSize3D,_widePixelColumnSeparation):
        local_pixel_centers =  local_framed_pixel_grid[:-1,:-1]+_standardPixelSize3D/2
        local_pixel_centers[_widePixelColumnSeparation:-2:_widePixelColumnSeparation] += (_widePixelSize3D-_standardPixelSize3D)/2
        return local_pixel_centers
    def _construct_local_framed_pixel_grid(_framedWidthInPixel,_framedHeightInPixel,_widePixelColumnSeparation,_widePixelSize,_standardPixelSize):
        localPixelGridInPixels=np.stack(np.meshgrid(np.arange(-1,_framedWidthInPixel), np.arange(-1,_framedHeightInPixel),indexing='ij' ),2)
        widePixelContribution=np.zeros(localPixelGridInPixels.shape)
        for widePixelColumn in np.arange(_widePixelColumnSeparation+1,_framedWidthInPixel,_widePixelColumnSeparation):
            widePixelContribution[widePixelColumn:,:]+=np.array([_widePixelSize[0]-_standardPixelSize[0],0])
        localPixelGrid2D=localPixelGridInPixels*_standardPixelSize+widePixelContribution
        localPixelGrid3D=np.insert(localPixelGrid2D,2,0,axis=2)
        return localPixelGrid3D

    localPixelGrid=_construct_local_pixel_grid(_widthInPixel,_heightInPixel,_widePixelColumnSeparation,_widePixelSize,_standardPixelSize)
    localFramedPixelGrid = _construct_local_framed_pixel_grid(_widthInPixel+2,_heightInPixel+2,_widePixelColumnSeparation,_widePixelSize,_standardPixelSize)
    localFramedPixelCenters = _construct_local_framed_pixel_centers(localFramedPixelGrid,_standardPixelSize3D,_widePixelSize3D,_widePixelColumnSeparation)

    def __init__(self,id,detection_plane = False, pixel_grid = False):
        self.pixel_grid=np.zeros([self._widthInPixel+1,self._heightInPixel+1,self._spaceDim])
        self.framed_pixel_grid=np.zeros([self._widthInPixel+3,self._heightInPixel+3,self._spaceDim])
        self.framed_pixel_centers=np.zeros([self._widthInPixel+2,self._heightInPixel+2,self._spaceDim])
        if isinstance(detection_plane,plane3D):
            self._detection_plane = detection_plane
        else:
            self._detection_plane = plane3D()#detection_plane
            
        self.id=id
        if isinstance(pixel_grid,bool):
            self._update_pixel_grid()
        else:
            self.pixel_grid=pixel_grid
    @property
    def detection_plane(self):
        return self._detection_plane
    @detection_plane.setter
    def detection_plane(self, plane:plane3D):
        self._detection_plane = plane
        self._update_pixel_grid()

    def _update_pixel_grid(self):
        plane = self._detection_plane
        base=plane.standardForm['base']
        x_direction=plane.standardForm['x_direction']
        y_direction=plane.standardForm['y_direction']
        
        transformationMatrix=np.array([x_direction,y_direction,np.zeros(3)]).T
        shape=self.pixel_grid.shape
        framed_shape = self.framed_pixel_grid.shape
        framed_center_shape = self.framed_pixel_centers.shape
        self.pixel_grid=base+transformationMatrix.dot(self.localPixelGrid.reshape(-1,3).T).T.reshape(shape)
        self.framed_pixel_grid=base+transformationMatrix.dot(self.localFramedPixelGrid.reshape(-1,3).T).T.reshape(framed_shape)        
        self.framed_pixel_centers=base+transformationMatrix.dot(self.localFramedPixelCenters.reshape(-1,3).T).T.reshape(framed_center_shape)       


        
