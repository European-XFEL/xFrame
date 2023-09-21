import abc
import numpy as np
import logging
log = logging.getLogger('root')
from xframe.library.mathLibrary import SampleShapeFunctions,spherical_to_cartesian
import sys

class RegionOfInterest(abc.ABC):
    def __init__(self,parameters:dict,geometry:dict,modules=np.arange(16,dtype = int)):
        self.parameters = parameters
        self.geometry = geometry
        self.pixel_grid_spher = self.geometry['data_grid_spherical']
        #The following assumes spherical grid in form of r,theta,phi
        self.pixel_grid_polar = self.pixel_grid_spher[...,::2]        
        self.pixel_grid_cart = spherical_to_cartesian(self.pixel_grid_polar)
        self.n_total_modules = self.pixel_grid_cart.shape[0]
        self.data_modules = modules
        ## mask as if all of 16 modules would be used
        self.mask_complete = self.generate()
        self.mask_true_modules = self.calc_used_modules(self.mask_complete)
        self._used_modules = self.mask_true_modules
        ## mask data only using the data_modules
        self.used_module_ids = np.arange(len(self._used_modules))
        #log.info(modules)
        #log.info(self.mask_complete.shape)
        self.mask =  self.mask_complete[self._used_modules]

    @property
    def used_modules(self):
        return self._used_modules
    
    @used_modules.setter
    def used_modules(self,modules):
        #log.info('setter is called with {} '.format(modules))
        self._used_modules = modules
        self.used_module_ids = len(modules)
        #log.info('Mask shape before = {}'.format(self.mask.shape))
        self.mask = self.mask_complete[modules]
        #log.info('Mask shape after = {}'.format(self.mask.shape))
        
    @abc.abstractmethod
    def generate(self):
        ''' Template for generation routine. Has to return the mask and the used modules'''
        mask=None
        return mask

    @staticmethod
    def calc_used_modules(mask):
        used_modules = []
        for module,module_mask in enumerate(mask):
            if module_mask.any():
                used_modules.append(module)
        #return as tuple so that it throws an error if one tries to modify it
        return np.array(used_modules)
    

class Rectangle(RegionOfInterest):
    def generate(self):
        pixel_grid_cart = self.pixel_grid_cart
        
        x_len = self.parameters['x_len']
        y_len = self.parameters['y_len']
        center = self.parameters['center']        
        lengths = [x_len,y_len]
        rectangle_fct = SampleShapeFunctions.get_rectangle_function(lengths,center = center, coordSys = 'cartesian')
        
        mask = rectangle_fct(pixel_grid_cart).astype(bool)
        return mask

class Annulus(RegionOfInterest):
    def generate(self):
        pixel_grid_cart = self.pixel_grid_cart       

        inner_radius = self.parameters['inner_radius']
        outer_radius = self.parameters['outer_radius']
        center = self.parameters['center']
        anulus_fct = SampleShapeFunctions.get_anulus_function(inner_radius,outer_radius,center = center, coordSys = 'cartesian')
        
        mask = anulus_fct(pixel_grid_cart).astype(bool)        
        
        return mask

class Pixel(RegionOfInterest):
    def generate(self):
        pixel_grid_cart = self.pixel_grid_cart

        pixels = self.parameters['pixels']
        mask = np.zeros(pixel_grid_cart.shape[:-1],dtype = bool)
        index = tuple(zip(*pixels))
        mask[index] = True
        #log.info('pixels = {}'.format(pixels))
        #log.info('index array = {}'.format(index))
        #log.info('mask shape = {}'.format(mask.shape))                    
        
        return mask

class Asic(RegionOfInterest):
    def generate(self):
        pixel_grid_cart = self.pixel_grid_cart
        asics = self.parameters['asics']
        mask = np.zeros(pixel_grid_cart.shape[:-1],dtype = bool)
        asic_slices = self.geometry['asic_slices']
        for asic in asics:
            #log.info('asic ={}'.format(asic))
            slices = asic_slices[asic[1]][asic[2]]
            module = asic[0]
            mask[module,slices[0],slices[1]] = True
        return mask

class All(RegionOfInterest):
    def generate(self):
        mask = np.ones(self.n_total_modules,dtype = bool)
        return mask


class ROIManager():
    def __init__(self,geometry,rois_dict={},used_rois = [],data_modules=np.arange(16)):
        self.geometry=geometry
        self.rois = {'all':All({},geometry)}
        self._current_module = sys.modules[__name__]
        self.add_rois(rois_dict)
        self._used_modules = tuple()
        self._used_rois=[]
        self.used_rois=used_rois
        self._update_used_modules_from_used_rois()
        
    def add_rois(self,_dict):
        current_module = self._current_module
        for key,val in _dict.items():
            name = key
            _type = val['class']
            parameters = val['parameters']
            geometry = self.geometry
            try:
                #log.info(name)
                self.rois[name]= getattr(current_module,_type[0].upper()+_type[1:])(parameters,geometry)
            except AttributeError as e:
                log.warning('ROI type {}  of roi named {} not found.Continue.'.format(_type,name))
                
    def _update_used_modules_from_used_rois(self):
        modules = tuple()
        for name in self._used_rois:
            try:
                roi = self.rois[name]
                roi_modules = roi.mask_true_modules
                modules += tuple(roi_modules)                
            except KeyError as e:
                log.warning('ROI {} not found Known rois are {}.Skipping.'.format(name,self.rois.keys()))
        used_modules = np.unique(modules)
        self.used_modules = used_modules
        
    @property
    def used_modules(self):
        return self._used_modules
    @used_modules.setter
    def used_modules(self,modules):
        if len(modules)>0:
            self._used_modules = modules
            #1log.info(modules)
            for roi in self.rois.values():
                roi.used_modules = np.asarray(modules)
    @property
    def used_rois(self):
        return self._used_rois
    @used_rois.setter
    def used_rois(self,roi_names):
        new_used_rois = []
        for name in roi_names:
            if name in self.rois:
                new_used_rois.append(name)
        self._used_rois = new_used_rois
    
    def get_combined_roi_mask(self,roi_names):
        if len(roi_names)==0:
            mask = np.array(True)
        else:
            mask = False
            for name in roi_names:
                roi = self.rois[name]
                mask |= roi.mask
        return mask
