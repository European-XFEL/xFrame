import logging
import sys
import os
import numpy as np
import traceback
from scipy.stats import unitary_group
from .extract import InvariantExtractor
file_path = os.path.realpath(__file__)
plugin_dir = os.path.dirname(file_path)
os.chdir(plugin_dir)


#from analysisLibrary.classes import ReciprocalProjectionData
from .projectLibrary import fxs_invariant_tools as i_tools
from .projectLibrary.ft_grid_pairs import max_order_from_n_angular_steps,get_grid,get_polar_fft_angles_from_max_order
from .projectLibrary.fourier_transforms import fourier_transform_from_settings
from .projectLibrary.harmonic_transforms import HarmonicTransform
from .projectLibrary.hankel_transforms import generate_weightDict
from xframe.library.math_transforms import SphericalFourierTransform,SphericalFourierTransformStruct
from .projectLibrary.misk import generate_calc_center
from .projectLibrary.fxs_Projections import generate_shift_by_operator
from xframe.library.pythonLibrary import xprint
from xframe.library.gridLibrary import SampledFunction,NestedArray,GridFactory
from xframe.library.mathLibrary import nearest_positive_semidefinite_matrix
from xframe.library.mathLibrary import polar_spherical_dft_reciprocity_relation_radial_cutoffs,distance_from_line_2d
from xframe.interfaces import ProjectWorkerInterface
from xframe import database,settings
from xframe.library.mathLibrary import SampleShapeFunctions
from .projectLibrary.misk import _get_reciprocity_coefficient
from xframe import Multiprocessing
from xframe.library.mathLibrary import spherical_to_cartesian
log=logging.getLogger('root')

#class Worker(RecipeInterface):
class ProjectWorker(ProjectWorkerInterface):
    def __init__(self):
        self.extractor = CCgenerator()
    #       global db
    #        db = database.project
    def run(self):
        db = database.project
        opt = settings.project
        
        self.extractor.extract()
            
        if self.extractor.success:
            xprint('Saving results')
            db.save('ccd', self.extractor.cc_data,model_density_pair = self.extractor.density_pair,grids=[self.extractor.ft.real_grid,self.extractor.ft.reciprocal_grid])
            
        return {},locals()


######### InvariantExtractor #########
class CCgenerator(InvariantExtractor):
    def __init__(self):
        super().__init__()

    ##################################
    ###   calc cross-correlation   ###
    def calc_cross_correlation(self):
        opt = settings.project
        mode = opt.cross_correlation.method
        data_grid = {'qs':self.data_radial_points,'phis':self.data_angular_points}
        wavelength = opt.cross_correlation.xray_wavelength
        bl = self.b_coeff['I1I1']
        if opt.dimensions ==3:
            cc = i_tools.deg2_invariant_to_cc_3d(bl,wavelength,data_grid,mode= mode,n_processes = opt.multi_process.n_processes)
        elif opt.dimensions ==2:
            cc = i_tools.deg2_invariant_to_cc_2d(bl,self.cht)
        self.cross_correlation = cc
        
    def create_density_from_shape_settings(self,real_grid):
        opt = settings.project
        
        density = np.zeros(real_grid.shape[:-1],dtype =float)

        types = np.asarray(opt.shapes.types)
        centers = np.asarray(opt.shapes.centers)
        sizes = np.asarray(opt.shapes.sizes)
        density_values = np.asarray(opt.shapes.densities)
        random_orientation = np.asarray(opt.shapes.random_orientation)
        for shape_type,center,size,dval,rand_rot in zip(types,centers,sizes,density_values,random_orientation):
            log.info(f'\n t {type}\n c {center}\n s {size}\n d {dval}\n r {random_orientation}')
            if shape_type == 'sphere':
                norm = 'standard'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot,coordSys='spherical')
            elif shape_type == 'tetrahedron':
                f = SampleShapeFunctions.get_tetrahedral_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,random_orientation=rand_rot)
            elif shape_type == 'cube':
                norm = 'inf'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            else:
                norm = 'inf'
                f = SampleShapeFunctions.get_disk_function(size,lambda points: np.full(points.shape[:-1],dval),center=center,norm=norm,random_orientation=rand_rot)
            density += f(real_grid)
        return density
        
    ##################
    ## main routine ##
    def extract(self):
        opt = settings.project
        self.dimentsions = opt.dimensions
        self.max_order = opt.grid.max_order
        
        self.ft = fourier_transform_from_settings()
        self.data_radial_points = self.ft.qs
        self.data_angular_points = self.ft.phis
        xprint('Creating density model:')
        density = self.create_density_from_shape_settings(self.ft.real_grid)
        self.density_pair = [density,self.ft.forward_cmplx(density.astype(complex))]
        xprint('done.\n')

        xprint('Calculating Cross-Correlation:')
        self._bl_from_density(density,ft=self.ft)

        self.calc_cross_correlation()
        xprint('done.\n')
        
        cc_data = {}
        cc_data['radial_points']=self.data_radial_points
        cc_data['angular_points']=self.data_angular_points
        cc_data['xray_wavelength']=opt.cross_correlation.xray_wavelength
        cc_data['cross_correlation']= {'I1I1':self.cross_correlation}
        cc_data['average_intensity']=self.average_intensity.data
        cc_data['deg_2_invariant']= {"I1I1":self.b_coeff['I1I1']}
        cc_data['number_of_particles']=self.number_of_particles
        cc_data['dimensions']=opt.dimensions
        
        self.cc_data = cc_data

        self.success =True
        #log.info(self.data_projection_matrices['I1I1'])
        log.info('Extraction completed! \n')
        
