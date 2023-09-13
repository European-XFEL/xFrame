import tempfile
import os
import numpy as np

import logging
import pdb_eda
from xframe.library.gridLibrary import GridFactory
from xframe.library.pythonLibrary import DictNamespace
from xframe.database.interfaces import PDBInterface
from xframe.library.mathLibrary import spherical_to_cartesian,gaussian_fft_cart,get_fft_reciprocal_grid,convolve_with_gaussian_cart,SphericalIntegrator
log=logging.getLogger('root')

class ProteinDB(PDBInterface):
    lib = pdb_eda
    @staticmethod
    def load(pdb_id : str,**kwargs):
        '''
        Loads density and grid of a pdb entry. Needs internet connection to work. 
        throws assertion error if no density is accessible for given pdb entry or if id does not exist.
        :param pdb_id: Proteine Id on https://www.rcsb.org/ e.g.: '1KP8' for GroEL 
        :type str: 
        :return density_and_grid: 
        :rtype DictNamespace: keys()=('density','grid')
        '''
        return DensityExtractor(pdb_id)
        
    @staticmethod
    def save(path,data,**kwargs):
        e=NotImplementedError('Saving to PDB not possible.')
        raise e

class DensityExtractor:
    def __init__(self,pdb_id):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            log.info('Start loading files from PDB. This might take a minute..')
            densityAnalysisObj = pdb_eda.densityAnalysis.fromPDBid(pdb_id,downloadFile = False,pdbi=True,mmcif=False,pdbbio=True)
            assert densityAnalysisObj != 0, 'given pdb_id = {} dies not exist or does not contain 2fo-fc density data'.format(pdb_id)
            self._densities= DictNamespace(
                fo = densityAnalysisObj.fo.density.T,
                fc = densityAnalysisObj.fc.density.T,
                two_fofc = densityAnalysisObj.densityObj.density.T,
                fofc = densityAnalysisObj.diffDensityObj.density.T
            )
            h=densityAnalysisObj.densityObj.header
            self.header_part = DictNamespace(
                n_crs = np.asarray(h.ncrs),
                grid_length = np.asarray(h.gridLength),
                origin = np.asarray(h.origin),
                xyz_interval = np.asarray(h.xyzInterval),
                crs_start = np.asarray(h.crsStart),
                map_to_xyz = np.asarray(h.map2xyz),
                map_to_crs = np.asarray(h.map2crs),
                ortho_mat = np.asarray(h.orthoMat),
                inv_ortho_mat = np.asarray(h.deOrthoMat),
                alpha = h.alpha,
                beta = h.beta,
                gamma = h.gamma        
            )
            self._atom_coords_xyz = np.asarray(densityAnalysisObj.asymmetryAtomCoords)

        self.atom_coords = DictNamespace(
            xyz = self._atom_coords_xyz,
            crs = self.xyz2crsCoord(self._atom_coords_xyz)
        )
        self.bounds = DictNamespace(
            xyz = np.array(tuple((self.atom_coords.xyz[...,n].min(),self.atom_coords.xyz[...,n].max()) for n in range(3))),
            crs = np.array(tuple((self.atom_coords.crs[...,n].min(),self.atom_coords.crs[...,n].max()) for n in range(3)))            
        )
        self.max_r = np.linalg.norm(self.atom_coords.xyz,axis= -1).max()
        os.chdir(cwd)
                
    def crs2xyzCoord(self,crs):
        '''
        Convert the crs coordinates into xyz coordinates.
        Assumes crs is of shape (N,3) where N is the number of coordinate sets.
        '''
        h = self.header_part
        crs = np.atleast_2d(crs)
        if h.alpha == h.beta == h.gamma == 90:
            xyz = crs[:,h.map_to_xyz]*h.grid_length[None,:]+h.origin[None,:]
            
        else:
            xyz = np.dot(h.ortho_mat, ((crs[:,h.map_to_xyz]+h.crs_start[None,h.map_to_xyz])/h.xyz_interval[None,:]).T).T
        return np.squeeze(xyz)
    
    def xyz2crsCoord(self,xyz):
        '''
        Converts xyz to crs coordinates. 
        Assumes xyz is of shape (N,3) where N is the number of coordinate sets.
        '''
        h = self.header_part
        xyz = np.atleast_2d(xyz)
        if h.alpha == h.beta == h.gamma == 90:
            crsGridPos =  np.rint((xyz - h.origin[None,:])/h.grid_length[None,:])[:,h.map_to_crs]
        else:
            fraction = np.dot(h.inv_ortho_mat, xyz.T).T
            crsGridPos = (np.rint(fraction*h.xyz_interval[None,:]) - h.crs_start[None,h.map_to_xyz])[:,h.map_to_crs]
        return np.squeeze(crsGridPos).astype(int)

    def sample_density_xyz(self,crs_density,points,fill_value = 0.0):
        '''
        Samples density at given xyz coordinates.
        '''
        shape = points.shape
        crs_points = self.xyz2crsCoord(points.reshape(-1,3))
        sampled_density = self.sample_density_crs(crs_density,crs_points,fill_value = fill_value).reshape(shape[:-1])
        return sampled_density
    
    def sample_density_crs(self,crs_density,points,fill_value = 0.0):
        '''
        Samples density at given crs coordinates.
        fills defaut value for crs coordinates which are smaller than 0 and bigger than self.n_crs.
        '''
        points = np.atleast_2d(points)
        shape = points.shape
        log.info('shape crs = {}'.format(shape))
        valid_crs_points , mask = self.return_valid_crs_coords(points.reshape(-1,3),return_mask= True)

        sampled_density=np.full(points.shape[:-1],fill_value,dtype = crs_density.dtype)
        log.info('sample density shape = {}'.format(sampled_density.shape))
        sampled_density[mask] = crs_density[valid_crs_points[:,0],valid_crs_points[:,1],valid_crs_points[:,2]]
        return sampled_density.reshape(shape[:-1])
    def return_valid_crs_coords(self,coords,return_mask = False):
        '''
        Returns all coordinates tripples that satisfy  0<= val[i] < heade.n_crs[i] for i in 0,1,2.
        '''
        valid_mask = (np.prod( (coords < self.header_part.n_crs[None,:]) * (coords >= 0),axis=-1)).astype(bool)
        coords = coords[valid_mask,:]
        if return_mask:
            return coords,valid_mask
        else:
            return coords
        
    def generate_density_mask(self,atom_neighbour_shells=2):
        '''
        Generates a mask around each atom of the PDB structure. (To get rid of symmetry parts of the density fuction.)
        :param atom_neighbor_shells: Number of cubical shells around a given crs coordinate which to include in the density
        :type int: 
        '''
        mask = np.full(self.header_part.n_crs,False)
        if atom_neighbour_shells >=1:
            crs = self.atom_coords.crs
            neighbours_1d = np.arange(-atom_neighbour_shells ,atom_neighbour_shells + 1)
            neighbours_3d = GridFactory.construct_grid('uniform',[neighbours_1d,neighbours_1d,neighbours_1d]).flatten().array
            valid_grid_positions = np.unique(self.return_valid_crs_coords((crs[:,None,:] + neighbours_3d[None,...]).reshape(-1,3)),axis = 0)        
            mask[valid_grid_positions.T[0],valid_grid_positions.T[1],valid_grid_positions.T[2]]=True
        else:
            mask[:]=True
        return mask

    def convolve_density_with_gaussian(self,crs_density,sigma):
        '''
        Convolves a density in self._densities with a gaussian.
        '''
        n_crs = self.header_part.n_crs        
        n_steps = 2**(np.log2(2*n_crs.max()) + 0.5).astype(int)
        grid_bounds = np.max(np.abs(self.bounds.xyz),axis=-1)
        step_size = grid_bounds/(n_steps//2)
        
        steps = [np.arange(-n_steps//2+1,n_steps//2+1) * step_size[i] for i in range(3)]        
        cart_grid = GridFactory.construct_grid('uniform',steps).array
        density_cart = self.sample_density_xyz(crs_density,cart_grid)
        
        convolved_density_cart = convolve_with_gaussian_cart(density_cart,cart_grid,sigma)
        log.info('convolved .shape = {}'.format(convolved_density_cart.shape))

        crs_grid = GridFactory.construct_grid('uniform',[np.arange(n) for n in n_crs]).array
        xyz_grid = self.crs2xyzCoord(crs_grid.reshape(-1,3)).reshape(crs_grid.shape)
        mask = np.prod((xyz_grid >= -grid_bounds[None,None,None,:]) & (xyz_grid <= grid_bounds[None,None,None,:]),axis = -1).astype(bool)
       # log.info('mas is true = {}'.format(mask.all()))
        #log.info("{} masked ccordinates".format(sum(~mask.flatten())))
        nearest_neighbor_indices = ((xyz_grid+grid_bounds[None,None,None,:])//step_size[None,None,None,:]).astype(int)
        #log.info('{} indices bigger than 128!'.format(np.sum(nearest_neighbor_indices>=128)))
        #log.info('mask shape = {}'.format(mask.shape))
        nni = nearest_neighbor_indices[mask]
        #log.info('{} indices bigger than 128!'.format(np.sum(nni>=128)))
        #log.info('nni shape unmasked = {}, nni shape masked = {}'.format(nearest_neighbor_indices.shape,nni.shape))

        convolved_density = np.zeros(n_crs,dtype = density_cart.dtype)
        convolved_density[mask] = convolved_density_cart[nni[:,0],nni[:,1],nni[:,2]].real
        #convolved_density[:] = convolved_density_cart[:].real
        return convolved_density

    def convolve_density_with_gaussian2(self,crs_density,sigma):
        '''
        Convolves a density in self._densities with a gaussian.
        '''
        n_crs = self.header_part.n_crs        
        n_steps = 2**(np.log2(2*n_crs.max()) + 0.5).astype(int)
        grid_bounds = np.max(np.abs(self.bounds.xyz),axis=-1)
        step_size = grid_bounds/(n_steps//2)
        
        steps = [np.arange(-n_steps//2+1,n_steps//2+1) * step_size[i] for i in range(3)]        
        cart_grid = GridFactory.construct_grid('uniform',steps).array
        density_cart = self.sample_density_xyz(crs_density,cart_grid)
        
        convolved_density_cart = convolve_with_gaussian_cart(density_cart,cart_grid,sigma)
        log.info('convolved .shape = {}'.format(convolved_density_cart.shape))

        crs_grid = GridFactory.construct_grid('uniform',[np.arange(n) for n in n_crs]).array
        xyz_grid = self.crs2xyzCoord(crs_grid.reshape(-1,3)).reshape(crs_grid.shape)
        mask = np.prod((xyz_grid >= -grid_bounds[None,None,None,:]) & (xyz_grid <= grid_bounds[None,None,None,:]),axis = -1).astype(bool)
       # log.info('mas is true = {}'.format(mask.all()))
        #log.info("{} masked ccordinates".format(sum(~mask.flatten())))
        nearest_neighbor_indices = ((xyz_grid+grid_bounds[None,None,None,:])//step_size[None,None,None,:]).astype(int)
        #log.info('{} indices bigger than 128!'.format(np.sum(nearest_neighbor_indices>=128)))
        #log.info('mask shape = {}'.format(mask.shape))
        nni = nearest_neighbor_indices[mask]
        #log.info('{} indices bigger than 128!'.format(np.sum(nni>=128)))
        #log.info('nni shape unmasked = {}, nni shape masked = {}'.format(nearest_neighbor_indices.shape,nni.shape))

        convolved_density = np.zeros(n_crs,dtype = density_cart.dtype)
        convolved_density[mask] = convolved_density_cart[nni[:,0],nni[:,1],nni[:,2]].real
        #convolved_density[:] = convolved_density_cart[:].real
        return convolved_density
    
    def extract_density(self,name,grid,atom_neighbour_shells = 2, gaussian_sigma = False ):
        '''
        Extracts the given density profile ("fo","fc","two_fofc","fofc") 
        '''
        integrator = SphericalIntegrator(grid)
        crs_density = self._densities[name]
        density_mask = self.generate_density_mask(atom_neighbour_shells = atom_neighbour_shells)
        log.info('densitymask dtype = {}'.format(density_mask.dtype))
        crs_density[~density_mask] = 0
        if not isinstance(gaussian_sigma ,bool):
            crs_density = self.convolve_density_with_gaussian(crs_density,float(gaussian_sigma))


        rs = grid[:,0,0,0]
        r_mask = rs<self.max_r
        cart_grid = spherical_to_cartesian(grid)
        
        
        density = np.zeros(cart_grid.shape[:-1])
        cart_support = cart_grid[r_mask,...]
        support_shape = cart_support.shape
        crs_support = self.xyz2crsCoord(cart_support.reshape(-1,3)).reshape(*support_shape)
        crs_support,mask = self.return_valid_crs_coords(crs_support,return_mask=True)
        supported_density_part = crs_density[crs_support[:,0],crs_support[:,1],crs_support[:,2]]

        temp = density[r_mask]
        temp[mask] = supported_density_part
        density[r_mask] = temp
        #density *= np.sqrt(1/integrator.integrate(density**2))
        return density

class DensityExtractor_old:
    def __init__(self,pdb_id):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            log.info('Start loading files from PDB. This might take a minute..')
            densityAnalysisObj = pdb_eda.densityAnalysis.fromPDBid(pdb_id,downloadFile = False,pdbi=True,mmcif=False,pdbbio=True)
            assert densityAnalysisObj != 0, 'given pdb_id = {} dies not exist or does not contain 2fo-fc density data'.format(pdb_id)
            self._densities= DictNamespace(
                fo = densityAnalysisObj.fo.density.T,
                fc = densityAnalysisObj.fc.density.T,
                two_fcfo = densityAnalysisObj.densityObj.density.T,
                fcfo = densityAnalysisObj.diffDensityObj.density.T
            )
            h=densityAnalysisObj.densityObj.header
            self.header_part = DictNamespace(
                n_crs = np.asarray(h.ncrs),
                grid_length = np.asarray(h.gridLength),
                origin = np.asarray(h.origin),
                xyz_interval = np.asarray(h.xyzInterval),
                crs_start = np.asarray(h.crsStart),
                map_to_xyz = np.asarray(h.map2xyz),
                map_to_crs = np.asarray(h.map2crs),
                ortho_mat = np.asarray(h.orthoMat),
                inv_ortho_mat = np.asarray(h.deOrthoMat),
                alpha = h.alpha,
                beta = h.beta,
                gamma = h.gamma        
            )
            self._atom_coords_xyz = np.asarray(densityAnalysisObj.asymmetryAtomCoords)

        self.atom_coords = DictNamespace(
            xyz = self._atom_coords_xyz,
            crs = self.xyz2crsCoord(self._atom_coords_xyz)
        )
        self.bounds = DictNamespace(
            xyz = np.array(tuple((self.atom_coords.xyz[...,n].min(),self.atom_coords.xyz[...,n].max()) for n in range(3))),
            crs = np.array(tuple((self.atom_coords.crs[...,n].min(),self.atom_coords.crs[...,n].max()) for n in range(3)))            
        )
        self.max_r = np.linalg.norm(self.atom_coords.xyz,axis= -1).max()
        os.chdir(cwd)
                
    def crs2xyzCoord(self,crs):
        '''
        Convert the crs coordinates into xyz coordinates.
        Assumes crs is of shape (N,3) where N is the number of coordinate sets.
        '''
        h = self.header_part
        crs = np.atleast_2d(crs)
        if h.alpha == h.beta == h.gamma == 90:
            xyz = crs[:,h.map_to_xyz]*h.grid_length[None,:]+h.origin[None,:]
            
        else:
            xyz = np.dot(h.ortho_mat, ((crs[:,h.map_to_xyz]+h.crs_start[None,h.map_to_xyz])/h.xyz_interval[None,:]).T).T
        return np.squeeze(xyz)
    
    def xyz2crsCoord(self,xyz):
        '''
        Converts xyz to crs coordinates. 
        Assumes xyz is of shape (N,3) where N is the number of coordinate sets.
        '''
        h = self.header_part
        xyz = np.atleast_2d(xyz)
        if h.alpha == h.beta == h.gamma == 90:
            crsGridPos =  np.rint((xyz - h.origin[None,:])/h.grid_length[None,:])[:,h.map_to_crs]
        else:
            fraction = np.dot(h.inv_ortho_mat, xyz.T).T
            crsGridPos = (np.rint(fraction*h.xyz_interval[None,:]) - h.crs_start[None,h.map_to_xyz])[:,h.map_to_crs]
        return np.squeeze(crsGridPos).astype(int)

    def sample_density_xyz(self,crs_density,points,fill_value = 0.0):
        '''
        Samples density at given xyz coordinates.
        '''
        shape = points.shape
        crs_points = self.xyz2crsCoord(points.reshape(-1,3))
        sampled_density = self.sample_density_crs(crs_density,crs_points,fill_value = fill_value).reshape(shape[:-1])
        return sampled_density
    
    def sample_density_crs(self,crs_density,points,fill_value = 0.0):
        '''
        Samples density at given crs coordinates.
        fills defaut value for crs coordinates which are smaller than 0 and bigger than self.n_crs.
        '''
        points = np.atleast_2d(points)
        shape = points.shape
        log.info('shape crs = {}'.format(shape))
        valid_crs_points , mask = self.return_valid_crs_coords(points.reshape(-1,3),return_mask= True)

        sampled_density=np.full(points.shape[:-1],fill_value,dtype = crs_density.dtype)
        log.info('sample density shape = {}'.format(sampled_density.shape))
        sampled_density[mask] = crs_density[valid_crs_points[:,0],valid_crs_points[:,1],valid_crs_points[:,2]]
        return sampled_density.reshape(shape[:-1])
    def return_valid_crs_coords(self,coords,return_mask = False):
        '''
        Returns all coordinates tripples that satisfy  0<= val[i] < heade.n_crs[i] for i in 0,1,2.
        '''
        valid_mask = (np.prod( (coords < self.header_part.n_crs[None,:]) * (coords >= 0),axis=-1)).astype(bool)
        coords = coords[valid_mask,:]
        if return_mask:
            return coords,valid_mask
        else:
            return coords
        
    def generate_density_mask(self,atom_neighbour_shells=2):
        '''
        Generates a mask around each atom of the PDB structure. (To get rid of symmetry parts of the density fuction.)
        :param atom_neighbor_shells: Number of cubical shells around a given crs coordinate which to include in the density
        :type int: 
        '''
        crs = self.atom_coords.crs
        neighbours_1d = np.arange(-atom_neighbour_shells ,atom_neighbour_shells + 1)
        neighbours_3d = GridFactory.construct_grid('uniform',[neighbours_1d,neighbours_1d,neighbours_1d]).flatten().array
        valid_grid_positions = np.unique(self.return_valid_crs_coords((crs[:,None,:] + neighbours_3d[None,...]).reshape(-1,3)),axis = 0)
        mask = np.full(self.header_part.n_crs,False)
        mask[valid_grid_positions.T[0],valid_grid_positions.T[1],valid_grid_positions.T[2]]=True
        return mask

    def convolve_density_with_gaussian(self,crs_density,sigma):
        '''
        Convolves a density in self._densities with a gaussian.
        '''
        n_crs = self.header_part.n_crs        
        n_steps = 2**(np.log2(2*n_crs.max()) + 0.5).astype(int)
        grid_bounds = np.max(np.abs(self.bounds.xyz),axis=-1)
        step_size = grid_bounds/(n_steps//2)
        
        steps = [np.arange(-n_steps//2+1,n_steps//2+1) * step_size[i] for i in range(3)]        
        cart_grid = GridFactory.construct_grid('uniform',steps).array
        density_cart = self.sample_density_xyz(crs_density,cart_grid)
        
        convolved_density_cart = convolve_with_gaussian_cart(density_cart,cart_grid,sigma)
        log.info('convolved .shape = {}'.format(convolved_density_cart.shape))

        crs_grid = GridFactory.construct_grid('uniform',[np.arange(n) for n in n_crs]).array
        xyz_grid = self.crs2xyzCoord(crs_grid.reshape(-1,3)).reshape(crs_grid.shape)
        mask = np.prod((xyz_grid >= -grid_bounds[None,None,None,:]) & (xyz_grid <= grid_bounds[None,None,None,:]),axis = -1).astype(bool)
       # log.info('mas is true = {}'.format(mask.all()))
        #log.info("{} masked ccordinates".format(sum(~mask.flatten())))
        nearest_neighbor_indices = ((xyz_grid+grid_bounds[None,None,None,:])//step_size[None,None,None,:]).astype(int)
        #log.info('{} indices bigger than 128!'.format(np.sum(nearest_neighbor_indices>=128)))
        #log.info('mask shape = {}'.format(mask.shape))
        nni = nearest_neighbor_indices[mask]
        #log.info('{} indices bigger than 128!'.format(np.sum(nni>=128)))
        #log.info('nni shape unmasked = {}, nni shape masked = {}'.format(nearest_neighbor_indices.shape,nni.shape))

        convolved_density = np.zeros(n_crs,dtype = density_cart.dtype)
        convolved_density[mask] = convolved_density_cart[nni[:,0],nni[:,1],nni[:,2]].real
        #convolved_density[:] = convolved_density_cart[:].real
        return convolved_density

    def convolve_density_with_gaussian2(self,crs_density,sigma):
        '''
        Convolves a density in self._densities with a gaussian.
        '''
        n_crs = self.header_part.n_crs        
        n_steps = 2**(np.log2(2*n_crs.max()) + 0.5).astype(int)
        grid_bounds = np.max(np.abs(self.bounds.xyz),axis=-1)
        step_size = grid_bounds/(n_steps//2)
        
        steps = [np.arange(-n_steps//2+1,n_steps//2+1) * step_size[i] for i in range(3)]        
        cart_grid = GridFactory.construct_grid('uniform',steps).array
        density_cart = self.sample_density_xyz(crs_density,cart_grid)
        
        convolved_density_cart = convolve_with_gaussian_cart(density_cart,cart_grid,sigma)
        log.info('convolved .shape = {}'.format(convolved_density_cart.shape))

        crs_grid = GridFactory.construct_grid('uniform',[np.arange(n) for n in n_crs]).array
        xyz_grid = self.crs2xyzCoord(crs_grid.reshape(-1,3)).reshape(crs_grid.shape)
        mask = np.prod((xyz_grid >= -grid_bounds[None,None,None,:]) & (xyz_grid <= grid_bounds[None,None,None,:]),axis = -1).astype(bool)
       # log.info('mas is true = {}'.format(mask.all()))
        #log.info("{} masked ccordinates".format(sum(~mask.flatten())))
        nearest_neighbor_indices = ((xyz_grid+grid_bounds[None,None,None,:])//step_size[None,None,None,:]).astype(int)
        #log.info('{} indices bigger than 128!'.format(np.sum(nearest_neighbor_indices>=128)))
        #log.info('mask shape = {}'.format(mask.shape))
        nni = nearest_neighbor_indices[mask]
        #log.info('{} indices bigger than 128!'.format(np.sum(nni>=128)))
        #log.info('nni shape unmasked = {}, nni shape masked = {}'.format(nearest_neighbor_indices.shape,nni.shape))

        convolved_density = np.zeros(n_crs,dtype = density_cart.dtype)
        convolved_density[mask] = convolved_density_cart[nni[:,0],nni[:,1],nni[:,2]].real
        #convolved_density[:] = convolved_density_cart[:].real
        return convolved_density
    
    def extract_density(self,name,grid,atom_neighbour_shells = 2, gaussian_sigma = False ):
        '''
        Extracts the given density profile ("fo","fc","two_fofc","fofc") 
        '''
        integrator = SphericalIntegrator(grid)
        crs_density = self._densities[name]
        density_mask = self.generate_density_mask(atom_neighbour_shells = atom_neighbour_shells)
        log.info('densitymask dtype = {}'.format(density_mask.dtype))
        crs_density[~density_mask] = 0
        if not isinstance(gaussian_sigma ,bool):
            crs_density = self.convolve_density_with_gaussian(crs_density,float(gaussian_sigma))


        rs = grid[:,0,0,0]
        r_mask = rs<self.max_r
        cart_grid = spherical_to_cartesian(grid)
        
        
        density = np.zeros(cart_grid.shape[:-1])
        cart_support = cart_grid[r_mask,...]
        support_shape = cart_support.shape
        crs_support = self.xyz2crsCoord(cart_support.reshape(-1,3)).reshape(*support_shape)
        crs_support,mask = self.return_valid_crs_coords(crs_support,return_mask=True)
        supported_density_part = crs_density[crs_support[:,0],crs_support[:,1],crs_support[:,2]]

        temp = density[r_mask]
        temp[mask] = supported_density_part
        density[r_mask] = temp
        #density *= np.sqrt(1/integrator.integrate(density**2))
        return density
        
        
