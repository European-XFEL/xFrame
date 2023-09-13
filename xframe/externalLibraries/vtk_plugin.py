import numpy as np
import logging
import vtk
from vtk.util import numpy_support as vtn
from xframe.library.gridLibrary import GridFactory
from xframe.library.gridLibrary import NestedArray
from xframe.database.interfaces import VTKInterface
log=logging.getLogger('root')

class vtkSaver(VTKInterface):
    lib = vtk    
    dset_names_default = 'data'
    default_grid = False
    file_type_dict={'cartesian':'vtr','spherical':'vts','polar':'vts'}
    @staticmethod
    def correct_file_path_ending(path,grid_type='cartesian'):
        old_path = path
        file_type = vtkSaver.file_type_dict.get(grid_type,'vtr')
        ending = path.split('.')[-1]
        if len(ending) == 3:
            path = path[:-3] + file_type
        else:
            path = path + '.'+ file_type
        #log.info('grid_type = {}\n path = {} \n corrected path = {} \n file_type = {}'.format(grid_type,old_path,path,file_type))
        return path
    @staticmethod
    def save(file_path,datasets, grid=default_grid,dset_names = dset_names_default, grid_type='cartesian',**kwargs):
        '''
        Saves vtk 3d models which can be opened e.g with Paraview.
        Currenty the types 'cartesian' and 'spherical' are supported.
        They correspond to uniform cartesian and sperical grids.
        Throws assertion error if unknown grid type is provided.
        '''
        try:
            assert len(datasets)>0,'No datasets provided, i.e.: len(datasets)=0'
            writer = False
            for t,get_writer in vtkSaver.writer_dict.items():
                if t==grid_type:
                    writer = get_writer(datasets,grid = grid,dset_names=dset_names)
                    file_path = vtkSaver.correct_file_path_ending(file_path,grid_type=grid_type)
                    break
            assert ~isinstance(writer,bool),'grid_type {} not known. Known grid_types are {}'.format(grid_type,vtkSaver.writer_dict.keys())

            
            writer.SetFileName(file_path)
            writer.Update()
        except AssertionError as e:
            log.error(e)
            
    @staticmethod
    def load(path,**kwargs):
        e=NotImplementedError('loading of VTK filetypes is not supported yet.')
        raise e
            


    @staticmethod
    def generate_spherical_writer( datasets, grid=default_grid, dset_names=dset_names_default):
        for n,data in enumerate(datasets):
            new_data = np.zeros((data.shape[0],)+(data.shape[1]+2,)+(data.shape[-1]+1,))
            new_data[:,1:-1,:-1] = data
            new_data[:,1:-1,-1] = data[:,:,0]
            new_data[:,0,:-1] = np.mean(data[:,0,:],axis = 1)[:,None]
            new_data[:,-1,:-1] = np.mean(data[:,-1,:],axis = 1)[:,None]
            datasets[n]=new_data
        if not (isinstance(grid,np.ndarray) or isinstance(grid,NestedArray)):
            n_r,n_theta,n_phi = datasets[0].shape
            rs = np.arange(n_r)*1/n_r
            phis = np.arange(n_phi-1)*np.pi*2/(n_phi-1)
            thetas = (np.arange(n_theta)*np.pi/(n_theta-1))[1:-1]
            grid = GridFactory.construct_grid('uniform',[rs,thetas,phis])
        phis = np.zeros(len(grid[0,0,:,2])+1,grid[0,0,:,2].dtype)
        phis[:-1]=grid[0,0,:,2]
        phis[-1] = 0 # periodic point
        thetas = np.zeros(len(grid[0,:,0,1])+2,grid[0,:,0,1].dtype)
        thetas[1:-1]=grid[0,:,0,1]
        thetas[0]=0
        thetas[-1]=np.pi
        grid= np.stack(np.meshgrid(phis,thetas,grid[:,0,0,0],indexing='ij'),3) 
        grid[...,1]-=np.pi/2
        xy_projection= grid[...,2]*np.cos(grid[...,1])
        x=xy_projection*np.cos(grid[...,0])
        y=xy_projection*np.sin(grid[...,0])
        z=grid[...,2]*np.sin(grid[...,1])
    
        points = np.empty((x.size, 3), x.dtype)
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')


        points = np.require(points, requirements=['C'])
        vtkpts = vtk.vtkPoints()
        vtk_arr = vtn.numpy_to_vtk(points, deep=True)
        vtkpts.SetData(vtk_arr)

        vtk_grid = vtk.vtkStructuredGrid()
        vtk_grid.SetDimensions(*grid.shape[:3])
        vtk_grid.SetPoints(vtkpts)
    
        for n,density in enumerate(datasets):
            density = np.swapaxes(density.copy(),0,2)
            density = np.require(density, requirements=['C']).real
            vtk_density = vtn.numpy_to_vtk(num_array=np.swapaxes(density,0,2).copy().ravel(), deep=True, array_type=vtk.VTK_FLOAT)    
            vtk_grid.GetPointData().AddArray(vtk_density)
            # print grid.GetPointData().GetNumberOfArrays()
            if isinstance(dset_names,str):
                vtk_density.SetName("{}_{}".format(dset_names,n))
            elif isinstance(dset_names,(list,tuple)):
                vtk_density.SetName("{}_{}".format(dset_names[n],n))                        
        
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(vtk_grid)
        return writer


    @staticmethod
    def generate_polar_vtkStructuredGrid(datasets, grid=default_grid, dset_names=dset_names_default):
        for n,data in enumerate(datasets):
            new_data = np.zeros((data.shape[0],)+(1,)+(data.shape[1]+1,))
            new_data[:,0,:-1] = data
            new_data[:,0,-1] = data[:,0]
            datasets[n]=new_data
        if not (isinstance(grid,np.ndarray) or isinstance(grid,NestedArray)):
            n_r,n_phi = datasets[0].shape
            n_theta = 1
            rs = np.arange(n_r)*1/n_r
            phis = np.arange(n_phi-1)*np.pi*2/(n_phi-1)
            thetas = np.array([np.pi/2])
            grid = GridFactory.construct_grid('uniform',[rs,thetas,phis])
        phis = np.zeros(len(grid[0,:,1])+1,grid[0,:,1].dtype)
        phis[:-1]=grid[0,:,1]
        phis[-1] = 0 # periodic point
        thetas = np.array([np.pi/2])
        #thetas[-1]=np.pi
        grid= np.stack(np.meshgrid(phis,thetas,grid[:,0,0],indexing='ij'),3) 
        grid[...,1]-=np.pi/2
        xy_projection= grid[...,2]*np.cos(grid[...,1])
        x=xy_projection*np.cos(grid[...,0])
        y=xy_projection*np.sin(grid[...,0])
        z=grid[...,2]*np.sin(grid[...,1])
    
        points = np.empty((x.size, 3), x.dtype)
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')


        points = np.require(points, requirements=['C'])
        vtkpts = vtk.vtkPoints()
        vtk_arr = vtn.numpy_to_vtk(points, deep=True)
        vtkpts.SetData(vtk_arr)

        vtk_grid = vtk.vtkStructuredGrid()
        vtk_grid.SetDimensions(*grid.shape[:3])
        vtk_grid.SetPoints(vtkpts)
    
        for n,density in enumerate(datasets):
            density = np.swapaxes(density.copy(),0,2)
            density = np.require(density, requirements=['C']).real
            vtk_density = vtn.numpy_to_vtk(num_array=np.swapaxes(density,0,2).copy().ravel(), deep=True, array_type=vtk.VTK_FLOAT)    
            vtk_grid.GetPointData().AddArray(vtk_density)
            # print grid.GetPointData().GetNumberOfArrays()
            if isinstance(dset_names,str):
                vtk_density.SetName("{}_{}".format(dset_names,n))
            elif isinstance(dset_names,(list,tuple)):
                vtk_density.SetName("{}_{}".format(dset_names[n],n))
            else:
                vtk_density.SetName("data_{}".format(n))
        return vtk_grid
        
    @staticmethod
    def generate_polar_writer( datasets, grid=default_grid, dset_names=dset_names_default):
        vtk_grid = vtkSaver.generate_polar_vtkStructuredGrid(datasets, grid=grid, dset_names=dset_names)
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(vtk_grid)
        return writer
    
    
    
    @staticmethod
    def generate_cartesian_writer(datasets, grid=default_grid, dset_names=dset_names_default):
        if not (isinstance(grid,np.ndarray) or isinstance(grid,NestedArray)):
            shape = datasets[0].shape
            axes = [np.arange(ax_len) for ax_len in shape]
            grid = GridFactory.construct_grid('uniform',axes).array
        g_shape  = grid[:].shape[:-1]
        n_dim = len(g_shape)
        if n_dim == 2:
            g_shape = g_shape + (1,)
        vtk_grid = vtk.vtkRectilinearGrid()
        vtk_grid.SetDimensions(g_shape)

        if n_dim == 2:
            vtk_grid.SetXCoordinates(vtn.numpy_to_vtk(grid[:,0,0],deep = True))
            vtk_grid.SetYCoordinates(vtn.numpy_to_vtk(grid[0,:,1],deep = True))
        if n_dim == 3:
            vtk_grid.SetXCoordinates(vtn.numpy_to_vtk(grid[:,0,0,0],deep = True))
            vtk_grid.SetYCoordinates(vtn.numpy_to_vtk(grid[0,:,0,1],deep = True))
            vtk_grid.SetZCoordinates(vtn.numpy_to_vtk(grid[0,0,:,2],deep = True))
            
        for n,data in enumerate(datasets):
            vtk_array = vtn.numpy_to_vtk(data.ravel('F'),deep = True ,array_type = vtk.VTK_FLOAT)
            vtk_grid.GetPointData().AddArray(vtk_array)
            if isinstance(dset_names,str):
                vtk_array.SetName("{}_{}".format(dset_names,n))
            elif isinstance(dset_names,(list,tuple)):
                vtk_array.SetName("{}_{}".format(dset_names[n],n))                        
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetInputData(vtk_grid)
        return writer        


    # writer_dict definition needs to be placed after the definitions of generate_(cartesian/spherical)_writer 
    writer_dict={'cartesian':generate_cartesian_writer.__func__,'spherical':generate_spherical_writer.__func__,'polar':generate_polar_writer.__func__}    

    
