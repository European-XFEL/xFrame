import time
import numpy as np
from scipy.interpolate import griddata
import logging
import traceback

log=logging.getLogger('root')

from xframe.library.physicsLibrary import ewald_sphere_theta
from xframe.library.gridLibrary import GridFactory
from xframe.library.mathLibrary import cartesian_to_spherical
from xframe.library.mathLibrary import spherical_to_cartesian
from xframe.library import units
from xframe.library.gridLibrary import get_linspace_log2
from xframe.library.gridLibrary import uniformGrid_func

class AgipdRegridder:
    def __init__(self,db,geometry,wavelength, file_name='regrid_data', save_regrid_data=False, generate_data=False):
        #framed_pixel_grid in spherical_coordinates
        self.pixel_grid_spher = geometry['framed_pixel_grid']
        self.pixel_grid_spher[...,0]*=units.standardLength
        self.lab_pixel_grid_cart = geometry['framed_lab_pixel_grid']
        self.pixel_mask = geometry['framed_mask']
        self.data_shape = geometry['data_shape']
        self.wavelength = wavelength
        self.file_name = file_name
        self.db=db
        if generate_data:
            self.regrid_data=self.generate_regrid_data()
            if save_regrid_data:
                db.save(file_name,self.regrid_data)
        else:
            try:
                self.regrid_data=db.load(file_name)
            except FileNotFoundError as e:
                log.warning('regrid_data file not found. Proceed by generating regrid_data.')
                self.regrid_data=self.generate_regrid_data()
                if save_regrid_data:
                    db.save(file_name,self.regrid_data)

    def approximate_pixel_size(self,grid_cart):
        #absolute minimal x-distance between consecutive agipd pixels along vertical(short direction 128 pixels per module) pixel row
        x_min = np.abs([np.diff(grid_cart[m][:,:,0],axis=1).min() for m in np.arange(len(grid_cart))]).min()
        #same but in y-direction
        y_min = np.abs([np.diff(grid_cart[m][:,:,1],axis=1).min() for m in np.arange(len(grid_cart))]).min()
        #same but in z-direction
        z_min = np.abs([np.diff(grid_cart[m][:,:,2],axis=1).min() for m in np.arange(len(grid_cart))]).min()        
        approximated_pixel_size= np.max([x_min,y_min,z_min])
        log.info('approximate_pixel_size = {}'.format(approximated_pixel_size))
        return approximated_pixel_size


    def generate_regrid_data(self):
        pixel_grid_cart = spherical_to_cartesian(self.pixel_grid_spher)
        data = self.generate_regrid_data_polar(self.pixel_grid_spher,grid_point_multiplicator = [3,1])    
        data2 = self.generate_regrid_data_cart(self.lab_pixel_grid_cart,grid_point_multiplicator = [4,4])
        lab_small = self.generate_regrid_data_cart(self.lab_pixel_grid_cart,grid_point_multiplicator = [2,2])
        q_cart_small = self.generate_regrid_data_cart(pixel_grid_cart,grid_point_multiplicator = [2,2])
        
        cart_data = {'lab':data2,'lab_small':lab_small,'q_small':q_cart_small}
        polar_data = {'reciprocal':data}
        regrid_data = {'cart':cart_data,'polar':polar_data}
        return regrid_data


    def generate_regrid_data_cart(self,pixel_grid,grid_point_multiplicator=[4,4],on_q_space=False):
        regrid_data={}
        d_shape = self.data_shape
        
        pix_size=self.approximate_pixel_size(pixel_grid)        
        
        old_global_x_bounds = (pixel_grid[...,0].min(),pixel_grid[...,0].max())
        old_global_y_bounds = (pixel_grid[...,1].min(),pixel_grid[...,1].max())

        try:
            assert (old_global_x_bounds[0]<=0) and (old_global_x_bounds[1]>0), 'global x bounds: ( {} ) do not contain 0.'.format(old_global_x_bounds)
            assert (old_global_y_bounds[0]<=0) and (old_global_y_bounds[1]>0), 'global y bounds: ( {} ) do not contain 0.'.format(old_global_y_bounds)
        except AssertionError as e:
            log.error(e)
            traceback.print_exc()
            raise e

        target_x_step = pix_size/grid_point_multiplicator[0]
        xs,x_param = get_linspace_log2(old_global_x_bounds,target_x_step,additional_steps = 1,return_parameters = True)
        x_step = x_param['step_size']
        global_x_bounds = x_param['domain']

        target_y_step = pix_size/grid_point_multiplicator[1]
        ys,y_param = get_linspace_log2(old_global_y_bounds,target_y_step,additional_steps = 1, return_parameters = True)
        y_step = y_param['step_size']
        global_y_bounds = y_param['domain']

        out_grid = GridFactory.construct_grid('uniform',[xs,ys]).array
        
        regrid_data['x_step'] = x_step
        regrid_data['xs'] = xs
        regrid_data['y_step'] = y_step
        regrid_data['ys'] = ys
        regrid_data['grid'] = out_grid

        global_x_min = global_x_bounds[0]
        global_y_min = global_y_bounds[0]
        log.info('out grid shape = {}'.format(out_grid.shape))
        for m,g in enumerate(pixel_grid):
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]
        
            x_bounds = [g[...,0].min(),g[...,0].max()]            
            x_min_index = max(0,int((x_bounds[0]-global_x_min)//x_step)+1)
            x_max_index = min(int((x_bounds[1]-global_x_min)//x_step)+1,out_grid.shape[0])        

            y_bounds = [g[...,1].min(),g[...,1].max()]
            #log.info('x_bounds = {}'.format(x_bounds))
            y_min_index = max(int((y_bounds[0]-global_y_min)//y_step)+1,0)
            y_max_index = min(int((y_bounds[1]-global_y_min)//y_step)+1,out_grid.shape[1])        
            m_grid = out_grid[x_min_index:x_max_index,y_min_index:y_max_index]
            log.info("x_range = {} y_range ={}".format([x_max_index,x_min_index],[y_max_index,y_min_index]))
            #log.info("x_range = {} y_range ={}".format([x_max_index-x_min_index],[y_max_index-y_min_index]))
            log.info('mgrid.shape = {}'.format(m_grid.shape))
            old_m_grid =g[...,:2]
            old_index = np.zeros(g.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
        
            index = griddata(old_m_grid.reshape(-1,2),old_index.flatten(),m_grid.reshape(-1,2),method='nearest',rescale = True).reshape(m_grid.shape[:-1])
            #log.info('index.shape = {}'.format(index.shape))
            #log.info('ranges y: {} x:{}'.format(y_max_index-y_min_index,x_max_index - x_min_index))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['x_range']=[x_min_index,x_max_index]
            m_dict['y_range']=[y_min_index,y_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data

    def generate_regrid_data_polar(self,pixel_grid,grid_point_multiplicator=[3,1],on_q_space=False):
        regrid_data={}
        d_shape = self.data_shape

        pixel_grid_c = spherical_to_cartesian(pixel_grid)
        pix_size=self.approximate_pixel_size(pixel_grid_c)
        
        old_global_r_bounds = (pixel_grid[...,0].min(),pixel_grid[...,0].max())
        old_global_phi_bounds = (pixel_grid[...,2].min(),pixel_grid[...,2].max())
        log.info('phi bounds = {}'.format(old_global_phi_bounds))

        target_r_step = pix_size/grid_point_multiplicator[0]
        r_length = old_global_r_bounds[1]-old_global_r_bounds[0]
        n_r_steps = int(r_length//target_r_step)
        r_step = r_length/(n_r_steps-1)
        rs = old_global_r_bounds[0]+ np.arange(n_r_steps)*r_step
        global_r_bounds = old_global_r_bounds
        log.info('r bounds = {} max,min = {}'.format(global_r_bounds,(rs.min(),rs.max())))


        # phi_step such that the arc length at max_x is proportional to the approximated pixel size    
        target_phi_step = pix_size/(global_r_bounds[1]*grid_point_multiplicator[1])
        phis,phi_param = get_linspace_log2(old_global_phi_bounds,target_phi_step,additional_steps = 1, return_parameters = True)
        phi_step = phi_param['step_size']
        global_phi_bounds = phi_param['domain']
        log.info('phi bounds = {} max,min = {}'.format(global_phi_bounds,(phis.min(),phis.max())))
                
        thetas = ewald_sphere_theta(self.wavelength,rs)
        tmp_grid = GridFactory.construct_grid('uniform',[rs,phis]).array
        out_grid = np.zeros(tmp_grid.shape[:-1]+(tmp_grid.shape[-1]+1,))
        out_grid[...,[0,2]] = tmp_grid
        out_grid[...,1] = thetas[:,None]
        
        
        regrid_data['r_step'] = r_step
        regrid_data['rs'] = rs
        regrid_data['phi_step'] = phi_step
        regrid_data['phis'] = phis
        regrid_data['grid'] = out_grid

        global_r_min = global_r_bounds[0]
        global_phi_min = global_phi_bounds[0]
        log.info('out grid shape = {}'.format(out_grid.shape))
        
        out_grid_c = spherical_to_cartesian(out_grid)[...,:2]
        pixel_grid_c = spherical_to_cartesian(pixel_grid)[...,:2]
        for m,g in enumerate(pixel_grid):            
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]

            g_c = pixel_grid_c[m]
        
            r_bounds = [g[...,0].min(),g[...,0].max()]            
            r_min_index = max(int((r_bounds[0]-global_r_min)//r_step)+1,0)
            r_max_index = min(int((r_bounds[1]-global_r_min)//r_step)+1,out_grid.shape[0])        

            phi_bounds = [g[...,2].min(),g[...,2].max()]
            #log.info('global phi_min = {}'.format(global_phi_min))
            #log.info('phi_bounds = {}'.format(phi_bounds))
            phi_min_index = max(int((phi_bounds[0]-global_phi_min)//phi_step)+1,0)
            phi_max_index = min(int((phi_bounds[1]-global_phi_min)//phi_step)+1,out_grid.shape[1])     
            out_g_c = out_grid_c[r_min_index:r_max_index,phi_min_index:phi_max_index]
            #log.info('min - glob =  {}'.format((phi_bounds[0]-global_phi_min)//phi_step+1))
            #log.info('max - glob =  {}'.format((phi_bounds[1]-global_phi_min)//phi_step+1))
            #log.info('out_grid shape = {}'.format(out_grid.shape))
            #log.info("r_range = {} phi_range ={}".format([r_max_index,r_min_index],[phi_max_index,phi_min_index]))
            #log.info("x_range = {} y_range ={}".format([x_max_index-x_min_index],[y_max_index-y_min_index]))
            log.info('mgrid.shape = {}'.format(out_g_c.shape))
            old_index = np.zeros(g.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
        
            index = griddata(g_c.reshape(-1,2),old_index.flatten(),out_g_c.reshape(-1,2),method='nearest',rescale = True).reshape(out_g_c.shape[:-1])
            #log.info('index nonzero: {}'.format((index!=0).any()))
            #log.info('index.shape = {}'.format(index.shape))
            #log.info('ranges y: {} x:{}'.format(y_max_index-y_min_index,x_max_index - x_min_index))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['r_range']=[r_min_index,r_max_index]
            m_dict['phi_range']=[phi_min_index,phi_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data    
    
    def generate_regrid_data_polar_old(self,pixel_grid,grid_point_multiplicator=[2,1]):
        regrid_data={}
        d_shape = self.data_shape

        grid_c = spherical_to_cartesian(pixel_grid)
        pix_size=self.approximate_pixel_size(grid_c)
        
        r_max = pixel_grid[...,0].max()
        # r_step such that it is proportional to the approximated pixel size
        target_r_step = (pix_size/grid_point_multiplicator[0])
        n_new_rs = (r_max//target_r_step)
        r_step = r_max/n_new_rs
        rs= np.arange(n_new_rs)*r_step

        thetas = ewald_sphere_theta(self.wavelength,rs)

        # phi_step such that the arc length at max_x is proportional to the approximated pixel size
        
        target_phi_step = pix_size/(r_max*grid_point_multiplicator[1])
        phis, phi_param = get_linspace_log2([0,2*np.pi -target_phi_step],target_phi_step,additional_steps = 1,return_parameters = True)
        phi_step = phi_param['step_size']

        tmp_grid = GridFactory.construct_grid('uniform',[rs,phis]).array
        out_grid = np.zeros(tmp_grid.shape[:-1]+(tmp_grid.shape[-1]+1,))
        out_grid[...,[0,2]] = tmp_grid
        out_grid[...,1] = thetas[:,None]
        
        regrid_data['r_step'] = r_step
        regrid_data['rs'] = rs
        regrid_data['phi_step'] = phi_step
        regrid_data['phis'] = phis
        regrid_data['grid'] = out_grid
        log.info('out grid shape = {}'.format(out_grid.shape))
        log.info('out grid shape = {}'.format(out_grid.shape))
        grid_c = spherical_to_cartesian(out_grid)[...,:2]
        pixel_grid_c = spherical_to_cartesian(pixel_grid)[...,:2]
        
        for m,g in enumerate(pixel_grid):
            log.info('generating regrid_data for module {}/{}'.format(m,len(pixel_grid)))
            #log.info('g shape = {}'.format(g.shape))
            g_c=pixel_grid_c[m]
            #log.info('g_c shape = {}'.format(g_c.shape))
            regrid_data[str(m)]={}
            m_dict=regrid_data[str(m)]
        
            r_bounds = [g[...,0].min(),g[...,0].max()]
            r_min_index = int(r_bounds[0]//r_step)
            r_max_index = int(r_bounds[1]//r_step)        
            
            phi_bounds = [g[...,2].min(),g[...,2].max()]
            #log.info('phi_bounds = {}'.format(phi_bounds))
            phi_min_index = int((np.pi+phi_bounds[0])//phi_step)
            phi_max_index = int((np.pi+phi_bounds[1])//phi_step)        
            m_grid = grid_c[r_min_index:r_max_index,phi_min_index:phi_max_index]
            log.info("r_range = {} phi_range ={}".format([r_min_index,r_max_index],[phi_min_index,phi_max_index]))
            #log.info('mgrid.shape = {}'.format(m_grid.shape))
            old_m_grid =g_c
            old_index = np.zeros(g_c.shape[:-1])
            old_index[:-1,:-1][self.pixel_mask[m]] = np.arange(1,d_shape[1]*d_shape[2]+1)
            log.info('old index nonzero = {}'.format((old_index !=0).any()))
            #log.info('old m grid  = {}'.format(old_m_grid.shape))
            
            index = griddata(old_m_grid.reshape(-1,2),old_index.flatten(),m_grid.reshape(-1,2),method='nearest',rescale = True).reshape(m_grid.shape[:-1])
            log.info('index nonzero = {}'.format((index!=0).any()))
            #log.info('ranges r: {} phi:{}'.format(r_max_index-r_min_index,phi_max_index - phi_min_index))
            #log.info('index.shape = {}'.format(index.shape))
            #new_data = griddata(old_grid,old_data.flatten(),new_grid,method='linear')

            m_dict['index']=index
            m_dict['r_range']=[r_min_index,r_max_index]
            m_dict['phi_range']=[phi_min_index,phi_max_index]
            valid_index_mask = (index!=0)
            m_dict['valid_data_mask'] = valid_index_mask
            m_dict['valid_index'] = (index[valid_index_mask]-1).astype(int)            
        return regrid_data
    
    def generate_and_save_regrid_data(self):
        self.regrid_data=self.generate_regrid_data()
        self.db.save(self.file_name,self.regrid_data)

        

    def put_data_into_grid(self,data,out_grid,r_range,phi_range):
        new_data=np.zeros((data.shape[0],)+out_grid.shape[:-1])
        #log.info('new data shape = {}'.format(new_data.shape))
        #log.info("r_range = {} phi_range ={}".format(r_range,phi_range))
        #log.info("len r = {} len phi ={} data shape = {}".format(np.diff(r_range),np.diff(phi_range),data.shape))
        new_data[:,r_range[0]:r_range[1],phi_range[0]:phi_range[1]] = data
        return new_data
        
    def regrid_polar(self,data, combine_modules=False, modules_on_full_grid=False,grid_type='reciprocal'):
        regrid_polar_dict = self.regrid_data['polar']
        for known_grid_type in regrid_polar_dict:
            if grid_type == known_grid_type:
                regrid_data = regrid_polar_dict[known_grid_type]
                break
            
        out_grid = regrid_data['grid']
        out_grid_max_index = regrid_data['grid']
        
        out_data=[]
        for m in np.arange(len(data)):
            #log.info("m = {} of {}".format(m,len(data)))
            valid_data_mask=regrid_data[str(m)]['valid_data_mask'].astype(bool)
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            valid_index=regrid_data[str(m)]['valid_index']
            #log.info('valid_index = {}'.format(valid_index.shape))

            if len(data.shape) == len(self.data_shape):
                data = data[None,...]
            regridded_m_data=np.zeros((data.shape[0],)+valid_data_mask.shape)
            regridded_m_data[:,valid_data_mask]=data[:,m].reshape(data.shape[0],-1)[:,valid_index]
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            if (combine_modules or modules_on_full_grid):
                r_range = regrid_data[str(m)]['r_range']
                phi_range = regrid_data[str(m)]['phi_range']
                regridded_m_data = self.put_data_into_grid(regridded_m_data,out_grid,r_range,phi_range)[:,:-1,:-1]
            out_data.append(regridded_m_data)            
            
        if combine_modules:
            out_data = np.sum(out_data,axis=0)
        if modules_on_full_grid:
            out_data = np.array(out_data)
        if isinstance(out_data,np.ndarray):
            out_data = np.squeeze(out_data)
            
        return out_data
    
    def regrid_cart(self,data, combine_modules=False, modules_on_full_grid=False,grid_type='lab'):
        regrid_cart_dict = self.regrid_data['cart']
        #log.info(regrid_cart_dict.keys())
        for known_grid_type in regrid_cart_dict:
            if grid_type == known_grid_type:
                regrid_data = regrid_cart_dict[known_grid_type]
                break
        
        out_grid = regrid_data['grid']
        out_data=[]
        for m in np.arange(len(data)):
            #log.info("m = {} of {}".format(m,len(data)))
            valid_data_mask=regrid_data[str(m)]['valid_data_mask'].astype(bool)
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            valid_index=regrid_data[str(m)]['valid_index']
            #log.info('valid_index = {}'.format(valid_index.shape))
            #log.info('valid_data_mask shape = {}'.format(valid_data_mask.shape))
            #log.info('regridded_m_data shape = {}'.format(regridded_m_data[~valid_data_mask].shape))
            if len(data.shape) == len(self.data_shape):
                data = data[None,...]
            regridded_m_data=np.zeros((data.shape[0],)+valid_data_mask.shape)
            regridded_m_data[:,valid_data_mask]=data[:,m].reshape(data.shape[0],-1)[:,valid_index]
            #log.info("regridded shape = {}".format(regridded_m_data.shape))
            
            if (combine_modules or modules_on_full_grid):
                r_range = regrid_data[str(m)]['x_range']
                phi_range = regrid_data[str(m)]['y_range']
                #log.info("x_range: {} y_range: {}".format(r_range,phi_range))
                regridded_m_data = self.put_data_into_grid(regridded_m_data,out_grid,r_range,phi_range)[:,:-1,:-1]                
                
            out_data.append(regridded_m_data)            
            
        if combine_modules:
            out_data = np.sum(out_data,axis=0)
        if modules_on_full_grid:
            out_data = np.array(out_data)
        if isinstance(out_data,np.ndarray):
            out_data = np.squeeze(out_data)
            
        return out_data
