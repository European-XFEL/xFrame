#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
#import matplotlib.pyplot as plt
import sys
import numpy as np
import logging
log=logging.getLogger('root')
from xframe.presenters.interfaces import MatplotlibDependency,MPLToolkitDependency


matplotlib = MatplotlibDependency
mpl_toolkits= MPLToolkitDependency

class pltWrapper:
    def __getattr__(self,key):
      return globals()['matplotlib'].pyplot.__getattribute__(key)
  
class locatorWrapper:
    def __getattr__(self,key):
      return globals()['mpl_toolkits'].axes_grid1.inset_locator.__getattribute__(key)
plt = pltWrapper()
inset_locator = locatorWrapper()

def depencency_injection_hook_mpl():
    matplotlib.rcParams['text.usetex'] = True
    module = sys.modules[__name__]
    module.plt = matplotlib.pyplot

def dependency_injection_hook_mpl_toolkits():
    module = sys.modules[__name__]
    module.inset_locator = mpl_toolkits.axes_grid1.inset_locator


from xframe.library import mathLibrary as mLib
from xframe.library.gridLibrary import getGridByXYValues
from xframe.library.gridLibrary import Grid
from xframe.library.pythonLibrary import flattenList
from xframe.library.pythonLibrary import xprint
from xframe.library.pythonLibrary import make_string_tex_conform

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    
def apply_layout_to_ax(ax,layout):
    title=layout.get('title',"Scatter plot 2D")
    xLable=layout.get('x_label',"r")
    yLable=layout.get('y_label',"phi")
    title_size = layout.get('title_size',16)
    text_size = layout.get('text_size',14)
    ax.set_title(title,fontsize=title_size)
    ax.set_xlabel(xLable,fontsize=text_size)
    ax.set_ylabel(yLable,fontsize=text_size)
    #ax.legend(ncol=3, loc='best')

def apply_layout_to_figure(fig,layout):
    caption_size = layout.get('caption_size',10)
    caption = layout.get('caption',False)
    if isinstance(caption,str):
        #log.info('caption is added')
        caption = make_string_tex_conform(caption)
        #log.info('caption = {}'.format(caption))
        fig.text(0.1,0.1,caption,fontsize = caption_size,va = 'top',wrap = True)


class Plotter:
    @classmethod
    def get_fig(cls,*args,**kwargs):
        pass
    @classmethod
    def show(cls,*args,**kwargs):
        fig = cls.get_fig(*args,**kwargs)
        fig.show()
class scatter2D:
    def applyLayoutToAx(self,ax,layout):
        title=layout.get('title',"Scatter plot 2D")
        xLable=layout.get('xLable',"x")
        yLable=layout.get('yLable',"y")
        
        ax.set_title(title,fontsize=16)
        ax.set_xlabel(xLable,fontsize=14)
        ax.set_ylabel(yLable,fontsize=14)
    def present2(self):
        pass
    def present(self,DataSets,mode='points',layout={}):
        colors = [[0,0,0]]
        pointArea=1
        for Data in DataSets:
            fig,ax=plt.subplots()
            if mode=='xy':
                x=Data[0]
                y=Data[1]
            else:
                x=Data[:,0]
                y=Data[:,1]
            ax.scatter(x, y,s=pointArea, c=colors, alpha=0.5)

            self.applyLayoutToAx(ax,layout)
#            ax.set_aspect('equal')
        plt.show()


class errorbar2D:
    def applyLayoutToAx(self,ax,layout):
        title=layout.get('title',"Scatter plot 2D")
        xLable=layout.get('xLable',"x")
        yLable=layout.get('yLable',"y")
        
        ax.set_title(title,fontsize=16)
        ax.set_xlabel(xLable,fontsize=14)
        ax.set_ylabel(yLable,fontsize=14)
    def present2(self):
        pass
    def present(self,Data,xerr=None,yerr=None,mode='points',layout={}):
        fig,ax=plt.subplots()
        if mode=='xy':
            x=Data[0]
            y=Data[1]
        else:
            x=Data[:,0]
            y=Data[:,1]
        ax.errorbar(x, y,xerr=xerr,yerr=yerr)

        self.applyLayoutToAx(ax,layout)
#            ax.set_aspect('equal')
        plt.show()

        
class scatterPolar2D:
    def present2(self):
        pass
    def present(self,Data):
        theta=Data[...,1]
        r=Data[...,0]
        colors = [[0,0,0]]
        pointArea=1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.grid(False)
        ax.set_yticklabels([])
        c = ax.scatter(theta, r, c=colors, s=pointArea, alpha=0.5)

        plt.show()
        
class scatter3D:
    def present2(self):
        pass
    def present(self,Data):
        x=Data[:,0]
        y=Data[:,1]
        z=Data[:,2]
        colors = [[0,0,0]]
        pointArea=1
        fig,ax=plt.subplots()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y,z,s=pointArea, c=colors, alpha=0.5)
        ax.set_title('Scatter plot 3D')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

class scatterSpherical3D:
    def applyLayoutToAx(self,ax,layout):
        title=layout.get('title',"Scatter plot 3D")
        xLable=layout.get('xLable',"x")
        yLable=layout.get('yLable',"y")
        zLable=layout.get('zLable',"z")
        
        ax.set_title(title,fontsize=16)
        ax.set_xlabel(xLable,fontsize=14)
        ax.set_ylabel(yLable,fontsize=14)
        ax.set_zlabel(zLable,fontsize=14)
        
    def present2(self):
        pass
    def present(self,Data,layout={}):

        cartesianData=mLib.sphericalToCartesian(Data)
        x=cartesianData[:,0]
        y=cartesianData[:,1]
        z=cartesianData[:,2]

        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = layout.get('colors',[[0,0,0]])
        pointArea=layout.get('pointArea',1)
        alpha=layout.get('alpha',.5)
        ax.scatter(x,y,z,s=pointArea, c=colors, alpha=alpha)

        self.applyLayoutToAx(ax,layout)
        
        
        set_axes_equal(ax)
        plt.show()

class scatterCylindrical3D:
    def applyLayoutToAx(self,ax,layout):
        title=layout.get('title',"Scatter plot 3D")
        xLable=layout.get('xLable',"x")
        yLable=layout.get('yLable',"y")
        zLable=layout.get('zLable',"z")
        
        ax.set_title(title,fontsize=16)
        ax.set_xlabel(xLable,fontsize=14)
        ax.set_ylabel(yLable,fontsize=14)
        ax.set_zlabel(zLable,fontsize=14)
        
    def present2(self):
        pass
    def present(self,Data,layout={}):

        cartesianData=mLib.cylindricalToCartesian(Data)
        x=cartesianData[:,0]
        y=cartesianData[:,1]
        z=cartesianData[:,2]

        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = layout.get('colors',[[0,0,0]])
        pointArea=layout.get('pointArea',1)
        alpha=layout.get('alpha',.5)
        ax.scatter(x,y,z,s=pointArea, c=colors, alpha=alpha)

        self.applyLayoutToAx(ax,layout)
        
        
        set_axes_equal(ax)
        plt.show()

class hist2D():
    def present(self,data):
        x=data[:,0]
        y=data[:,1]
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist2d(x,y)
        plt.show()
        
class pcolor2D:
    def present2(self):
        pass
    def present(self,grid,data):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_title('Color Mesh')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        x=grid[:,0]
        y=grid[:,1]
        z=data       
        
        c = ax.pcolormesh(x, y, z, cmap='RdBu')
        fig.colorbar(c, ax=ax)

        plt.show()

class heatPolar2D:
    @classmethod   
    def get_fig(cls,data,grid=False,layout={},scale='lin'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        if not isinstance(grid,bool):
            grid=grid.copy()
            #        log.info('grid shape = {}'.format(grid.shape))
#            grid=np.concatenate((grid[:],np.expand_dims(grid[:,0],1)),axis=1)
            r=grid[...,0]
            phi=grid[...,1]
            #            log.info('phi shape={}'.format(phi.shape))
        else:
            shape=data.shape
            rValues=np.arange(0,shape[0])
            phiValues=np.zeros(shape[1]+1)
            phiValues[:-1]=np.arange(shape[1])*2*np.pi/(shape[1])
            
            r,phi=np.meshgrid(rValues,phiValues)

            if type(data)==Grid:
                data=data.array

                #        plt.pcolormesh(th, r, z)
        cmap=plt.get_cmap('inferno')
        if scale=='log':
            # data expected to come as r,phi but matplotlib wants phi,r
            #cmap.set_bad(cmap(0))
            cmap.set_bad(color='w')
            heatmap=ax.pcolormesh(phi,r,np.swapaxes(data,0,1),norm=matplotlib.colors.LogNorm(),cmap=cmap)
        else:
            heatmap=ax.pcolormesh(phi,r,np.swapaxes(data,0,1),cmap=cmap)
        fig.colorbar(heatmap,extend='both')
        apply_layout_to_ax(ax,layout)
        return fig
    
    @classmethod
    def show(cls,data,grid=False,layout={},scale='lin'):
        fig = cls.get_fig(data,grid=grid,layout=layout,scale=scale)
        fig.show()


        
class heat2D:
    @classmethod
    def get_fig(cls,data,grid=False,layout={},scale='lin',vmin = False,vmax =False,cmap='inferno'):   
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #            cmap=plt.get_cmap('viridis')
        cmap=plt.get_cmap(cmap).copy()

        if not isinstance(grid,bool):
            grid=grid.copy()
            #        log.info('grid shape = {}'.format(grid.shape))
            x=grid[...,0]
            y=grid[...,1]
            #            log.info('phi shape={}'.format(phi.shape))
        else:
            shape=data.shape
            xValues=np.arange(0,shape[0])
            yValues=np.arange(0,shape[1])
            
            x,y=np.meshgrid(xValues,yValues)
            x=np.swapaxes(x,0,1)
            y=np.swapaxes(y,0,1)
            #log.debug('r shape = {} phi shape={}'.format(x.shape,y.shape))
            #        plt.pcolormesh(th, r, z)
        if scale=='log':
            cmap.set_bad(cmap(0))
            if isinstance(vmin,bool) and (not isinstance(vmax,bool)):
                vmin = data.min()
            elif isinstance(vmax,bool) and (not isinstance(vmin,bool)):
                vmax = data.min()
            elif isinstance(vmax,bool) and isinstance(vmin,bool):
                median_value = np.median(data)
                max_value = data.max()
                min_value = data.min()
                if (median_value>0) and (median_value!=np.nan) and (max_value<np.inf):
                    orders = int(np.log10(max_value/median_value)*0.8)
                    vmin,vmax = [median_value*10**(-orders//2),median_value*10**orders]
                elif (max_value==np.inf) and (median_value>0) and (median_value!=np.nan):            
                    orders = 12
                    vmin,vmax = [median_value*10**(-orders//2),median_value*10**orders]
                else:            
                    orders = 12
                    vmin,vmax = [max_value*10**(-orders//2),max_value]
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
            heatmap=ax.pcolormesh(x,y,data,norm = norm,cmap=cmap, shading ='auto')
        else:
            heatmap=ax.pcolormesh(x,y,data,cmap=cmap,vmin = vmin,vmax =vmax,shading='auto')
        fig.colorbar(heatmap)
        
        apply_layout_to_ax(ax,layout)
        return fig

    @classmethod
    def show(cls,data,grid=False,layout={},scale='lin',vmin = False,vmax =False):
        fig = cls.get_fig(data,grid=grid,layout=layout,scale=scale,vmin = vmin,vmax =vmax)
        fig.show()

class heat2D_multi(Plotter):
    @classmethod
    def get_fig(cls,datasets,grid=False,layout={},scale='lin',shape=(1,1),size = False,vmin = None,vmax =None,cmap = 'inferno'):
        cmap=plt.get_cmap(cmap).copy()
        if isinstance(size,bool):
            size = tuple(9*i for i in shape[::-1])
        #cmap=plt.get_cmap('viridis')        
        fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1],figsize=size, gridspec_kw = {'wspace':0.2, 'hspace':0.45})
        if shape[0]==1:
            axes = [axes]
            datasets = [datasets]
        if shape[1]==1:
            axes = [[ax] for ax in axes]
            datasets = [[data] for data in datasets]

        if not isinstance(grid,bool):
            grid=grid.copy()
            #        log.info('grid shape = {}'.format(grid.shape))
            x=grid[...,0]
            y=grid[...,1]
            #            log.info('phi shape={}'.format(phi.shape))
        else:
            shape=flattenList(datasets)[0].shape
            xValues=np.arange(0,shape[0])
            yValues=np.arange(0,shape[1])
            
            x,y=np.meshgrid(xValues,yValues)
            x=np.swapaxes(x,0,1)
            y=np.swapaxes(y,0,1)
            #log.debug('r shape = {} phi shape={}'.format(x.shape,y.shape))
            #        plt.pcolormesh(th, r, z)
        heatmaps = []
        if scale=='log':
            cmap.set_bad(cmap(0))
            for ax_row,dataset_row in zip(axes,datasets):
                heatmap_row = []
                for ax,data in zip(ax_row,dataset_row):
                    #if isinstance(vmin,bool):
                    #    vmin = data.min()
                    #if isinstance(vmax,bool):
                    #    vmax = data.min()
                    heatmap_row.append(ax.pcolormesh(x,y,data,norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap,shading='auto'))
                    ax.set_aspect('equal')
                heatmaps.append(heatmap_row)
        else:
            for ax_row,dataset_row in zip(axes,datasets):
                heatmap_row = []
                for ax,data in zip(ax_row,dataset_row):
                    heatmap_row.append(ax.pcolormesh(x,y,data,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto'))
                heatmaps.append(heatmap_row)
        #fig.colorbar(heatmap)
        for ax_row,heatmap_row in zip(axes,heatmaps):
            for ax,heatmap in zip(ax_row,heatmap_row):
                    fig.colorbar(heatmap , ax = ax)

        if isinstance(layout,(tuple,list)):
            for ax_row,dataset_row,layout_row in zip(axes,datasets,layout):
                for ax,data,l in zip(ax_row,dataset_row,layout_row):
                    apply_layout_to_ax(ax,l)
        else:
            for ax_row,dataset_row in zip(axes,datasets):
                for ax,data in zip(ax_row,dataset_row):
                    apply_layout_to_ax(ax,layout)
        return fig



class imshow:
    @classmethod
    def get_fig(cls,data,layout={},scale='lin',vmin = False,vmax =False,cmap = 'inferno',aspect = 1):     
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap=plt.get_cmap(cmap).copy()
        if isinstance(vmin,bool):
            vmin = data.min()
        if isinstance(vmax,bool):
            vmax = data.min()
        if scale=='log':
            cmap.set_bad(cmap(0))
            heatmap=ax.imshow(data,norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap,aspect = aspect)
        else:
            heatmap=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,aspect = aspect)
        fig.colorbar(heatmap)
        
        apply_layout_to_ax(ax,layout)
        return fig

    @classmethod
    def show(cls,data,layout={},scale='lin',vmin = False,vmax =False,cmap = 'inferno',aspect = 1):     
        fig = cls.get_fig(data,grid=grid,layout=layout,scale=scale,vmin = vmin,vmax =vmax,aspect = aspect,cmap=cmap)
        fig.show()


class agipd_heat_multi():
    @classmethod
    def get_fig(cls,densities,grid,print_mask,shape = (1,1),layouts=[{'x_label':r'$q_x \quad [\AA^-1] $','y_label':r'$q_y \quad [\AA^-1]$'}],scale='log',vmin=None,vmax=None):
        # assumes grid and density is ordered by modules
        cmap=plt.get_cmap('jet')
        fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1],figsize=(9, 9), gridspec_kw = {'wspace':0.2, 'hspace':0.45})

        n_densities = len(densities)
        for ax in axes.ravel()[n_densities:]:
            ax.axis('off')
        ax_tuple = tuple(axes.flat)
        g_grid=grid[:,:-1,:-1][print_mask].reshape(-1,512,128,3)

        if scale=='log':
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        else:
            morm = None
        base_shape = 100*shape[0]+10*shape[1]
        for id,density in enumerate(densities):
            ax = ax_tuple[id]
            print_data=np.zeros(print_mask.shape,dtype=float)
            print_data[print_mask]=density.flatten()
            for m_id in range(len(grid)):
                x = grid[m_id,:,:,0]
                y = grid[m_id,:,:,1]
                c = np.swapaxes(print_data[m_id],0,0)
                #heatmap = ax.pcolormesh(x, y, c, norm=norm ,cmap = cmap,vmin=density.min()+1e-10,vmax=density.max())
                heatmap = ax.pcolormesh(x, y, c, norm=norm ,cmap = cmap)
                ax.set_aspect(1)
            r = np.abs(x.max()-x.min())*0.01
            circ = plt.Circle((0,0), radius=r, linewidth=2, color='r')
            circ.set_fill(True)
            ax.add_patch(circ)
            
            apply_layout_to_ax(ax,layouts[id])
        #add color bar below chart
        #divider = make_axes_locatable(plt.gca())
        #cax = divider.append_axes("right", "5%", pad="3%")
        cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
        bar = fig.colorbar(ax.collections[0],ax = cax,fraction = 0.3 ,aspect = 30)
        cax.axis('off')
        fig.tight_layout()
        return fig

    @classmethod
    def show(cls,densities,grid,print_mask,shape = (1,1),layouts=[{'x_label':r'$q_x \quad [\AA^-1] $','y_label':r'$q_y \quad [\AA^-1]$'}],scale='log',vmin=None,vmax=None):
        fig = cls.get_fig(densities,grid,print_mask,shape = shape, layouts = layouts, scale = scale,vmin=vmin,vmax=vmax)
        fig.show()
                
class agipd_heat():
    @classmethod
    def get_fig(cls,density, grid,print_mask,fill_value = 0,gradients=False,layout={'x_label':r'$q_x \quad [\AA^-1] $','y_label':r'$q_y \quad [\AA^-1]$'},scale='log',vmin=None,vmax=None,cmap = 'viridis'):
        # assumes grid and density is ordered by modules
        cmap=plt.get_cmap(cmap).copy()
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
               
        print_data=np.full(print_mask.shape,fill_value,dtype=float)#np.zeros(print_mask.shape)
        print_data[print_mask]=density.flatten()
        g_grid=grid[:,:-1,:-1][print_mask].reshape(-1,512,128,3)

        if scale=='log':
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        else:
            norm = None
        log.info('gradients = {}'.format(gradients))
        
        for m_id in range(len(grid)):
            x = grid[m_id,:,:,0]
            y = grid[m_id,:,:,1]
            c = np.swapaxes(print_data[m_id],0,0)
            if (scale=='log') and (c.max()==0):
                continue
            #heatmap = ax.pcolormesh(x, y, c, norm=norm ,cmap = cmap,vmin=density[density>0].min()*0.1,vmax=density.max())
            heatmap = ax.pcolormesh(x, y, c, norm=norm ,cmap = cmap)
            if not isinstance(gradients,bool):
                log.info('fu8uuuu')
                X = g_grid[m_id,...,0]                
                Y = g_grid[m_id,...,1]
                U = gradients[m_id,...,0]
                V = gradients[m_id,...,1]
                q = ax.quiver(X, Y, U, V, scale=np.abs(gradients).max()*0.9, scale_units='inches')

            ax.set_aspect(1)
            r = np.abs(x.max()-x.min())*0.01
            circ = plt.Circle((0,0), radius=r, linewidth=2, color='r')
            circ.set_fill(True)
            ax.add_patch(circ)
        apply_layout_to_ax(ax,layout)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bar = fig.colorbar(ax.collections[0],cax =cax)
        apply_layout_to_figure(fig,layout)
        return fig

    @classmethod
    def show(cls,density, grid,print_mask,fill_value = 0,gradients=False,layout={'x_label':r'$q_x \quad [\AA^-1]$','y_label':r'$q_y \quad [\AA^-1]$'},scale='log',vmin=None,vmax=None,cmap = 'inferno'):
        fig = cls.get_fig(density,grid,print_mask,fill_value=fill_value,gradients = gradients, layout = layout, scale = scale,vmin=vmin,vmax=vmax,cmap = cmap)
        fig.show()

class centering_heat:
    @classmethod
    def get_fig(cls,convolution,sigma_mask,center,errors,grid, layout_conv, layout_mask,scale='lin'):
     
        fig = plt.figure()
        ax_conv = fig.add_axes((0.05, 0.25, 0.4, 0.65))
        ax_mask = fig.add_axes((0.55, 0.25, 0.4, 0.65))
        #            cmap=plt.get_cmap('viridis')
        cmap=plt.get_cmap('inferno')

        if not isinstance(grid,bool):
            grid=grid.copy()
            #        log.info('grid shape = {}'.format(grid.shape))
            x=grid[...,0]
            y=grid[...,1]
            #            log.info('phi shape={}'.format(phi.shape))
        if scale=='log':
            cmap.set_bad(cmap(0))
            hm_conv=ax_conv.pcolormesh(x,y,convolution,norm=matplotlib.colors.LogNorm(),cmap=cmap)
            hm_mask=ax_mask.pcolormesh(x,y,sigma_mask,norm=matplotlib.colors.LogNorm(),cmap=cmap)
        else:
            hm_conv=ax_conv.pcolormesh(x,y,convolution,cmap=cmap)
            hm_mask=ax_mask.pcolormesh(x,y,sigma_mask,cmap=cmap)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        text = 'Center at: {}'.format(center)+'\n'+r'$e^{-\frac{1}{2}}$'+' errors: x {} Y {}'.format(errors[0],errors[1])      
        ax_mask.text(0.05, 0.05, text,horizontalalignment='left', verticalalignment='bottom',bbox=props, transform=ax_mask.transAxes)

            
        fig.suptitle(layout_conv["sup_title"],fontsize = 24)
        apply_layout_to_ax(ax_conv,layout_conv)
        apply_layout_to_ax(ax_mask,layout_mask)
        divider = make_axes_locatable(ax_conv)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bar = fig.colorbar(ax_conv.collections[0],cax =cax)
        apply_layout_to_figure(fig,layout_conv)
        return fig

    @classmethod
    def show(cls,convolution,sigma_mask,center,errors,grid, layout_conv, layout_mask,scale='lin'):
        fig = cls.get_fig(convolution,sigma_mask,center,errors,grid, layout_conv, layout_mask,scale=scale)
        fig.show()
            

class plot1D:
    @classmethod
    def get_fig(cls,data_set,labels=False, grid = False,layout={},y_scale='lin',x_scale='lin',ylim=False,xlim=False,xticks=False,yticks=False):
        if isinstance(grid,bool):
            grid=np.arange(data_set.shape[-1])
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        if not isinstance(ylim,bool):
            ax.set_ylim(ylim)
        if not isinstance(xlim,bool):
            ax.set_xlim(xlim)
        if not isinstance(yticks,bool):
            ax.set_yticks(yticks)
        if not isinstance(xticks,bool):
            ax.set_xticks(xticks)
        if data_set.ndim>1:
            if not isinstance(labels,(list,tuple)):
                labels = [None]*len(data_set)
            colors = plt.get_cmap('viridis')(np.linspace(0,1,len(data_set)))
            for part,label,color in zip(data_set,labels,colors):
                #print(part.shape)
                #log.info('shapes grid = {} part = {} '.format(grid.shape,part.shape))
                #print('shapes grid = {} part = {} '.format(grid.shape,part.shape))
                ax.plot(grid,part,label = label,color = color)
        else:
            ax.plot(grid,data_set,label=labels[0])
        ax.grid()
        if isinstance(labels,(tuple,list,np.ndarray)):
            if len(labels)>=6:
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2-0.05*len(labels)/6), ncol=6)
            else:
                ax.legend()
        else:
            ax.legend()
        if x_scale=='log':
            ax.set_xscale('log')
        if y_scale=='log':
            ax.set_yscale('log')
        apply_layout_to_ax(ax,layout)
        apply_layout_to_figure(fig,layout)
        return fig
    @classmethod
    def show(cls,data_set,labels=False,grid=False,layout={},x_scale='lin',y_scale='lin',ylim=False,xlim=False,xticks=False,yticks=False):
        fig = cls.get_fig(data_set,labels = labels, grid = grid, layout = layout, x_scale = x_scale, y_scale=y_scale,ylim=ylim,xlim=xlim,xticks=xticks,yticks=yticks)
        fig.show()

        
class scatter1D:
    @classmethod
    def get_fig(cls,data_set,s=None, c=None, marker=None,labels=False, grid = False,layout={},y_scale='lin',x_scale='lin',ylim=False,xlim=False,xticks=False,yticks=False):
        if isinstance(grid,bool):
            grid=np.arange(data_set.shape[-1])
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        if not isinstance(ylim,bool):
            ax.set_ylim(ylim)
        if not isinstance(xlim,bool):
            ax.set_xlim(xlim)
        if not isinstance(yticks,bool):
            ax.set_yticks(yticks)
        if not isinstance(xticks,bool):
            ax.set_xticks(xticks)
        if data_set.ndim>1:
            if not isinstance(labels,(list,tuple)):
                labels = [None]*len(data_set)
            colors = plt.get_cmap('viridis')(np.linspace(0,1,len(data_set)))
            for part,label,color in zip(data_set,labels,colors):
                print(part.shape)
                log.info('shapes grid = {} part = {} '.format(grid.shape,part.shape))
                #print('shapes grid = {} part = {} '.format(grid.shape,part.shape))
                ax.scatter(grid,part,s=s, c=c, marker=marker,label = label,color = color)
        else:
            ax.scatter(grid,data_set,s=s, c=c, marker=marker)
        ax.grid()
        #ax.legend(fontsize=26)
        if x_scale=='log':
            ax.set_xscale('log')
        if y_scale=='log':
            ax.set_yscale('log')
        apply_layout_to_ax(ax,layout)
        apply_layout_to_figure(fig,layout)
        return fig
    @classmethod
    def show(cls,data_set,labels=False,grid=False,layout={},x_scale='lin',y_scale='lin'):
        fig = cls.get_fig(data_set,labels = labels, grid = grid, layout = layout, x_scale = x_scale, y_scale=y_scale)
        fig.show()
                    
class bar1D:
    @classmethod
    def get_fig(cls,data_set,labels=False, grid = False,layout={},y_scale='lin',x_scale='lin',y_min = False):
        if isinstance(grid,bool):
            grid=np.arange(data_set.shape[0])
        log.info('grid type = {}'.format(type(grid)))
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        max_value = 0
        if data_set.ndim>1:
            if not isinstance(labels,(list,tuple)):
                labels = [None]*len(data_set)
            for part,label in zip(data_set,labels):
                ax.bar(grid,part,label = label,width = .95)
                max_value = np.max((max_value,np.max(part)))
        else:
            ax.plot(grid,data_set)
            max_value = np.max(data_set)
        if not isinstance(y_min,(bool)):
            ax.set_ylim(0,max_value)
        ax.legend(fontsize=16)
        if y_scale=='log':
            ax.set_yscale('log')
        if x_scale=='log':
            ax.set_xscale('log')

        apply_layout_to_ax(ax,layout)
        apply_layout_to_figure(fig,layout)
        return fig
    @classmethod
    def show(self,data_set,labels=False,grid=False,layout={},scale='lin'):
        fig = cls.get_fig(data_set,labels = labels, grid = grid, layout = layout, scale = scale)
        fig.show()

class hist1D:
    @classmethod
    def get_fig(cls,data_set,labels=False, grid = False,layout={},y_scale='lin',x_scale='lin',y_lim = False):
        log.info('y_lim={}'.format(y_lim))
        if isinstance(grid,bool):
            grid=np.arange(data_set.shape[0])
        log.info('grid type = {}'.format(type(grid)))
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        max_value = 0
        if data_set.ndim>1:
            if not isinstance(labels,(list,tuple)):
                labels = [None]*len(data_set)
            for part,label in zip(data_set,labels):
                ax.hist(part, bins='auto',label = make_string_tex_conform(label))
                max_value = np.max((max_value,np.max(part)))
        else:
            ax.hist(data_set,bins='auto')
            max_value = np.max(data_set)
        if isinstance(y_lim,(tuple,list)):
            ax.set_xlim(*y_lim) #on purpose since functionvalues are on x axis    
        ax.legend(fontsize=16)
        if y_scale=='log':
            ax.set_yscale('log')
        if x_scale=='log':
            ax.set_xscale('log')
        apply_layout_to_ax(ax,layout)
        apply_layout_to_figure(fig,layout)
        return fig
    @classmethod
    def show(cls,data_set,labels=False,grid=False,layout={},y_scale='lin'):
        fig = cls.get_fig(data_set,labels = labels, grid = grid, layout = layout, y_scale = y_scale)
        fig.show()        

class hist2D:
    @classmethod
    def get_fig(cls,data,bins = 10,labels=False, grid = False,layout={},y_scale='lin',x_scale='lin',range = None,norm = None,cmap ='viridis'):
        x=data[:,0]
        y=data[:,1]
        fig=plt.figure()
        ax=fig.add_subplot(111)
        cmap=plt.get_cmap(cmap).copy()
        if norm == 'log':
            cmap.set_bad(cmap(0))
            norm=matplotlib.colors.LogNorm()
            print('yay')
        hist = ax.hist2d(x,y,bins=bins,range = range,norm = norm,cmap=cmap)
        if y_scale=='log':
            ax.set_yscale('log')
        if x_scale=='log':
            ax.set_xscale('log')
        fig.colorbar(hist[3])
        apply_layout_to_ax(ax,layout)
        return fig
    @classmethod
    def show(cls,data_set,labels=False,grid=False,layout={},y_scale='lin'):
        fig = cls.get_fig(data_set,labels = labels, grid = grid, layout = layout, y_scale = y_scale)
        fig.show()        
        

class fxs_presenter:
    '''
    
    '''
    def applyLayoutToAx(self,ax,layout):
        title=layout.get('title',"Scatter plot 2D")
        title_fontsize=layout.get('title_fontsize',16)
        xLable=layout.get('x_label',"r")
        yLable=layout.get('y_label',"phi")
        label_fontsize=layout.get('label_fontsize',14)        
        ax.set_title(title,fontsize=title_fontsize)
        ax.set_xlabel(xLable,fontsize=label_fontsize)
        ax.set_ylabel(yLable,fontsize=label_fontsize)

    
    def present(self,plt_id,arg_dict):
#        log.info('arg dict={}'.format(arg_dict))
        if plt_id=='generic':            
            figures=self.generate_result_figures(arg_dict)
        elif plt_id=='reconstructed_densities':
            figures=self.generate_density_figures(arg_dict)
        elif plt_id=='reconstructed_densities_and_mask':
            figures=self.generate_density_mask_figures(arg_dict)
        elif plt_id=='reciprocal_projection_mask_sweep':
            figures=self.generate_reciprocal_projection_mask_sweep_figures(arg_dict)
        elif plt_id=='averaged_results':
            pass
        elif plt_id=='single_run_evolution':
            pass
        else:
            e=AssertionError('Unknown plot id: {} '.format(plt_id))
            log.error(e)
            raise
        return figures
        
    def present2(self):
        pass

    def generate_subplot(self,fig,ax_specifier,plot_dict):
        subplot_id=ax_specifier['key']
        if subplot_id=='real_density':
            self.draw_real_density_on_ax(fig,ax_specifier,plot_dict)
        elif subplot_id=='reciprocal_density':
            self.draw_reciprocal_density_on_ax(fig,ax_specifier,plot_dict)
        elif subplot_id=='support_mask':
            pass
        elif subplot_id=='error_metrics':
            self.draw_error_metric_on_ax(fig,ax_specifier)
        elif subplot_id=='harmonic_order_bCoeff':
            self.draw_bCoeff_on_ax(fig,ax_specifier,plot_dict)
            

    def generate_density_figures(self,data_dict):
        conf_dict=data_dict['conf_dict']
        harm_orders=conf_dict['fourier_transform']['harmonic_orders']
 
        
        plot_dict={}
        figures_data=[]
        n_datasets=len(data_dict['datasets'])
        title_part1=' (Run {} on {})'.format(data_dict.get('run','n/A'),data_dict.get('date','N/A'))
        title_part2=' consisting of {} HIO, {} ER and {} SW steps each. Using harmonic orders in $\{{ {},{},\ldots,{}\}}$'.format(conf_dict['main_loop']['nHIO'],conf_dict['main_loop']['nER'],conf_dict['main_loop']['nSW'],harm_orders.min(),harm_orders.min()+(harm_orders[1]-harm_orders[0]),harm_orders.max())
        for num,dataset in enumerate(data_dict['datasets']):
            axes_data={}
            axes_data['real_density']=[dataset['real_density'].real,{'data':dataset['initial_density'],'title':'Initial Density'}]
            axes_data['reciprocal_density']={'data':np.square(np.abs(dataset['reciprocal_density'])),'title':'Intensity'}
            axes_data['error_list']={'data':dataset['error_list'],'title':'Error Metric'}
            figure_data={'axes_data':axes_data}
            figure_data['title']='Reconstruction Results {} of {}'.format(num,n_datasets,plot_dict.get('run','n/A'),plot_dict.get('date','N/A'))+title_part1+'\n {} loops'.format(dataset['loop_iterations'])+title_part2
            figures_data.append(figure_data)
        plot_dict['figures_data']=figures_data
        plot_dict['default_figure_shape']=(int(2),int(2))
        plot_dict['default_subplot_positions']={'real_density':['221','223'],'reciprocal_density':['222'],'error_list':['224']}
        plot_dict['grid_pair']=conf_dict['internal_grids']
        figures=self.generate_figures(plot_dict)
        return figures

    def generate_density_mask_figures(self,data_dict):
        conf_dict=data_dict['conf_dict']
        harm_orders=conf_dict['fourier_transform']['harmonic_orders']
 #       log.info('configuration dict keys=\n {}'.format(conf_dict.keys()))
        
        plot_dict={}
        figures_data=[]
        n_datasets=len(data_dict['datasets'])
        title_part1=' (Run {} on {})'.format(data_dict.get('run','n/A'),data_dict.get('date','N/A'))
        title_part2=' consisting of {} HIO, {} ER and {} SW steps each. Using harmonic orders in $\{{ {},{},\ldots,{}\}}$'.format(conf_dict['main_loop']['HIO']['loop_iterations'],conf_dict['main_loop']['ER']['loop_iterations'],conf_dict['main_loop']['SW']['loop_iterations'],harm_orders.min(),harm_orders.min()+(harm_orders[1]-harm_orders[0]),harm_orders.max())
        for num,dataset in enumerate(data_dict['datasets']):
#            log.info('plot dataset ={}'.format(dataset))
#            log.info('dataset type ={}'.format(type(dataset)))
#            log.info('dataset shape={}'.format(dataset.shape))
#            log.info('dataset[0] type={}'.format(type(dataset[0])))
            
            axes_data={}
            axes_data['real_density']=[dataset['real_density'].real,{'data':dataset['support_mask'].astype(int),'title':'Support Mask'}]
            axes_data['reciprocal_density']={'data':np.square(np.abs(dataset['reciprocal_density'])),'title':'Intensity'}
            axes_data['error_list']={'data':dataset['error_list'],'title':'Error Metric'}
            figure_data={'axes_data':axes_data}
            figure_data['title']='Reconstruction Results {} of {}'.format(num,n_datasets,plot_dict.get('run','n/A'),plot_dict.get('date','N/A'))+title_part1+'\n {} loops'.format(dataset['loop_iterations'])+title_part2
            figures_data.append(figure_data)
        plot_dict['figures_data']=figures_data
        plot_dict['default_figure_shape']=(int(2),int(2))
        plot_dict['dec@fault_subplot_positions']={'real_density':['221','223'],'reciprocal_density':['222'],'error_list':['224']}
        plot_dict['grid_pair']=conf_dict['internal_grids']
        figures=self.generate_figures(plot_dict)
        return figures
    def generate_error_metric_figure(self,data_dict):
        grids=data_dict['configuration']['internal_grid']
        opt=data_dict['configuration']['opt']
        #       log.info('configuration dict keys=\n {}'.format(conf_dict.keys()))
        plt.rcParams.update({'font.size': 26})
        plot_dict={}
        figures_data=[]
        n_datasets=len(data_dict['reconstruction_results'])
        name = str(opt['name'])[2:-1].replace('_','\_')
        title_part1='{}\n {},  Run {}  \n'.format(name,data_dict.get('date','N/A').replace('_','\_'),data_dict.get('run','n/A'))
        log.info('reconstruction_result keys = {}'.format(data_dict['reconstruction_results'].keys()))
        for num,dataset in data_dict['reconstruction_results'].items():
#            log.info('plot dataset ={}'.format(dataset))
#            log.info('dataset type ={}'.format(type(dataset)))
#            log.info('dataset shape={}'.format(dataset.shape))
#            log.info('dataset[0] type={}'.format(type(dataset[0])))
            
            axes_data={}            
            axes_data['error_metrics']={'data':dataset['error_dict'],'title':'Error Metrics'}
            figure_data={'axes_data':axes_data}
            figure_data['title']=title_part1+'Reconstruction Results {} of {}'.format(int(num)+1,n_datasets)
            figures_data.append(figure_data)
        plot_dict['figures_data']=figures_data
        plot_dict['default_figure_shape']=(int(1),int(1))
        plot_dict['dec@fault_subplot_positions']={'error_metrics':['111']}
        plot_dict['grid_pair']=grids
        figures=self.generate_figures(plot_dict)
        return figures

    def generate_reciprocal_projection_mask_sweep_figures(self,data_dict):
        conf_dict=data_dict['conf_dict']
        masks=conf_dict['multi_process']['arg_dict']['reciprocal_projection']['mask']
        harm_orders=conf_dict['used_harmonic_orders']
        log.info('configuration dict keys=\n {}'.format(conf_dict.keys()))
        
        plot_dict={}
        figures_data=[]
        n_datasets=len(data_dict['datasets'])
        title_part1=' (Run {} on {})'.format(data_dict.get('run','n/A'),data_dict.get('date','N/A'))
        title_part2=' consisting of {} HIO, {} ER and {} SW steps each + {} ER refinment steps. Using harmonic orders in $\{{ {},{},\ldots,{}\}}$'.format(conf_dict['main_loop']['HIO_iterations'],conf_dict['main_loop']['ER_iterations'],conf_dict['main_loop']['SW_iterations'],conf_dict['main_loop']['ER_refinement_iterations'],harm_orders.min(),harm_orders.min()+(harm_orders[1]-harm_orders[0]),harm_orders.max())
        for num,dataset in enumerate(data_dict['datasets']):
            axes_data={}
            axes_data['real_density']=[dataset['real_density'].real,{'data':dataset['support_mask'].astype(int),'title':'Support Mask'}]
            reciprocal_density=np.square(np.abs(dataset['reciprocal_density']))
            reciprocal_density[~masks[num]]*=1e-3
            axes_data['reciprocal_density']={'data':reciprocal_density,'title':'Intensity'}
            axes_data['error_list']={'data':dataset['error_list'],'title':'Error Metric'}
            figure_data={'axes_data':axes_data}
            figure_data['title']='Reconstruction Results {} of {}'.format(num,n_datasets,plot_dict.get('run','n/A'),plot_dict.get('date','N/A'))+title_part1+'\n {} loops'.format(dataset['loop_iterations'])+title_part2
            figures_data.append(figure_data)
        plot_dict['figures_data']=figures_data
        plot_dict['figure_shape']=(int(2),int(2))
        plot_dict['default_subplot_positions']={'real_density':['221','223'],'reciprocal_density':['222'],'error_list':['224']}
        plot_dict['grid_pair']=conf_dict['internal_grids']
        figures=self.generate_figures(plot_dict)
        return figures
    

    def generate_result_figures(self,plot_dict):
        figures=self.generate_figures(plot_dict)
        return figures


    def generate_figures(self,plot_dict):
        def calc_plot_shape(n_subplots):
            if n_subplots==1:
                shape=(1,1)
            elif n_subplots==2:
                shape=(1,2)
            else:
                n_rows_prelim=np.floor(np.sqrt(n_subplots))
                n_rows=n_rows_prelim+n_rows_prelim%2
                n_cols=np.ceil(n_subplots/n_rows)
                shape=(int(n_rows),int(n_cols))
#            log.info('shape={}'.format(shape))
            return shape

        def calc_n_subplots_per_key(axes_data,keys):
            n_subplots_per_key={}
            for key in keys:
                dataset=axes_data[key]
                if isinstance(dataset,(list,tuple)):
                    n_subplots_per_key[key]=len(dataset)
                else:
                    n_subplots_per_key[key]=1
#            log.info('n_subplots_per_key={}'.format(n_subplots_per_key))
            return n_subplots_per_key
        
        def generate_subplot_positions(plot_shape,n_subplots_per_key):
            n_rows=plot_shape[0]
            n_assigned_positions=0
            subplot_positions={}            
            for key in n_subplots_per_key.keys():
                subplot_positions[key]=[]                
                for subplot_num in range(1,n_subplots_per_key[key]+1):
                    num=str(n_assigned_positions+subplot_num)
                    subplot_positions[key].append(str(plot_shape[0])+str(plot_shape[1])+num)
                n_assigned_positions+=n_subplots_per_key[key]
#            log.info('subplot_positions=\n{}'.format(subplot_positions))   
            return subplot_positions
        def extract_figures_data(plot_dict):        
            figure_data=plot_dict.get('figure_data',False)
            if not isinstance(figure_data,bool):
                figures_data=[figure_data]
            else:
                figures_data=plot_dict['figure_data']
            return figures_data

        def select_first_non_bool(*args):
            selection=False
            for arg in args:
                if not isinstance(arg,bool):
                    selection=arg
                    break
            return selection

        def call_generate_subplot(key,ax_position,dataset,ax_titles,fig,plot_dict):
            if isinstance(dataset,dict):
                data=dataset.pop('data')
                ax_options=dataset                
            else:
                data=dataset
                ax_options={}
                
            if not isinstance(ax_titles,bool):
                title_by_pos=ax_titles.get(ax_position,False)
#                log.info('axoptions = {}'.format(ax_options))
                ax_options['title_by_pos']=title_by_pos
#            log.info('ax_option keys={}'.format(ax_options.keys()))
            ax_specifier={'key':key,'dataset':data,'position':ax_position,'options':ax_options}
            self.generate_subplot(fig,ax_specifier,plot_dict)
            
        figures_data=extract_figures_data(plot_dict)
        default_figure_shape=plot_dict.get('default_figure_shape',False)
        default_datakeys_to_plot=plot_dict.get('default_datakeys_to_plot',False)
        default_subplot_positions=plot_dict.get('default_subplot_positions',False)
        default_title=plot_dict.get('default_title',False)
        
#        plt.rcParams['font.size'] = plot_dict.get('default_font_size',16)
        figure_list=[]
        for figure_data in figures_data:
            axes_data=figure_data['axes_data']
            datakeys_to_plot=select_first_non_bool(figure_data.get('datakeys_to_plot',False),default_datakeys_to_plot,list(axes_data))
            n_subplots_per_key=calc_n_subplots_per_key(axes_data,datakeys_to_plot)
            n_subplots=sum(n_subplots_per_key.values())
            shape=select_first_non_bool(figure_data.get('shape',False),default_figure_shape,calc_plot_shape(n_subplots))
            try:
                assert np.prod(shape)>=n_subplots,'Specified plot shape {} conflicts with number of subplots:{}. Using autogenerated shape!'.format(shape,n_subplots)
            except AssertionError as e:
                log.error(e)
                shape=calc_plot_shape(n_subplots)            
            subplot_positions=select_first_non_bool(figure_data.get('subplot_positions',False),default_subplot_positions,generate_subplot_positions(shape,n_subplots_per_key))
            title=select_first_non_bool(figure_data.get('title',False),default_title,'')
            ax_titles=select_first_non_bool(plot_dict.get('default_ax_titles',False),figure_data.get('ax_titles',False))
            
            fig=plt.figure(figsize=(20,20))
            fig.suptitle(title)
            for key,datasets in figure_data['axes_data'].items():
#                log.info('key={}'.format(key))
                subplot_positions_per_key=subplot_positions[key].copy()
                if isinstance(datasets,(list,tuple)):
                    for dataset in datasets:
                        ax_position=subplot_positions_per_key.pop(0)
                        call_generate_subplot(key,ax_position,dataset,ax_titles,fig,plot_dict)
                else:
                    dataset=datasets
                    ax_position=subplot_positions_per_key.pop(0)
                    call_generate_subplot(key,ax_position,dataset,ax_titles,fig,plot_dict)
            figure_list.append(fig)
        return figure_list

    
    def draw_1D_funciton_on_ax(self,data,fig,ax,layout={},scale='lin'):
#        log.info('1D data=\n{}'.format(data))
        ax.plot(np.arange(data.shape[0]),data)
        self.applyLayoutToAx(ax,layout)
        if scale=='log':
            ax.set_yscale('log')
            
    def draw_polar_heatplot_on_ax(self,data,fig,ax,layout={},scale='lin',grid=False):
        #            cmap=plt.get_cmap('viridis')
        cmap=plt.get_cmap('inferno')

        if not isinstance(grid,bool):
            grid=grid.copy()
            #        log.info('grid shape = {}'.format(grid.shape))
            grid.array=np.concatenate((grid[:],np.expand_dims(grid[:,0],1)),axis=1)
            r=np.swapaxes(grid[...,0],0,1)
            phi=np.swapaxes(grid[...,1],0,1)
            #            log.info('phi shape={}'.format(phi.shape))
        else:
            shape=data.shape
            rValues=np.arange(0,shape[0])
            phiValues=np.zeros(shape[1]+1)
            phiValues[:-1]=np.arange(0,shape[1])*2*np.pi/(shape[1])            
            r,phi=np.meshgrid(rValues,phiValues)
            
        if type(data)==Grid:
            data=data.array
                
        if scale=='log':
            cmap.set_bad(cmap(0))
            heatmap=ax.pcolormesh(phi,r,np.swapaxes(data,0,1),norm=matplotlib.colors.LogNorm(),cmap=cmap)
        else:
            heatmap=ax.pcolormesh(phi,r,np.swapaxes(data,0,1),cmap=cmap)
#        log.info(' last r value = {}'.format(r[-1]))
        ax.set_yticks([r[-1,-1]])
        axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="90%",  # height : 50%
                   loc='center left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
        fig.colorbar(heatmap,cax=axins,ax=ax)        
        self.applyLayoutToAx(ax,layout) 

    def draw_heatplot_on_ax(self,data,fig,ax,layout={},scale='lin',grid=False):
        #            cmap=plt.get_cmap('viridis')
        cmap=plt.get_cmap('inferno')
        if not isinstance(grid,bool):
            grid=Grid.copy(grid)
            #        log.info('grid shape = {}'.format(grid.shape))
            x=np.swapaxes(grid[:][:,0].reshape(grid.shape),0,1)
            y=np.swapaxes(grid[:][:,1].reshape(grid.shape),0,1)
            #            log.info('phi shape={}'.format(phi.shape))
        else:
            shape=data.shape
            xValues=np.arange(0,shape[0])
            yValues=np.arange(0,shape[1])            
            x,y=np.meshgrid(xValues,yValues)
            
        if type(data)==Grid:
            data=data.array
#        log.info('bcoeff data shape=\n{}'.format(data.shape))
        if scale=='log':
            cmap.set_bad(cmap(0))
            heatmap=ax.pcolormesh(x,y,np.swapaxes(data,0,1),norm=matplotlib.colors.LogNorm(),cmap=cmap)
        else:
            heatmap=ax.pcolormesh(x,y,np.swapaxes(data,0,1),cmap=cmap)
        fig.colorbar(heatmap,ax=ax)        
        self.applyLayoutToAx(ax,layout) 
        
    def draw_error_metric_on_ax(self,fig,ax_specifier):
        position=ax_specifier['position']
        title=ax_specifier['options'].get('title',False)
        std_error=ax_specifier['dataset']['real_rel_mean_square']
        cauchy_real=ax_specifier['dataset']['cauchy_real']
        cauchy_reciprocal=ax_specifier['dataset']['cauchy_reciprocal']
        _max=cauchy_real.max()
        cauchy_real/=_max
        cauchy_reciprocal/=_max
        cauchy_real[:2]=[2,1]
        cauchy_reciprocal[:2]=[2,1]

        x=np.arange(len(std_error))
        ax=fig.add_subplot()
        layout={'title':'','x_label':'Number of Reconstruciton Steps','y_label':'Error Estimate','title_fontsize':26,'label_fontsize':26}
        if not isinstance(title,bool):
            layout['title']=title
        
        ax.plot(x,cauchy_real,'#20F0A0')
        ax.plot(x,cauchy_reciprocal,'#8080F0')
        ax.plot(x,std_error,'k', linewidth=3)
        ax.legend(('Real: Cauchy', 'Reciprocal: Cauchy','Real: Relative Squared Intensity'),
           loc='upper right')
        self.applyLayoutToAx(ax,layout)
        ax.set_yscale('log')


        
    def draw_real_density_on_ax(self,fig,ax_specifier,plot_dict):
        position=ax_specifier['position']
        title=ax_specifier['options'].get('title',False)
        dataset=ax_specifier['dataset']
        log.info('dataset shape = {}'.format(dataset.shape))
        
        ax=fig.add_subplot(position,projection='polar')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        grid_pair=plot_dict.get('grid_pair',False)
        if isinstance(grid_pair,bool):
            realGrid=False
        else:
            realGrid=grid_pair.realGrid
        layout={'title':'Real Density','x_label':'','y_label':'','title_fontsize':12,'label_fontsize':10}
        if not isinstance(title,bool):
            layout['title']=title
        #log.info('realGrid.shape={},dataset shape = {}'.format(realGrid.shape,dataset.array.shape))
        self.draw_polar_heatplot_on_ax(dataset,fig,ax,layout=layout,grid=realGrid)

    def draw_reciprocal_density_on_ax(self,fig,ax_specifier,plot_dict):
        position=ax_specifier['position']
        title=ax_specifier['options'].get('title',False)
        dataset=ax_specifier['dataset']
        dataset=np.where(dataset<=0,1e-12,dataset)
        
        ax=fig.add_subplot(position,projection='polar')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        grid_pair=plot_dict.get('grid_pair',False)
        if isinstance(grid_pair,bool):
            reciprocalGrid=False
        else:
            reciprocalGrid=grid_pair.reciprocalGrid
            
        layout={'title':'reciprocal density','x_label':'','y_label':'','title_fontsize':12,'label_fontsize':10}
        if not isinstance(title,bool):
            layout['title']=title
        #log.info('realGrid.shape={},dataset shape = {}'.format(realGrid.shape,dataset.array.shape))
        self.draw_polar_heatplot_on_ax(dataset,fig,ax,layout=layout,grid=reciprocalGrid,scale='log')

    def draw_bCoeff_on_ax(self,fig,ax_specifier,plot_dict):
        options=ax_specifier['options']
        order=options['order']
        position=ax_specifier['position']
        title=options.get('title',False)        
        dataset=ax_specifier['dataset']
        
        ax=fig.add_subplot(position)
        grid_pair=plot_dict.get('grid_pair',False)
        if isinstance(grid_pair,bool):
            reciprocalGrid=False
        else:
            reciprocalGrid=grid_pair.reciprocalGrid
            coords=reciprocalGrid[:,0][:,0]
            grid=getGridByXYValues(coords,coords)
            
        
            
        layout={'title':'B-Coefficients','x_label':'$q_1$  []','y_label':'$q_2$  []','title_fontsize':12,'label_fontsize':10}
        if not isinstance(title,bool):
            layout['title']=title
#        log.info('data shape = {}'.format(dataset.shape))
#        log.info('grid shape = {}'.format(grid.array.shape))
        self.draw_heatplot_on_ax(dataset,fig,ax,layout=layout,scale='log')
        
    def generate_reciprocal_density_plot(self,dataset,fig,ax):        
        pass
