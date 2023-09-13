import cv2 as cv
import numpy as np
from xframe.database.interfaces import OpenCVInterface
from xframe.presenters.openCVPresenter import OpenCVInterface as PresenterInterface

class CV_Plugin(OpenCVInterface,PresenterInterface):
    colormaps={
        'autumn':cv.COLORMAP_AUTUMN,
        'bone':cv.COLORMAP_BONE,
        'jet':cv.COLORMAP_JET,
        'winter':cv.COLORMAP_WINTER,
        'rainbow':cv.COLORMAP_RAINBOW,
        'ocean': cv.COLORMAP_OCEAN,
        'summer': cv.COLORMAP_SUMMER,
        'spring': cv.COLORMAP_SPRING,
        'cool': cv.COLORMAP_COOL,
        'hsv': cv.COLORMAP_HSV,
        'pink': cv.COLORMAP_PINK,
        'hot': cv.COLORMAP_HOT,
        'parula': cv.COLORMAP_PARULA,
        'magma': cv.COLORMAP_MAGMA,
        'inferno': cv.COLORMAP_INFERNO,
        'plasma': cv.COLORMAP_PLASMA,
        'viridis': cv.COLORMAP_VIRIDIS,
        'cvidis': cv.COLORMAP_CIVIDIS,
        'twilight': cv.COLORMAP_TWILIGHT,
        'twilight_shfted': cv.COLORMAP_TWILIGHT_SHIFTED,
        'turbo': cv.COLORMAP_TURBO,
        'deep_green':cv.COLORMAP_DEEPGREEN
    }

    @classmethod
    def get_polar_image_data(cls,data,n_pixels=False):
        Nr,Nphi=data.shape
        pdata=data.copy()
        _min = pdata.min()
        if isinstance(n_pixels,bool):
            n_pixels=Nr*2
        radius = int(n_pixels//2)
        center = (int(n_pixels//2),int(n_pixels//2))
            
        pic = cv.warpPolar(pdata.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)
        return pic
    

    @classmethod
    def get_polar_image(cls,data,n_pixels=False,scale='lin',colormap='viridis',vmin=False,vmax=False,print_colorscale=False,transparent_backgound=False):
        Nr,Nphi=data.shape
        pdata=data.copy()
        use_log_scale= (scale=='log')
        if use_log_scale:
            pdata[pdata<=0]=1e-15
            pdata=np.log10(pdata)         
        

        max_d = pdata.max()
        min_d = pdata.min()
        if isinstance(vmin,bool):
            vmin=min_d
        else:
            if use_log_scale:
                vmin=np.log10(np.abs(vmin))
            pdata[pdata<vmin]=vmin
            
        if isinstance(vmax,bool):
            vmax=max_d
        else:
            if use_log_scale:
                vmax=np.log10(np.abs(vmax))
            pdata[pdata>vmax]=vmax
        zero_in_range= (0>=vmin) and (0<=vmax)
        if zero_in_range:
            bg_value = 0
        else:
            bg_value = vmin

        enlarged_Nr = int(Nr*np.sqrt(2))
        temp_pic = np.zeros((enlarged_Nr,Nphi),dtype=float)
        bg_pic = np.zeros((enlarged_Nr,Nphi),dtype=float)
        temp_pic[:Nr]=pdata
        temp_pic[Nr:]=bg_value
        bg_pic[Nr:]=1
        
        if isinstance(n_pixels,bool):
            n_pixels=Nr*2
        radius = int(n_pixels/np.sqrt(2))
        center = (int(n_pixels//2),int(n_pixels//2))
            
        pic = cv.warpPolar(temp_pic.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)
        bg_mask = cv.warpPolar(bg_pic.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)==1
        
        pic-=vmin
        pic*=255/(vmax-vmin)
        pic=pic.astype(np.uint8)
        pic= cv.applyColorMap(pic, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
        if print_colorscale:
            h_black = int(pic.shape[0]*0.05)
            h_color_bar = int(pic.shape[0]*0.05)
            cscale=np.zeros((h_black+h_color_bar,pic.shape[0]),dtype=np.uint8)
            cscale[h_black:]=np.arange(pic.shape[0])*255/pic.shape[0]
            cscale = cv.applyColorMap(cscale, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
            cscale[:h_black]=0

            if use_log_scale:
                vmin_txt = f"{10**(vmin):1.2e}"
                vmax_txt = f"{10**(vmax):1.2e}"
            else:
                vmin_txt = f"{vmin:1.2e}"
                vmax_txt = f"{vmax:1.2e}"
            WHITE = (255,255,255)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.03 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
            font_size = n_pixels/(25/scale)
            font_color = WHITE
            font_thickness = int(font_size*2)
            img_text = cv.putText(cscale, vmin_txt, (int(n_pixels*0),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            img_text = cv.putText(cscale, vmax_txt, (int(n_pixels*0.80),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            pic=np.concatenate((pic,cscale),axis=0)
            

        if transparent_backgound:
            # First create the image with alpha channel
            pic_rgba = cv.cvtColor(pic, cv.COLOR_RGB2RGBA)

            # Then assign the mask to the last channel of the image
            pic_rgba[:n_pixels, :n_pixels, 3] = ((~bg_mask).astype(np.uint8)*255)
            pic=pic_rgba.copy()

        return pic


    @classmethod
    def get_cart_image(cls,data,scale='lin',colormap='viridis',vmin=False,vmax=False,print_colorscale=False):
        Nx,Ny=data.shape
        pdata=data.copy()
        n_pixels = Nx
        use_log_scale= (scale=='log')
        if use_log_scale:
            pdata[pdata<=0]=1e-15
            pdata=np.log10(pdata)         
        

        max_d = pdata.max()
        min_d = pdata.min()
        if isinstance(vmin,bool):
            vmin=min_d
        else:
            if use_log_scale:
                vmin=np.log10(np.abs(vmin))
            pdata[pdata<vmin]=vmin
            
        if isinstance(vmax,bool):
            vmax=max_d
        else:
            if use_log_scale:
                vmax=np.log10(np.abs(vmax))
            pdata[pdata>vmax]=vmax
        zero_in_range= (0>=vmin) and (0<=vmax)
        if zero_in_range:
            bg_value = 0
        else:
            bg_value = vmin

        pic = pdata
        
        pic-=vmin
        pic*=255/(vmax-vmin)
        pic=pic.astype(np.uint8)
        pic= cv.applyColorMap(pic, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
        if print_colorscale:
            h_black = int(pic.shape[0]*0.05)
            h_color_bar = int(pic.shape[0]*0.05)
            cscale=np.zeros((h_black+h_color_bar,pic.shape[0]),dtype=np.uint8)
            cscale[h_black:]=np.arange(pic.shape[0])*255/pic.shape[0]
            cscale = cv.applyColorMap(cscale, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
            cscale[:h_black]=0

            if use_log_scale:
                vmin_txt = f"{10**(vmin):1.2e}"
                vmax_txt = f"{10**(vmax):1.2e}"
            else:
                vmin_txt = f"{vmin:1.2e}"
                vmax_txt = f"{vmax:1.2e}"
            WHITE = (255,255,255)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.03 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
            font_size = n_pixels/(25/scale)
            font_color = WHITE
            font_thickness = int(font_size*2)
            img_text = cv.putText(cscale, vmin_txt, (int(n_pixels*0),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            img_text = cv.putText(cscale, vmax_txt, (int(n_pixels*0.80),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            pic=np.concatenate((pic,cscale),axis=0)            
        return pic

    @classmethod
    def save(cls,path,image):
        path=path[:-3]+'.png'
        cv.imwrite(path, image)
        cv.waitKey(0)
    def load(cls,path):
        image = cv.imread(path)
        cv.waitKey(0)
        return image
