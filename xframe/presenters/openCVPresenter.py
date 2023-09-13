import logging
log = logging.getLogger('root')
from xframe.presenters.interfaces import OpenCVInterface

CV_Plugin=OpenCVInterface

class Polar2D:
    @staticmethod
    def get_fig(*args,**kwargs):
        image=CV_Plugin.get_polar_image(*args,**kwargs)
        return image
    def get_fig_data(*args,**kwargs):
        image=CV_Plugin.get_polar_image_data(*args,**kwargs)
        return image

class Cart2D:
    @staticmethod
    def get_fig(*args,**kwargs):
        image=CV_Plugin.get_cart_image(*args,**kwargs)
        return image
