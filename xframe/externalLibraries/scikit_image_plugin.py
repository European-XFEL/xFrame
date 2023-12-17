import skimage
from skimage.restoration import denoise_tv_chambolle as denoiser
from xframe.library.interfaces import SkimageInterface

class SkimagePlugin(SkimageInterface):
    lib = skimage
    @staticmethod
    def denoise_tv_chambolle(data,**kwargs):
        return denoiser(data,**kwargs)
