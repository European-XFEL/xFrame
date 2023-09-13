import flt
from xframe.library.interfaces import DiscreteLegendreTransform_interface

class LegendreTransform(DiscreteLegendreTransform_interface):
    lib = flt
    @staticmethod
    def forward(x,closed = False):
        return flt.dlt(x,closed = closed)
    @staticmethod
    def inverse(x,closed = False):
        return flt.idlt(x, closed = closed)
