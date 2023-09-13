import click as cl
from xframe.interfaces import ClickInterface

class ClickPlugin(ClickInterface):
    def __init__(self):
        self._click = cl
    @property
    def click(self):
        return self._click
        
