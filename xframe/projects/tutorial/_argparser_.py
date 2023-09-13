from xframe.startup_routines import DefaultProjectArgparser
from xframe.startup_routines import DefaultProjectClick

class ProjectArgparser(DefaultProjectArgparser):
    def __init__(self):
        super().__init__()
        self.project_description = 'How to setup a project in xFrame'




class ProjectClick(DefaultProjectClick):
    def __init__(self):
        super().__init__()
        self.project_description = 'A short tutorial explaining how to setup a project in xFrame. With details on Settings and File handling as well as multiprocessing and GPU access.'
        self.short_description = 'Example project in xFrame'
