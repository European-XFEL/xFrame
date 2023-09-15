from xframe.startup_routines import DefaultProjectArgparser
from xframe.startup_routines import DefaultProjectClick

class ProjectArgparser(DefaultProjectArgparser):
    def __init__(self):
        super().__init__()
        self.project_description = 'Toolkit for the analysis of fluctuation X-ray scattering (FXS) experiments.'
    def add_correlate_argparser(self,subparser,worker):
        subparser.add_parser(worker, description='computes angular cross-correlations',help=f'computes angular cross-correlations',parents=[self.settings_parser])
    def add_extract_argparser(self,subparser,worker):
        subparser.add_parser(worker, description='extracts rotational invarants from cross-correlation data',help=f'extracts rotational invarants from cross-correlation data',parents=[self.settings_parser])
    def add_reconstruct_argparser(self,subparser,worker):
        subparser.add_parser(worker, description='phase retrieval based on rotational invariants',help=f'phase retrieval based on rotational invariants',parents=[self.settings_parser])
    def add_average_argparser(self,subparser,worker):
        parser = subparser.add_parser(worker, description='aligns and averages phase retrieval results',help=f'aligns and averages phase retrieval results',parents=[self.settings_parser])
    def add_reconstruct2_argparser(self,subparser,worker):
        pass
    def add_simulate_ccd_argparser(self,subparser,worker):
        subparser.add_parser(worker, description='Only for testing: generates ccd for simple shapes',help=f'Only for testing: generates ccd for simple shapes',parents=[self.settings_parser])        



class ProjectClick(DefaultProjectClick):
    def __init__(self):
        super().__init__()
        self.project_description = 'Toolkit for the analysis of fluctuation X-ray scattering (FXS) experiments.'
        self.short_description = 'Fluctuation X-ray scattering toolkit'

        self.worker_short_help = {
            'correlate':'computes angular cross-correlations ',
            'extract':'extracts rotational invarants',
            'reconstruct':'runs phase retrieval',
            'average':'aligns and averages reconstructions',
            'simulate_ccd':"testing: simulates cc's for simple shapes"
        }
        
        self.worker_help = {
            'correlate':'Computes the averaged angular cross-correlation function C(q1,q2,\delta) for a set of diffraction patterns. A settings file name must be provided via SETTINGS_NAME.',
            'extract':'Extracts rotational invariants Bl(q1,q2) from an averaged angular cross-correlation function and computes the projection matrices Vl|vn necessary for phase retrieval. A settings file name must be provided via SETTINGS_NAME.',
            'reconstruct':'Reconstructs the single particle electron density using the MTIP scheme for iterative phase retrieval. A settings file name must be provided via SETTINGS_NAME.',
            'average':'Aligns and averages multiple reconstructions. A settings file name must be provided via SETTINGS_NAME.',
            'simulate_ccd': 'Testing/Tutorial only: This tool allows to calculate averaged cross-correlation functions C(q1,q2,\delta) from simple shapes (known electron densities). A settings file name must be provided via SETTINGS_NAME.'
        }
        self.worker_epilog = {
            'correlate':'',
            'extract':'',
            'reconstruct':'',
            'average':'',
            'simulate_ccd':''
        }

    def add_fxs(self,group,click):
        @group.group(self.project_name,chain=True,help = self.project_description, short_help= self.short_description)
        def project():
            pass
        #project = self.add_default_project_click(group,click)
        return project
    def add_worker_correlate(self,group,click,worker):
        short_help=self.worker_short_help['correlate']
        help=self.worker_help['correlate']
        self.add_default_worker_click(group,click,worker,short_help=short_help,help=help)
    def add_worker_extract(self,group,click,worker):
        short_help=self.worker_short_help['extract']
        help=self.worker_help['extract']
        self.add_default_worker_click(group,click,worker,short_help=short_help,help=help)
    def add_worker_reconstruct(self,group,click,worker):
        short_help=self.worker_short_help['reconstruct']
        help=self.worker_help['reconstruct']
        self.add_default_worker_click(group,click,worker,short_help=short_help,help=help)
    def add_worker_average(self,group,click,worker):
        short_help=self.worker_short_help['average']
        help=self.worker_help['average']
        self.add_default_worker_click(group,click,worker,short_help=short_help,help=help)
    def add_worker_reconstruct2(self,group,click,worker):
        pass
    def add_worker_simulate_ccd(self,group,click,worker):
        short_help=self.worker_short_help['simulate_ccd']
        help=self.worker_help['simulate_ccd']
        self.add_default_worker_click(group,click,worker,short_help=short_help,help=help)
 
