import argparse
import importlib
import time,timeit
import profile
import time
import traceback
import os
import sys
from os.path import join as pjoin



def create_project_parsers(parser,known_projects,projects_to_load,default):
   from xframe.startup_routines import DefaultProjectArgparser
   
   if not isinstance(projects_to_load,(list,tuple)):
      projects_to_load = tuple(known_projects.keys())
   subparser = parser.add_subparsers(help='Available Projects',dest='project')
   subparser.default=default
   for project_name,project_path in known_projects.items():
      if project_name in projects_to_load:
         try:
            _argparser_ = importlib.import_module('xframe.projects.'+project_name+'._argparser_')
            #print('alive')
            p = getattr(_argparser_,'ProjectArgparser',DefaultProjectArgparser)()
            #print('alive2')
            p.add_parser(subparser,project_name,project_path)
            #print('alive3')
         except Exception as e:
            import logging
            log = logging.getLogger('root')
            log.info(f'processing parser of Project {project_name} failed with error:\ {e}')
            p = DefaultProjectArgparser()
            p.add_parser(subparser,project_name,project_path)

def create_experiment_parsers(parser,known_experiments,experiments_to_load):
   if not isinstance(experiments_to_load,(list,tuple)):
      experiments_to_load = tuple(known_projects.keys())
   subparser = parser.add_subparsers(help='Available xframe Experiments')      
   for exp_name in known_experiments:
      if exp_name in experiments_to_load:
         try:
            _argparser_ = importlib.import_module('xframe.experiments.'+exp_name+'._argparser_')
            exp_parser = subparser.add_parser(exp_name, help=_argparser_.help)
            #_argparser_.add_to_parser(exp_parser)               
         except Exception as e:
            import logging
            log = logging.getLogger('root')
            log.info(f'processing parser of Project {project_name} failed with error:\ {e}')
            project_parser = subparser.add_parser(project_name, help='No help available')

def create_argument_parser(defaults,xframe):
   p_name = defaults['project_name']
   p_opt = defaults['project_opt']
   e_name = defaults['exp_name']
   e_opt = defaults['exp_opt']
   xframe_help = f'xFrame {xframe.__version__}: A framwork for scientific computing targeting X-ray scattering experiments.'
   parser = argparse.ArgumentParser(prog='xframe', description=xframe_help)
   #parser.add_argument('ana_name', help='analysisWorker name', nargs = '?',default = a_name , type=str)
   parser.add_argument('-e','--experiment',choices=list(xframe.known_experiments.keys()), help='Available Experiments',default = e_name , type=str)
   parser.add_argument('-eset','--experiment_settings', help='experiment settings to be used',metavar='FILE_NAME',default = e_opt , type=str)
   parser.add_argument('-v','--verbose', help='Set loglevel to INFO.',action='store_true')
   parser.add_argument('-d','--debug', help='Set loglevel to DEBUG.',action='store_true')
   parser.add_argument('--version',action='store_true')
   parser.add_argument('--setup_home',metavar='HOME_PATH',nargs = '?',const='~/.xframe/',default=False, help='setup xframe home directory at HOME_PATH default is ~/.xframe')
   parser.add_argument('--print_home',action='store_true', help='show current home path')
   
   create_project_parsers(parser,xframe.known_projects,xframe.settings.general.load_projects,p_name)
   return parser

def start_routine_argparse():
   import xframe
   import logging

   defaults = {'project_name':False,'project_opt':False,'exp_name':False,'exp_opt':False}
   parser = create_argument_parser(defaults,xframe)
   args=parser.parse_args()
   #print(f'parsed args = {args}')
   if args.debug:
      logging.getLogger('root').setLevel(logging.DEBUG)
   elif args.verbose:
      logging.getLogger('root').setLevel(logging.INFO)   

      
   if args.setup_home:      
      from xframe.startup_routines import setup_home
      setup_home(args.setup_home)
   if args.print_home:
      xframe.lib.python.xprint(f'Current xframe home directory is: {xframe.settings.general.home_folder}')
   if args.version:
      xframe.lib.python.xprint(f'xFrame {xframe.__version__}')

   if not isinstance(args.project,bool):
      results = xframe.select_and_run(project = args.project, project_worker=args.worker,project_settings=args.settings,exp_name=args.experiment,exp_settings=args.experiment_settings,oneshot = True)
      xframe.Multiprocessing.update_free_cpus()
   else:
      logging.getLogger('root').info('No project name was given, terminating.')
   sys.exit()



def start_routine_click(click):
   import xframe
   import logging

   xprint = xframe.library.pythonLibrary.xprint
   db = xframe.database.default
   
   startup_dict = {'experiment_name':False,'exp_settings':False}
   known_experiments = xframe.known_experiments
   xframe_help = f'xFrame {xframe.__version__}: A framwork for scientific algorithms targeting X-ray scattering'
   
   @click.group(invoke_without_command=True,help=xframe_help)
   @click.option('--setup_home',is_flag = False,flag_value=xframe.settings.general.default_folder,is_eager=True,type=click.Path(exists=False,resolve_path=True), help='create home folder. If PATH not specified use ~/.xframe')
   @click.option('--print_home',is_flag = True,is_eager=True, help='print path of current home folder')
   @click.option('-e','--experiment',is_flag = False,flag_value=False,help='specify optional experiment module',type=click.Choice(list(known_experiments.keys()), case_sensitive=True))
   @click.option('-eopt','--experiment_options',is_flag = False,flag_value=False,help='settings name for experiment module',type=str)
   @click.option('-d','--debug', is_flag=True, help = 'set loglevel to DEBUG')
   @click.option('-v','--verbose', is_flag = True, help = 'set loglevel to INFO')
   @click.option('--version', is_flag = True,is_eager=True)
   @click.pass_context
   def run(ctx,*args,**kwargs):
      if kwargs['debug']:
         logging.getLogger('root').setLevel(logging.DEBUG)
         if ctx.invoked_subcommand is None:
            print(ctx.get_help())
      elif kwargs['verbose']:
         logging.getLogger('root').setLevel(logging.INFO)
         if ctx.invoked_subcommand is None:
            print(ctx.get_help())
      if isinstance(kwargs['setup_home'],str):
         xframe.startup_routines.setup_home(kwargs['setup_home'])
      if kwargs['print_home']:
         xprint(f"xframe's current home directory is: {xframe.settings.general.home_folder}")
      if kwargs['version']:
         xprint(f'xFrame {xframe.__version__}')
      if isinstance(kwargs['experiment'],str):
         startup_dict['experiment_name']=kwargs['experiment']
         xframe.select_experiment(exp_name=kwargs['experiment'],exp_settings=kwargs['experiment_options'])
         if ctx.invoked_subcommand is None:
            print(ctx.get_help())

      @ctx.call_on_close
      def close_workers():
         xframe.controller.control_worker.stop_working()
         del(xframe.controller.control_worker.gpu_manager)

   from xframe.startup_routines import DefaultProjectClick
   projects_to_load = xframe.settings.general.load_projects
   known_projects = xframe.known_projects
   if not isinstance(projects_to_load,(list,tuple)):
      projects_to_load = tuple(known_projects.keys())
   for project_name,project_path in known_projects.items():
      if project_name in projects_to_load:
         try:
            _argparser_ = importlib.import_module('xframe.projects.'+project_name+'._argparser_')
            p = getattr(_argparser_,'ProjectClick',DefaultProjectClick)()
            p.add_click(run,click,project_name,project_path)
         except Exception as e:
            import logging
            log = logging.getLogger('root')
            log.info(f'processing parser of Project {project_name} failed with error:\ {e}.Use default.')
            p = DefaultProjectClick()
            p.add_click(run,click,project_name,project_path)
   run()

   
def start_routine_cmd():
   import xframe
   try:
      click = xframe.interfaces.ClickDependency.click
      start_routine_click(click)
   except xframe.interfaces.DependencyImportError as e :
      log.info('Could not import click. Fallback to to argparse.')
      start_routine_argparse()
   
if __name__ == '__main__':
   variables = start_routine_cmd()
