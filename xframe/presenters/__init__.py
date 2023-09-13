import importlib
import traceback
import logging
log = logging.getLogger('root')

modules_to_import  = ['matplotlibPresenter','openCVPresenter']
for module in modules_to_import:
    try:
        if isinstance(module,(list,tuple)):
            module_name = 'xframe.presenters.'+module[0]
            globals()[module[1]] = importlib.import_module(module_name)
        else:
            splitted_name = module.split('.')
            module_name = 'xframe.presenters.'+module
            if len(splitted_name) > 1:                
                globals()[splitted_name[-1]] = importlib.import_module(module_name)
            else:
                globals()[module] = importlib.import_module(module_name)
                
    except Exception as e:
        traceback.print_exc()
        log.error('Caught exeption while loading {} with content: {}'.format(module_name,e))
