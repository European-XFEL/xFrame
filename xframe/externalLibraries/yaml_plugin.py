from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap,CommentedSeq
from xframe.library.pythonLibrary import DictNamespace
from xframe.settings.tools import SettingsConverter
import numpy as np
import logging
import traceback

log=logging.getLogger('root')
yaml = YAML()

class YAML_access:
    @staticmethod
    def save(path, data:dict,convert=False,**kwargs):
        with open(path,'w') as _file:
            if convert:
                yaml.dump(SettingsConverter.convert(data),_file)
            else:
                yaml.dump(data,_file)
    @staticmethod
    def load(path,**kwargs):
        with open(path,'r') as _file:
            data = yaml.load(_file.read())
        if isinstance(data,type(None)):
            data = {}
        return data
    @staticmethod
    def format_settings_dict(settings):
        return SettingsConverter.convert(settings)

