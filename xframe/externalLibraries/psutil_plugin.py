from xframe.settings import general
import logging
log = logging.getLogger('root')

try:
    import psutil
    class PsutilPlugin:
        @staticmethod
        def get_free_memory():
            return psutil.virtual_memory().available
except Exception as e:
    log.warning('psutils not installed. Use RAM value specified in settings.general instead.')
    class PsutilPlugin:
        @staticmethod
        def get_free_memory():
            return general.RAM
