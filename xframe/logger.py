import logging
import logging.handlers
import os
from xframe.settings import general

log_file_path = os.path.expanduser(general.log_file)
log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
existing_logs = {}

def setup_custom_logger(name,loglevelName):
    # logger settings
    if name in existing_logs:
        logger = existing_logs[name]
    else:
        log_file = log_file_path
        log_file_max_size = 1024 * 1024 * 20 # megabytes
        log_num_backups = 3
        log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
        log_date_format = "%m/%d/%Y %I:%M:%S %p"
        log_filemode = "w" # w: overwrite; a:     
        loglevel=getattr(logging,loglevelName.upper())

        # setup logger
        # datefmt=log_date_format
        logging.basicConfig(filename=log_file, format=log_format, filemode=log_filemode ,level=loglevel)
        rotate_file = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=log_file_max_size, backupCount=log_num_backups
        )
        logger = logging.getLogger(name)
        logger.propagate=False
        if len(logger.handlers)<3:
            logger.addHandler(rotate_file)
            #consoleHandler = logging.StreamHandler()
            # print log messages to console
            consoleHandler = logging.StreamHandler()
            logFormatter = logging.Formatter(log_format)
            consoleHandler.setFormatter(logFormatter)
            logger.addHandler(consoleHandler)
        logger.setLevel(loglevel)
        existing_logs[name]=logger        
    return logger

# source: https://docs.python.org/2/howto/logging.html
