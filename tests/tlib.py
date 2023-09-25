import shutil
import os

def copy(source,destination):
    shutil.copyfile(source,destination)

def save(path,data,**kwargs):
    with open(path,'wt') as txt_file:
        if isinstance(data,str):
            txt_file.write(data)
        elif isinstance(data,(list,tuple)):
            txt_file.writelines(data)
  
def load(path,**kwargs):
    with open(path,'rt') as txt_file:
        text = txt_file.readlines()
    return text

def create_path_if_nonexistent(path):
    dirname=os.path.dirname(path)
    if not (os.path.exists(dirname) or dirname==''):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass
