from xframe.interfaces import ProjectWorkerInterface
from xframe.settings import project as opt
from xframe.database import project as db
import numpy as np

class ProjectWorker(ProjectWorkerInterface):
    def run(self):
        print(f'Hello {opt.name}, your random number is: {opt.random_number}')
        data = {'name':opt.name,'random_number':opt.random_number,'data':np.arange(10,dtype=float)}
        #path = '~/.xframe/data/hello_dat.h5'
        db.save('my_data',data)
        data2 = db.load('my_data')
        print(f'Loaded data = {data2}')
