from xframe.analysis.analysisWorker import AnalysisWorker
from xframe import Multiprocessing
import numpy as np

def multiply_matrix_with_vectors(vects,matrix,**kwargs):
    vects= np.atleast_2d(vects)
    new_vects = np.sum(matrix[None,:,:]*vects[:,None,:],axis=2)
    return np.squeeze(new_vects)
	
class Worker(AnalysisWorker):
    def start_working(self):
        vectors = np.random.rand(200,10)
        matrix = np.random.rand(10,10)
        result = Multiprocessing.process_mp_request(multiply_matrix_with_vectors,input_arrays=[vectors],const_inputs = [matrix])
        test_result = multiply_matrix_with_vectors(vectors,matrix)
        if (result == test_result).all():
            print('Test passed!')

