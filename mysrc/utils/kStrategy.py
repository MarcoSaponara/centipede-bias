import numpy as np

class KStrategy():
    def __init__(self,
                 kernel : np.ndarray,
                ):
        
        self.kernel = kernel
        assert kernel.shape[0] == kernel.shape[1]
        assert np.allclose(kernel.sum(axis=1), 1.)
        
    def calculate_mixed_strategy(self, 
                                 k : int,
                                 start : np.ndarray,
                                ) -> np.ndarray:   
        l = self.kernel.shape[0]
        nb_actions = l//2
        assert k >= 0
        assert len(start) == l
        assert np.isclose(start[:nb_actions].sum(),1.)
        assert np.isclose(start[nb_actions:].sum(),1.)

        return start @ np.linalg.matrix_power(self.kernel, k)


#     def calculate_mixed_strategy_for_infinite_k(self,
#                                           k_test = 100,
#                                           tol = 1.0e-4,
#                                          ):
#         from scipy.linalg import eig
#         eigenvalues, eigenvectors = eig(self.kernel, left = True, right = False)
#         sd = abs(eigenvectors[:, np.argmax(eigenvalues.real)].real)
#         sd /= sd.sum()
        
#         test = np.linalg.matrix_power(self.kernel, k_test)[0,:]
#         assert np.isclose(np.abs(sd - test).sum(), tol)
        
#         return 2 * sd