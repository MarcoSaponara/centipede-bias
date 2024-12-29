import numpy as np

from scipy.linalg import circulant
    
    
def brain_unbiased(nb_steps : int, 
                   gamma : float,
                   eps : float,
                  ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 =  circulant(np.array([1. - eps] + (n - 1) * [eps/(n - 1)])).T
   
    #nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    nbr_pl2 = np.vstack((np.ones(n, dtype=float)/n, nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    assert np.allclose(kernel.sum(axis=1), 1.)
    
    return kernel


def brain_proself(nb_steps : int, 
                  gamma : float,
                  eps : float,
                 ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 = (1 - eps) * np.eye(n, dtype = float)
    nbr_pl1[0,0] = 1.
    for i in range(1, n):
        nbr_pl1[i,:i] = eps / i

    #nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    nbr_pl2 = np.vstack((np.ones(n, dtype=float)/n, nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    assert np.allclose(kernel.sum(axis=1), 1.)
    
    return kernel


def brain_prosocial(nb_steps : int, 
                    gamma : float,
                    eps : float,
                   ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 = (1 - eps) * np.eye(n, dtype = float)
    for i in range(n-1):
        nbr_pl1[i,i+1:] = eps / (nb_steps // 2 - i) 
    nbr_pl1[-1,-1] = 1.
    
    #nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    nbr_pl2 = np.vstack((np.ones(n, dtype=float)/n, nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    assert np.allclose(kernel.sum(axis=1), 1.)
    
    return kernel