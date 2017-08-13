from __future__ import print_function
#symbolic algebra
import theano
import theano.tensor as T
import numpy as np

# This code is copied from https://github.com/rmcgibbo/mullermsm 

def muller_potential(x, y):
    """Muller potential
    
    Parameters
    ----------
    x : {float, np.ndarray}
        X coordinate. Can be either a single number or an array. If you supply
        an array, x and y need to be the same shape.
    y : {float, np.ndarray}
        Y coordinate. Can be either a single number or an array. If you supply
        an array, x and y need to be the same shape.
    Returns
    -------
    potential : {float, np.ndarray}
        Potential energy. Will be the same shape as the inputs, x and y.
    
    Reference
    ---------
    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """
    
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    
    
    # use symbolic algebra if you supply symbolic quantities
    exp = T.exp if isinstance(x, T.TensorVariable) else np.exp
    
    value = 0
    for j in range(0, 4):
        value += AA[j] * exp(aa[j] * (x - XX[j])**2 + \
            bb[j] * (x - XX[j]) * (y - YY[j]) + cc[j] * (y - YY[j])**2)
    return value
    
    
def muller_force():
    """Compile a theano function to compute the negative grad
     of the muller potential"""
    sym_x, sym_y = T.scalar(), T.scalar()
    sym_V = muller_potential(sym_x, sym_y)
    sym_F =  T.grad(-sym_V, [sym_x, sym_y])
    
    # F takes two arguments, x,y and returns a 2
    # element python list
    F = theano.function([sym_x, sym_y], sym_F)
    
    def force(position):
        """force on the muller potential
        
        Parameters
        ----------
        position : list-like
            x,y in a tuple or list or array
        
        Returns
        -------
        force : np.ndarray
            force in x direction, y direction in a length 2 numpy 1D array
        """
        return np.array(F(*position))
        
    return force


def propagate(n_frames, initial_x, kT, dt, mGamma, force):
    """Propagate high-friction Langevin dynamics on
    the Muller potential
    
    Parameters
    ----------
    n_steps : int
        Number of steps to take
    initial_x : tuple
        x,y coordinates to start at
    kT : float
        boltzmans constant times temperature
    mGamma : float
        mass of the particle ties the friction coefficient
    force : callable
        Callable (e.g. function which takes a length n_dimensions 1D
        array as a single argument and return a length n_dimensions 1D
        array of force in each direction
        
    Reference
    ---------
    .. [1] http://gold.cchem.berkeley.edu/Pubs/DC150.pdf
    """
    
    n_dims = len(initial_x)
    # check to make sure we can really call it
    if not len(force(initial_x)) == n_dims:
        raise ValueError('force not returning the right stuff')
    
   
    # the variance is 2*kT*dt/m*gamma
    random = np.random.normal(scale=np.sqrt((2.0 * kT * dt) / (mGamma)),
        size=(n_frames - 1, 2))
    position = np.zeros((n_frames, 2))
    position[0] = initial_x
    
    for i in range(n_frames - 1):
        position[i+1] = position[i] + dt / (mGamma) * force(position[i]) + random[i]
    
    return position

