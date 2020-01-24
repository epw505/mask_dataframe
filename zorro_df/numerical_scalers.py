import numpy as np


def min_max_values(array_like):
    """Calculates the minimum and maximum of a numerical array-like object.
    
    The output is a tuple, with first and second values
    representing the min and max respectively.

    Parameters
    ----------
    array_like : array-like
        Numerical array.
    
    Returns
    -------
    min_max_vals : tuple
        Tuple contating min and max values for the passed array-like.
    
    """

    if np.ndim(array_like) != 1:
        raise TypeError("array_like should be an array-like with np.ndim==1")

    for val in array_like:

        if np.isnan(val):
            raise ValueError("array_like contains np.NaN value")
        if type(val) not in [int, float]:
            raise TypeError("array_like should contain only numeric values")
    
    min_val = min(array_like)
    max_val = max(array_like)

    return (min_val, max_val)
