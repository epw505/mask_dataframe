import numpy as np


class Scaler(object):
    """Scaler class containing key functionality for scaling.

    Specific scaling classes inherit from this parent class.

    Parameters
    ----------
    array_like : array_like
        1d array of numerical values to be scaled.

    Attributes
    ----------
    array_like : array_like

    """

    def __init__(self, array_like):

        if np.ndim(array_like) != 1:
            raise TypeError("array_like should be an array-like with np.ndim==1")

        for val in array_like:

            if np.isnan(val):
                raise ValueError("array_like contains np.NaN value")
            if type(val) not in [int, float]:
                raise TypeError("array_like should contain only numeric values")
        
        self.array_like = array_like

    def get_min_max_values(self):
        """Calculates the minimum and maximum of a numerical array-like object.
        
        The output is a tuple, with first and second values
        representing the min and max respectively.
        
        Attributes
        -------
        min_max_vals : tuple
            Tuple contating min and max values for the passed array-like.
        
        """

        min_val = min(self.array_like)
        max_val = max(self.array_like)

        self.min_max_val = (min_val, max_val)
