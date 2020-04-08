import numpy as np
import pandas as pd

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
        self.array_type = type(array_like)
    
    def convert_array_type(self, array_like, new_type):
        """Converts an array to a given type.

        The funtion converts an array-like object to either a pandas Series or
        an numpy ndarray. If neither of those, it defaults to creating a list.

        Parameters
        ----------
        array_like : array_like
            1d numerical array to be converted to given type.
        new_type : type
            Given type to convert the data to.
        
        Returns
        -------
        array_like : array_like
            1d numerical array after type conversion.
        
        """

        if new_type in [list, pd.Series]:
            array_like = new_type(array_like)
        
        elif new_type == np.ndarray:
            array_like = np.array(array_like)
        
        return array_like

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


class MinMaxScaler(Scaler):
    """Class to scale numerical values using the min max method.

    Parameters
    ----------
    array_like : array_like
        1d array of numerical variables to be scaled.
    
    Attributes
    ----------
    array_like : array_like
        1d array of numerical variables to be scaled.
    min_max_val : tuple
        Tuple of the minimum and maximum values of the array_like.
    
    """

    def __init__(self, array_like):
        
        super().__init__(array_like=array_like)

        self.get_min_max_values()
    
    def min_max_scaling(self, x, min, max):
        """Numerical computation for min_max scaling a value.

        Parameters
        ----------
        x : float
            Value to be scaled.
        min : float
            Minimum value of the 1d array that x is contained in.
        max : float
            Maximum value of the 1d array that x is contained in.
        
        Returns
        -------
        x : float
            Scaled value for the given input.
        
        """

        if x is np.NaN:
            raise ValueError("x value is np.NaN")
        
        if max-min == 0:
            return 0

        x = (x - min) / (max - min)

        return x
    
    def scale_array(self):
        """Scale the array_like that the object was initialised with.

        This method scales the array_like and saves the new array as an
        attribute. It also returns this array.

        Attributes
        ----------
        scaled_array : array_like
            1d array of scaled numerical values.
        
        Returns
        -------
        scaled_array : array_like
            1d array of scaled numerical values.
        
        """

        # List comprehension, using the min_max_scaling function deifned above.
        self.scaled_array = [
            self.min_max_scaling(
                x=x,
                min=self.min_max_val[0],
                max=self.min_max_val[1],
            ) for x in self.array_like
        ]

        self.convert_array_type(self.scaled_array, self.array_type)

        return self.scaled_array
