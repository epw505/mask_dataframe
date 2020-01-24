from zorro_df.mask_dataframe import Masker
from zorro_df import numerical_scalers as scale
import pytest
import builtins
import pandas as pd
import numpy as np


class TestMinMaxValues(object):
    """Tests for the min_max_values function in numerical_scalers.py."""

    def test_array_like_type(self):
        """Test error is thrown if array_like is not the correct type."""

        with pytest.raises(TypeError):
            scale.min_max_values(array_like=123)
    
    def test_array_like_value_type(self):
        """Test error is thrown if array_like values are not the correct type."""

        with pytest.raises(TypeError):
            scale.min_max_values(array_like=[1, 2, "dummy"])
    
    def test_error_thrown_with_nan(self):
        """Test error is thrown if array_like contains missing values."""

        with pytest.raises(ValueError):
            scale.min_max_values(array_like=[1, 2, np.NaN])
    
    def test_output_type(self):
        """Test the output is the correct type."""

        x = [3, 2, 5, 4]
        min_max_x = scale.min_max_values(x)

        assert isinstance(min_max_x, tuple)  
    
    def test_output_values(self):
        """Test the output values are correct."""

        x = [3, 2, 5, 4]
        min_max_x = scale.min_max_values(x)

        assert min_max_x == (2, 5)
