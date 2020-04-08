from zorro_df.mask_dataframe import Masker
from zorro_df import numerical_scalers as scale
import pytest
from pytest_mock import mocker
import builtins
import pandas as pd
import numpy as np


class TestScaler(object):
    """Tests for the Scaler class in numerical_scalers.py."""

    def test_array_like_type(self):
        """Test error is thrown if array_like is not the correct type."""

        with pytest.raises(TypeError):
            scale.Scaler(array_like=123)
    
    def test_array_like_value_type(self):
        """Test error is thrown if array_like values are not the correct type."""

        with pytest.raises(TypeError):
            scale.Scaler(array_like=[1, 2, "dummy"])
    
    def test_error_thrown_with_nan(self):
        """Test error is thrown if array_like contains missing values."""

        with pytest.raises(ValueError):
            scale.Scaler(array_like=[1, 2, np.NaN])
    
    def test_has_attr_array_like(self):
        """Test initialised Scaler object has array_like attribute."""

        test_scaler = scale.Scaler([1, 2, 3])

        assert hasattr(test_scaler, "array_like")
    
    def test_array_like_value(self):
        """Test array_like attribute assigned correctly."""

        test_scaler = scale.Scaler([1, 2, 3])

        assert test_scaler.array_like == [1, 2, 3]
    
    def test_min_max_val_attribute(self):
        """Test min_max_val attribute is assigned."""

        test_scaler = scale.Scaler([3, 2, 5, 4])
        test_scaler.get_min_max_values()

        assert hasattr(test_scaler, "min_max_val")
    
    def test_min_max_values_1(self):
        """Test the min_max values are correct."""

        test_scaler = scale.Scaler([1, 2, 3, 4])
        test_scaler.get_min_max_values()

        assert test_scaler.min_max_val == (1, 4)
    
    def test_min_max_values_2(self):
        """Test the min_max values are correct, with negatives."""

        test_scaler = scale.Scaler([-2, -5, 10, 3])
        test_scaler.get_min_max_values()

        assert test_scaler.min_max_val == (-5, 10)
    
    def test_min_max_values_3(self):
        """Test the min_max values are correct, with length 1."""

        test_scaler = scale.Scaler([3])
        test_scaler.get_min_max_values()

        assert test_scaler.min_max_val == (3, 3)
    
    def test_convert_array_type_list_to_pd_series(self):
        """Test convert_array_type converts a list to a pd.Series"""

        test_scaler = scale.Scaler([1, 2, 3])
        output = test_scaler.convert_array_type([1, 2, 3], pd.Series)

        assert isinstance(output, pd.Series)
    
    def test_convert_array_type_pd_Series_to_list(self):
        """Test convert_array_type converts a pd.Series to a list."""

        test_scaler = scale.Scaler([1, 2, 3])
        output = test_scaler.convert_array_type(pd.Series([1, 2, 3]), list)

        assert isinstance(output, list)
    
    def test_convert_array_type_list_to_np_array(self):
        """Test convert_array_type converts a list to a np.ndarray."""

        test_scaler = scale.Scaler([1, 2, 3])
        output = test_scaler.convert_array_type([1, 2, 3], np.ndarray)
        
        assert isinstance(output, np.ndarray)
