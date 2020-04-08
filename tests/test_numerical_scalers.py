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


class TestMinMaxScaler(object):
    """Tests for the MinMaxScaler class in numerical_scalers.py"""

    def test_super_init_called(self, mocker):
        """Test the super init is called when the object is initialised."""

        mocked_method = mocker.spy(scale.Scaler, "__init__")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        mocked_method.assert_called()
        assert min_max_scaler.array_like == [1, 2, 3]
    
    def test_get_min_max_values_called(self, mocker):
        """Test the get_min_max_values method from parent class is called."""

        mocked_method = mocker.spy(scale.Scaler, "get_min_max_values")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        mocked_method.assert_called()
        assert min_max_scaler.min_max_val == (1, 3)
    
    def test_min_max_scaling_nan_value(self):
        """Test error is thrown if value is np.NaN."""

        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        with pytest.raises(ValueError):
            min_max_scaler.min_max_scaling(x=np.NaN, min=0, max=1)

    def test_min_max_scaling_output_zeros(self):
        """Test min_max_scaling method with zeroes."""

        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        scaled_val = min_max_scaler.min_max_scaling(0, 0, 0)

        assert scaled_val == 0

    def test_min_max_scaling_output_positive(self):
        """Test min_max_scaling method with positive values."""

        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        scaled_val = min_max_scaler.min_max_scaling(2, 0, 4)

        assert scaled_val == 0.5
    
    def test_min_max_scaling_output_negative(self):
        """Test min_max_scaling method with negative values."""

        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])

        scaled_val = min_max_scaler.min_max_scaling(-2, -9, -1)

        assert scaled_val == 0.875
    
    def test_scale_array_min_max_called_times(self, mocker):
        """Test the number of times min_max_scaling is called in scale_array."""

        mocked_method = mocker.spy(scale.MinMaxScaler, "min_max_scaling")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])
        min_max_scaler.scale_array()

        assert mocked_method.call_count == 3
    
    def test_scale_array_min_max_call_args(self, mocker):
        """Test the min_max_scaling method is called with the correct arguments."""

        mocked_method = mocker.spy(scale.MinMaxScaler, "min_max_scaling")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])
        min_max_scaler.scale_array()

        assert mocked_method.call_args_list[0][1] == {"max": 3, "min": 1, "x": 1}
        assert mocked_method.call_args_list[1][1] == {"max": 3, "min": 1, "x": 2}
        assert mocked_method.call_args_list[2][1] == {"max": 3, "min": 1, "x": 3}
    
    def test_scale_array_convert_array_type_called(self, mocker):
        """Test the convert_array_type method is called during scale_array."""

        mocked_method = mocker.spy(scale.Scaler, "convert_array_type")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])
        min_max_scaler.scale_array()

        
        mocked_method.assert_called()
    
    def test_scale_array_convert_array_type_arguments(self, mocker):
        """Test the convert_array_type method is called with the correct arguments."""

        mocked_method = mocker.spy(scale.Scaler, "convert_array_type")
        min_max_scaler = scale.MinMaxScaler(array_like=[1, 2, 3])
        min_max_scaler.scale_array()

        assert mocked_method.call_args_list[0][0][1]== [0.0, 0.5, 1.0]
        assert mocked_method.call_args_list[0][0][2]== list
