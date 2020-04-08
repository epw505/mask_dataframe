from zorro_df.mask_dataframe import Masker
from zorro_df.numerical_scalers import MinMaxScaler
import pytest
import builtins
import pandas as pd
import numpy as np


@pytest.fixture
def data():
    data = pd.DataFrame(
        {
            "col1": ["one", "two", "three"],
            "col2": [1, 2, 3],
            "col3": ["1", "2", "2"],
            "col4": [np.NaN, "blue", np.NaN],
        }
    )

    yield data


class TestInit(object):
    """Tests for the __init__ function for the Masker class."""

    def test_numerical_scaling_type(self):
        """Test error is thrown if numerical_scaling is not the correct type."""

        with pytest.raises(TypeError):
            test_masker = Masker(numerical_scaling=123)

    def test_numerical_scaling_attribute_created(self):
        """Tests the Masker object is created with the numerical_scaling attribute."""

        test_masker = Masker()

        assert hasattr(test_masker , "numerical_scaling")
    
    def test_numerical_scaling_default(self):
        """Test numerical_scaling argument defaults correctly."""

        test_masker = Masker()

        assert test_masker.numerical_scaling
    
    def test_scaling_method_type(self):
        """Test error is thrown if scaling_method is not the correct type."""

        with pytest.raises(TypeError):
            test_masker = Masker(scaling_method=123)
    
    def test_scaling_method_value(self):
        """Test error is thrown if scaling_method is not the correct value."""

        with pytest.raises(ValueError):
            test_masker = Masker(scaling_method="dummy_method")
    
    def test_scaling_method_attribute_created(self):
        """Test the Masker object is created with the scaling_method attribute."""

        test_masker = Masker()

        assert hasattr(test_masker, "scaling_method")
    
    def test_scaling_method_default(self):
        """Test scaling_method argument defaults correctly."""

        test_masker = Masker()

        assert test_masker.scaling_method == "MinMaxScaler"


class TestGetNumericalMap(object):
    """Tests for the get_numerical_map method for the Masker class."""

    def test_X_type(self):
        """Test error is thrown if X is not a pd.DataFrame."""

        test_masker = Masker()

        with pytest.raises(TypeError):
            test_masker.get_numerical_map(X=123)
    
    def test_numerical_map_attribute_created(self, data):
        """Tests that the numerical_map attribute is created for the Masker object"""

        test_masker = Masker()
        test_masker.get_numerical_map(X=data)

        assert hasattr(test_masker, "numerical_map")
    
    def test_numerical_map_values(self, data):
        """Test map values for MinMaxScaler."""

        test_masker = Masker()
        test_masker.get_numerical_map(X=data)

        assert list(test_masker.numerical_map.keys()) == ["col2"]
        assert isinstance(test_masker.numerical_map["col2"], MinMaxScaler)


class TestGetColumnMap(object):
    """Tests the get_column_map method for the Masker class."""

    def test_X_type(self):
        """Test error is thrown if X is not a pd.DataFrame."""

        test_masker = Masker()

        with pytest.raises(TypeError):
            test_masker.get_column_map(X=123)

    def test_column_map_attribute_created(self, data):
        """Test that the column_map attribute is created for the Masker object."""

        test_masker = Masker()
        test_masker.get_column_map(X=data)

        assert hasattr(test_masker, "column_map")

    def test_column_map_attribute_value(self, data):
        """Test that the column_map attribute takes the expect values."""

        test_masker = Masker()
        test_masker.get_column_map(X=data)

        expected_map = {
            "col1": "column_0",
            "col2": "column_1",
            "col3": "column_2",
            "col4": "column_3",
        }

        assert test_masker.column_map == expected_map


class TestGetCategoricalMap(object):
    """Tests for the get_categorical_map method of the Masker class."""

    def test_X_type(self):
        """Test error is thrown if X is not the correct type."""

        test_masker = Masker()

        with pytest.raises(TypeError):
            test_masker.get_categorical_map(X=123)

    def test_categorical_map_attribute_created(self, data):
        """Test that the categorical_map attribute is created for the Masker object."""

        test_masker = Masker()
        test_masker.get_categorical_map(X=data)

        assert hasattr(test_masker, "categorical_map")

    def test_categorical_map_value(self, data):
        """Test that the categorical_map attribute takes the correct value."""

        test_masker = Masker()
        test_masker.get_categorical_map(X=data)

        expected_map = {
            "col1": {"one": "level_0", "two": "level_1", "three": "level_2"},
            "col3": {"1": "level_0", "2": "level_1"},
            "col4": {np.NaN: "level_0", "blue": "level_1"},
        }

        assert test_masker.categorical_map == expected_map


class TestFit(object):
    """Test the fit method of the Masker class."""

    def test_X_type(self):
        """Test error is thrown if X is not the correct type."""

        test_masker = Masker()

        with pytest.raises(TypeError):
            test_masker.fit(X=123)

    def test_get_categorical_map_called(self, data, mocker):
        """Test that the get_categorical_map method is called."""

        mocker.patch.object(Masker, "get_categorical_map")

        test_masker = Masker()
        test_masker.fit(X=data)

        Masker.get_categorical_map.assert_called()
    
    def test_get_numerical_map_called(self, data, mocker):
        """Test that the get_numerical_map method is called."""

        mocker.patch.object(Masker, "get_numerical_map")

        test_masker = Masker()
        test_masker.fit(X=data)

        Masker.get_numerical_map.assert_called()

    def test_get_column_map_called(self, data, mocker):
        """Test that the get_column_map method is called."""

        mocker.patch.object(Masker, "get_column_map")

        test_masker = Masker()
        test_masker.fit(X=data)

        Masker.get_column_map.assert_called()


class TestTransform(object):
    """Tests for the transform method of the Masker class."""

    def test_X_type(self, data):
        """Test error is thrown if X is not the correct type."""

        test_masker = Masker()
        test_masker.fit(data)

        with pytest.raises(TypeError):
            test_masker.transform(X=123)

    def test_values_single_column(self):
        """Test column values mapped correctly for a single column."""

        dummy_data = pd.DataFrame({"col1": ["A", "B", "A"]})
        test_masker = Masker()
        dummy_data = test_masker.fit_transform(dummy_data)

        assert all(dummy_data.iloc[:, 0] == ["level_0", "level_1", "level_0"])

    def test_values_multiple_columns(self):
        """Test column values maped correctly for multple columns."""

        dummy_data = pd.DataFrame({"col1": ["A", "B", "C"], "col2": ["1", "1", "1"]})

        test_masker = Masker()
        dummy_data = test_masker.fit_transform(dummy_data)

        assert all(dummy_data.iloc[:, 0] == ["level_0", "level_1", "level_2"])

        assert all(dummy_data.iloc[:, 1] == ["level_0", "level_0", "level_0"])

    def test_columns_single_column(self):
        """Test column names mapped correctly for a single column."""

        dummy_data = pd.DataFrame({"col1": [1, 2, 3]})
        test_masker = Masker()
        dummy_data = test_masker.fit_transform(dummy_data)

        assert dummy_data.columns[0] == "column_0"

    def test_columns_multiple_columns(self):
        """Test column names mapped correctly for multiple columns."""

        dummy_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 3, 4]})
        test_masker = Masker()
        dummy_data = test_masker.fit_transform(dummy_data)

        assert all(dummy_data.columns == ["column_0", "column_1"])

    def test_dataframe_matches_no_scaling(self):
        """Full test that the dataframe is as expected, without numeric scaling."""

        dummy_data = pd.DataFrame(
            {"col1": ["1", "2", "1"], "col2": [1, 2, 3], "col3": ["red", "red", "blue"]}
        )

        expected_data = pd.DataFrame(
            {
                "column_0": ["level_0", "level_1", "level_0"],
                "column_1": [1, 2, 3],
                "column_2": ["level_0", "level_0", "level_1"],
            }
        )

        test_masker = Masker(numerical_scaling=False)
        dummy_data = test_masker.fit_transform(dummy_data)

        assert dummy_data.equals(expected_data)
    
    def test_dataframe_matches_with_scaling(self):
        """Full test that the dataframe is as expected, with numeric scaling."""

        dummy_data = pd.DataFrame(
            {"col1": ["1", "2", "1"], "col2": [1, 2, 3], "col3": ["red", "red", "blue"]}
        )

        expected_data = pd.DataFrame(
            {
                "column_0": ["level_0", "level_1", "level_0"],
                "column_1": [0.0, 0.5, 1.0],
                "column_2": ["level_0", "level_0", "level_1"],
            }
        )

        test_masker = Masker(numerical_scaling=True)
        dummy_data = test_masker.fit_transform(dummy_data)

        assert dummy_data.equals(expected_data)