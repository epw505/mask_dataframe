import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Masker(BaseEstimator, TransformerMixin):
    """Class to mask pandas dataframes to make data anonymous.

    The class transforms column names to a generic numbered column system. It
    also convert categorical levels to a generic numbered system. The mappings
    are saved in the Masker object to reference for interparability.

    """

    def __init__(self):

        super().__init__()

    def get_column_map(self, X):
        """Construct the dictionary map for masking column names.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to take columns from.

        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pd.DataFrame")

        new_columns = ["column_" + str(n) for n in range(0, X.shape[1])]

        self.column_map = dict(zip(X.columns, new_columns))

    def get_categorical_map(self, X):
        """Construct the dictionary map for masking categorical levels.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to take categorical values from.

        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pd.DataFrame")

        categorical_columns = list(X.select_dtypes(include=["object", "category"]))

        self.categorical_map = {}

        for col in categorical_columns:

            value_map = {}
            unique_vals = X[col].unique()

            for i, j in enumerate(unique_vals):

                value_map[j] = "level_" + str(i)

            self.categorical_map[col] = value_map

    def fit(self, X, y=None):
        """Fits the Masker class to the training data.

        This makes a call to get_categorical_map and get_column_map for a given
        dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to generate the maps from.
        y : None
            Not required, only there for scikit-learn functionality.

        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pd.DataFrame")

        self.get_categorical_map(X)
        self.get_column_map(X)

        return self

    def transform(self, X, y=None):
        """Masks the dataframe using the maps generated in the fit method.

        This can only be called once the transformer has been fit. Also, the
        dataframe you're transforming should match the data that the
        transformer was fit on.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to mask.
        y : None
            Not required, only there for scikit-learn functionality.

        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pd.DataFrame")

        X = X.copy()

        for col, col_map in self.categorical_map.items():

            X[col] = X[col].map(col_map)

        X.columns = self.column_map.values()

        return X
