![zorro logo](zorro_df_logo.png)
# Zorro DF
Zorro DF is a python package for masking pandas dataframe objects in order to 
anonymise data. It allows you to strip away identifiable column names and string 
values, replacing them with a generic naming convention. The package is built 
under the scikit-learn transformer framework and hence can be plugged into any 
scikit-learn Pipeline.

The package source-code can be found at http://github.com/epw505/zorro_df

## Getting Started
### Requirements
```
pandas>=0.25.3
scikit-learn>=0.22.1
```

### Installation
Zorro DF can be installed using `pip` with the following command:
```
pip install zorro_df
```

## Examples
Once the package is installed, you can load Zorro DF into your python session 
and use the Masker object to mask your data.
```
from zorro_df import mask_dataframe as mf

example_masker = mf.Masker()
example_masker.fit(data)
masked_data = example_masker.transform(data)
```

## Tests
The test suite for Zorro DF is built using `pytest` with the `pytest-mock` 
plugin. Install both as follows.
```
pip install pytest
pip install pytest-mock
```
Once they are installed, you can run the test suite from the root directory of
Zorro Df.
```
pytest tests/
```

## Future Development
* Reverse masking to allow retrieval of original data
* Scaling of numerical data to fully anonymise recognisable numeric values