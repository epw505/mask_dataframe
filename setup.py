import setuptools

import version


with open("README.md", "r") as read_me:
    long_description = read_me.read()

setuptools.setup(
    name="zorro_df", 
    version=version.__version__,
    author="Ned Webster",
    author_email="edwardpwebster@gmail.com",
    description="Package to mask pd.DataFrame data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/epw505/zorro_df",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
