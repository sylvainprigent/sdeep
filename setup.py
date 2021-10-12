import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdeep",
    version="0.0.1",
    author="Sylvain Prigent",
    author_email="meriadec.prigent@gmail.com",
    description="Deep learning models using pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sylvainprigent/sdeep",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "pytorch>=1.9.1",
        "scikit-image>=1.7.1",
        "natsort>=7.1.1",
    ],
)
