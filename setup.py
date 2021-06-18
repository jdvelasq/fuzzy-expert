from setuptools import setup, find_packages


setup(
    name="fuzzy-toolbox",
    version="0.1.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/fuzzy_toolbox",
    description="Fuzzy Toolbox",
    long_description="Fuzzy Inference Systems Toolbox",
    keywords="fuzzy",
    platforms="any",
    provides=["fuzzy_toolbox"],
    install_requires=[
        "numpy",
        "matplotlib",
        "progressbar2",
        "pandas",
    ],
    packages=find_packages(),
    package_dir={"fuzzy_toolbox": "fuzzy_toolbox"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
