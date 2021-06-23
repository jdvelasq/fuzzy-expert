from setuptools import setup, find_packages


setup(
    name="fuzzy_expert",
    version="0.1.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/fuzzy-expert",
    description="Fuzzy Expert System in Python",
    long_description="Fuzzy Expert System",
    keywords="fuzzy",
    platforms="any",
    provides=["fuzzy_expert"],
    install_requires=[
        "numpy",
        "matplotlib",
        "progressbar2",
        "pandas",
        "ipywidgets",
    ],
    packages=find_packages(),
    package_dir={"fuzzy_expert": "fuzzy_expert"},
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
