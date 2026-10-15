from setuptools import setup, find_packages

setup(
    name='passive_sensing',
    version='0.0.1',
    description='Passive UWB sensing',
    author='pirkl',
    packages=find_packages(),
    install_requires=[
        "torch",
        "lightning",
        "pandas",
        "prettytable",
        "tensorboardX",
        "h5py",
        "matplotlib",
        "scipy",
    ]
)