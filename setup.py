from setuptools import setup, find_packages

setup(
    name='rombo',
    version='0.1.0',
    packages=find_packages(include=['rombo*']),
    install_requires=["numpy>=1.21.6", "scipy>=1.7.3", "botorch>=0.8.0", "py-pde"],
)