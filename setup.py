from setuptools import setup, find_packages

setup(
    name='aromad',
    version='0.1.0',
    packages=find_packages(include=['aromad*']),
    install_requires=["numpy>=1.21.6", "scipy>=1.7.3", "smt>=2.0.0", "botorch"],
)