#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.rst').read()

VERSION = '0.0.2'

requirements = [
    'numpy',
    'pillow',
    'six',
]

setup(
    # Metadata
    name='mxbox',
    version=VERSION,
    author='Lyken from TuSimple',
    author_email='lykensyu+github@gmail.com',
    url='https://github.com/lyken17/mxbox',
    description='Image and video datasets and models for mxnet deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
)
