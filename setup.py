#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib']

setup(
    name='time_rescale',
    version='0.2.2',
    description=('Tools for evaluating the goodness of fit of a point'
                 'process model via the time rescaling theorem'),
    author='Eric Denovellis',
    author_email='eric.denovellis@ucsf.edu',
    packages=find_packages(),
    url='https://github.com/Eden-Kramer-Lab/time_rescale',
    license='MIT',
    install_requires=INSTALL_REQUIRES,
)
