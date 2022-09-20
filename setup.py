#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'scipy', 'networkx']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='trajectory_analysis_tools',
    version='1.0.0',
    license='MIT',
    description=(''),
    author='Eric Denovellis',
    author_email='eric.denovellis@ucsf.edu',
    url='',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
