#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import sys
from os import path as os_path
from io import open as io_open
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

__author__ = None
__licence__ = None
__version__ = None
main_file = os_path.join(os_path.dirname(__file__), '__init__.py')
for l in io_open(main_file, mode='r'):
    if any(l.startswith(i) for i in ('__author__', '__licence__', '__version__')):
        exec(l)

README_rst = ''
#fndoc = os_path.join(os_path.dirname(__file__), 'README.rst')
#with io_open(fndoc, mode='r', encoding='utf-8') as fd:
#    README_rst = fd.read()

setup(
    name='brainweb',
    version=__version__,
    description='20 multimodal BrainWeb phantoms',
    long_description=README_rst,
    license=__licence__.lstrip('[').split(']')[0],
    author=__author__.split('<')[0].strip(),
    author_email=__author__.split('<')[1][:-1],
    #url='https://github.com/casperdcl/brainweb',
    #bugtrack_url='https://github.com/casperdcl/brainweb/issues',
    platforms=['any'],
    packages=['brainweb'],
    #install_requires=['argparse'] if sys.version_info[:2] < (2, 7) else [],
    #package_data={'': ['LICENCE']},
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 3 - Alpha',
        'Framework :: IPython',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='brainweb medical imaging fdg pet mri',
    #test_suite='nose.collector',
    #tests_require=['nose', 'flake8', 'coverage'],
)
