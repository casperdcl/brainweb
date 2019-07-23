#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path as os_path
from io import open as io_open
from setuptools import setup

__version__ = None
main_file = os_path.join(os_path.dirname(__file__), 'brainweb', '__init__.py')
for l in io_open(main_file, mode='r'):
    if l.startswith('__version__'):
        __version__ = l.rsplit('=', 1)[1].strip().strip('"')

setup(version=__version__)
