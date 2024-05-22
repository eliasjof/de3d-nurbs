#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEBUG usage

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8


Usage:
1. Set the global variable DEBUG to enable the debug functions

Example:

from __debug import *
import __debug 
__debug.DEBUG = False # True or False
printD('OK')

"""

######################### DEBUG
DEBUG = False
def printD(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
####################

