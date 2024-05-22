#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code provides x,y,z for a topography

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8


Usage:
1. Run the planner to obtain the NURBS parameters' curve

Example: 
   from topography import *

"""

import numpy as np

from __debug import *
import __debug 

# Topographies parameters
topographies = dict(
    {1: dict({ 'list_peaks': np.array([ (0,0) ]), 'list_decrement': np.array([(.2,.5)]), 'list_amplitude': np.array([.2])}),
     2: dict({ 'list_peaks': np.array([ (0,-0.25), (0., 0.25) ]), 'list_decrement': np.array([(.25,.2), (.2,.2)]), 'list_amplitude': np.array([0.2, 0.3])}),
     3: dict({ 'list_peaks': np.array([ (0,-0.25), (0., 0.25), (0.25,0.) ]), 'list_decrement': np.array([(.18,.22), (.11,.11), (.1,.1)]), 'list_amplitude': np.array([0.15, 0.1, 0.1])}),
     4: dict({ 'list_peaks': np.array([ (0,-0.25), (0., 0.25), (0.25,0.), (0.4,0.4) ]), 'list_decrement': np.array([(.18,.22), (.11,.11), (.1,.1), (.1,.1)]), 'list_amplitude': np.array([0.15, 0.2, 0.15, 0.05])}),
     5: dict({ 'list_peaks': np.array([ (0,-0.25), (0., 0.25), (0.25,0.), (0.4,0.4), (0.4,-0.4) ]), 'list_decrement': np.array([(.18,.22), (.11,.11), (.1,.1), (.1,.1), (.1,.1)]), 'list_amplitude': np.array([0.15, 0.2, 0.15, 0.05, 0.05])}),
     6: dict({ 'list_peaks': np.array([ (0,-0.25), (0., 0.25), (0.25,0.), (0.4,0.4), (-0.4,0.4), (-0.4,-0.4) ]), 'list_decrement': np.array([(.18,.22), (.11,.11), (.1,.1), (.1,.1), (.1,.1), (.1,.1)]), 'list_amplitude': np.array([0.15, 0.25, 0.15, 0.05, 0.25, 0.05])}),
     7: dict({ 'list_peaks': np.array([ (-0.4,-0.2), (-0.3, 0.2), (0.0,0.0), (0.2,0.3), (0.4,0.2), (0.4,-0.3) ]), 'list_decrement': np.array([(.13,.13), (.13,.13), (.13,.13), (.13,.13), (.13,.13), (.13,.13)]), 'list_amplitude': 3*np.array([0.01, 0.02, 0.035, 0.02, 0.03, 0.03])}),
     8: dict({ 'list_peaks': np.array([ (-0.4,-0.2), (-0.3, 0.2), (0.0,0.0), (0.2,0.3), (0.4,0.2), (-0.4,0.3) ]), 'list_decrement': np.array([(.13,.13), (.13,.13), (.13,.13), (.13,.13), (.13,.13), (.13,.13)]), 'list_amplitude': 4*np.array([0.01, 0.02, 0.035, 0.02, 0.03, 0.03])}),


})

##############################################################
#### Topography
def get_topography(x, y, topography_index=1, scale=1):
    global topographies

    if topography_index == 0:
        return np.array([0])
    if topography_index > list(topographies.keys())[-1]:
        topography_index = list(topographies.keys())[-1]
        printD(f"Using topopgraphy index = {topography_index}")
    list_peaks = topographies[topography_index]['list_peaks']*scale
    list_decrement = topographies[topography_index]['list_decrement']*scale
    list_amplitude = topographies[topography_index]['list_amplitude']*scale

    z = np.zeros_like(x, dtype=float)    
    for Hi, xi_yi, xsi_ysi in zip(list_amplitude, list_peaks, list_decrement):
        z += Hi*np.exp(-(((x - xi_yi[0])/xsi_ysi[0]) ** 2 + ((y - xi_yi[1])/xsi_ysi[1]) ** 2))
    
    return z

def generate_topography(N_points=2500, scale=1,  topography_index=1):
    # Generate mesh grid
    x = np.linspace(-0.5*scale, 0.5*scale, N_points)
    y = np.linspace(-0.5*scale, 0.5*scale, N_points)

    x, y = np.meshgrid(x, y)

    # Generate topography
    z = get_topography(x, y,  topography_index=topography_index, scale=scale)

    return x, y, z