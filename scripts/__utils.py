#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils functions to do many things

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8

Example:
    $ from __utils import *

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch


##### Check functions

def check_inside_circle(point, pos_circle=[0.0, 0.0], r_circle=1):
        
        if (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2 <=r_circle**2:
            return True
        return False

def check_inside_sphere(point, pos_circle=[0.0, 0.0, 0.0], r_circle=1):
        
        if (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2 + (pos_circle[2] - point[2])**2 <=r_circle**2:
            return True
        return False


def check_inside_ellipsoid(point, pos_circle=[0.0, 0.0, 0.0], r_circle=[1.,1.,1.]):
        
        if ((pos_circle[0] - point[0])**2)/r_circle[0]**2 + ((pos_circle[1] - point[1])**2)/r_circle[1]**2 + \
           ((pos_circle[2] - point[2])**2)/r_circle[2]**2 <=1:
            return True
        return False

def check_in_line_ab(point, line_a=1, line_b=0):
    if (line_a*point[0] + line_b == point[1]):
        return True
    return False
    
def check_in_line(point, line_p=np.array([0,0]), line_v=np.array([0,1])):
    p = np.array([line_p[1]-point[1],-line_p[0]+point[0]])
    if line_v.dot(p) == 0:
        return True
    return False

def line_segment_intersect_block(start, end, block_min, block_max):
    # Check if the start or end points of the line segment are inside the block
    if point_inside_block(start, block_min, block_max) or point_inside_block(end, block_min, block_max):
        return True
    
    # Check if the line segment intersects any of the six faces of the block
    faces = [
        (block_min, np.array([block_max[0], block_min[1], block_min[2]]), np.array([block_max[0], block_max[1], block_min[2]]), np.array([block_min[0], block_max[1], block_min[2]])),
        (block_min, np.array([block_min[0], block_min[1], block_max[2]]), np.array([block_max[0], block_min[1], block_max[2]]), np.array([block_max[0], block_max[1], block_max[2]])),
        (np.array([block_max[0], block_min[1], block_min[2]]), block_max, np.array([block_max[0], block_max[1], block_max[2]]), np.array([block_max[0], block_min[1], block_max[2]])),
        (block_min, np.array([block_min[0], block_min[1], block_max[2]]), np.array([block_min[0], block_max[1], block_max[2]]), np.array([block_min[0], block_max[1], block_min[2]])),
        (np.array([block_min[0], block_max[1], block_min[2]]), np.array([block_min[0], block_max[1], block_max[2]]), block_max, np.array([block_max[0], block_max[1], block_min[2]])),
        (np.array([block_max[0], block_min[1], block_min[2]]), np.array([block_max[0], block_max[1], block_min[2]]), np.array([block_max[0], block_max[1], block_max[2]]), block_max)
    ]
    
    for face in faces:
        if line_intersect_polygon(start, end, face):
            return True
    
    return False

def point_inside_block(point, block_min, block_max):
    return np.all(block_min <= point) and np.all(point <= block_max)


def check_collision_cylinder2line(curve, obs_cylinders, r, debug=0, tol=1):
    """ Check if a segment line collides with an infinite elliptical cylinder.
    curve: points [x,y,z] of a curve
    obs_cylinders: list of infinite elliptical cylinders [([x,y], radius_x, radius_y), ...]
    r: robot's radius    
    """
    curve_diff = np.diff(curve.T).T
    d = np.zeros(0)
    r = r/tol
    for ob in obs_cylinders:
        o = np.array([ob[0], ob[1], 0.0])
        li = curve[:-1] - o
        norm_pfi = np.linalg.norm(curve_diff, axis=1)
        
        vi = -np.sum(li * curve_diff, axis=1) / norm_pfi
        vi = vi[:, np.newaxis] * curve_diff

        pci = li + vi
        # closest point inside cylinder
        di = ((1 / (r + ob[2]))**2) * pci[:, 0]**2 + ((1 / (r + ob[3]))**2) * pci[:, 1]**2
        d = np.concatenate((d, di))

        # initial point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] - ob[1])**2
        d = np.concatenate((d, di))

        # end point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] + curve_diff[:, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] + curve_diff[:, 1] - ob[1])**2
        d = np.concatenate((d, di))

    d = d[d**0.5 <= 1]

    return d

def check_collision_cylinder2line2(curve, obs_cylinders,r):
    """ Check if a segment line collides with a infinite elliptical cylinder 
    curve: points [x,y,z] of a curve
    obs_cylinders: list of inifite elliptical cylinder [([x,y], radius x, radius,y), ...]
    r: robot's radius    
    """
    d = []
    curve_diff = np.diff(curve.T).T    
    
    for ob in obs_cylinders:
        o = np.array([ob[0], ob[1], 0.0])

        li = curve[:-1] - o
        norm_pfi = np.linalg.norm(curve_diff, axis=1)  

        # print(f"li = {li.shape}, diff = {curve_diff.shape}")    
        
        vi = -np.sum(li * curve_diff, axis=1) / norm_pfi
        vi = vi[:, np.newaxis] * curve_diff   
        
        pci = li + vi #+ o

        # closest point inside cylinder
        di = ((1 / (r + ob[2]))**2) * pci[:, 0]**2 + ((1 / (r + ob[3]))**2) * pci[:, 1]**2
        d.extend(di[di <= 1])        
        # initial point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] - ob[1])**2
        d.extend(di[di <= 1])
        # end point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] + curve_diff[:, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] + curve_diff[:, 1] - ob[1])**2
        d.extend(di[di <= 1])

    d = np.array(d)
    return d
        
def line_intersect_polygon(start, end, polygon):
    # Check if the line intersects the plane of the polygon
    print(polygon)
    plane_normal = calculate_plane_normal(*polygon[1:])
    line_direction = end - start
    t = np.dot(plane_normal, polygon[0] - start) / np.dot(plane_normal, line_direction)
    
    if not (0 <= t <= 1):
        return False
    
    intersection_point = start + t * line_direction
    
    # Check if the intersection point is inside the polygon
    return point_inside_polygon(intersection_point, polygon)

def calculate_plane_normal(p1, p2, p3):
    vector1 = p2 - p1
    vector2 = p3 - p1
    
    normal = np.cross(vector1, vector2)
    
    return normal

def point_inside_polygon(point, polygon):
    n = len(polygon)
    inside = False
    
    p1 = polygon[0]
    
    for i in range(n + 1):
        p2 = polygon[i % n]
        
        if (point[1] > min(p1[1], p2[1])) and (point[1] <= max(p1[1], p2[1])) and (point[0] <= max(p1[0], p2[0])):
            if p1[1] != p2[1]:
                x_intersection = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                
                if p1[0] == p2[0] or point[0] <= x_intersection:
                    inside = not inside
        
        p1 = p2
        
    return inside

check_block_collision = lambda segments, blocks: [line_segment_intersect_block(seg[0], seg[1], blk[0], blk[1]) for seg in segments for blk in blocks]



 ### CONVERTIONS

def convertAnglestoVector3d(yaw, beta=0):
    """ Convert Angles (yaw, beta) to a vector 3d [x,y,z]"""
    x = np.cos(yaw) * np.cos(beta)
    y = np.sin(yaw) * np.cos(beta)
    z = np.sin(beta)
    return np.array([x, y, z])

#### Random functions


def randomColor(n=1, use_list_colors=True):
    '''Random a color to use in matplotlib
    n = number of colors returned 
    '''
    if (use_list_colors) and (n <= 25):
         list_colors = ['blue', 'indigo', 'green', 'sienna', 'magenta', 'darkkhaki', 'cyan', 'olive', 'chocolate', 'silver', 'red', 'yellow', 'orange', 'yellowgreen', 'teal', 'violet', 'indigo', 'cornflowerblue', 'darkkhaki', 'dimgray']
         colors = list_colors[0:n]
    else:
        colors = []
        
        for i in range(n):
            colors.append("#"+''.join(np.random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']) for i in range(6)))   
        
    if n==1: return colors[0]
    return colors

### PLOTS

def plot_ellipsoid(center=(0,0,0), dimensions=(0.1,0.1,0.1), ax=None, Npoints=100, color='red', alpha=0.2):

    u = np.linspace(0.0, 2.0 * np.pi, Npoints)
    v = np.linspace(0.0, np.pi, Npoints)
    z = center[2] + dimensions[2]*np.outer(np.cos(u), np.sin(v))
    y = center[1] + dimensions[1]*np.outer(np.sin(u), np.sin(v))
    x = center[0] + dimensions[0]*np.outer(np.ones_like(u), np.cos(v))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)
    
    return ax

def draw_disc(p=np.array([0, 0]), r=1, ax=None, color='blue', fill=True, alpha=0.5):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    circle = plt.Circle((p[0], p[1]),radius=r, alpha=alpha, facecolor=color, edgecolor='none')    
    ax.add_artist(circle)
    return ax

def draw_disc3D(p=np.array([0, 0, 0]), r=1, zdir="z", ax=None, color='blue', fill=True, alpha=0.5):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    p1 = Circle((p[0], p[1]), r, facecolor=color, edgecolor='none',alpha=alpha)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=p[2], zdir=zdir)
    return ax

def plot_cylinder(center=[0,0,0], rx=1, ry=1, height=1, min_height=0, ax=None, color_surface = 'green', alpha=0.5, color_wireframe='green', line_wireframe=0.5, orientation='z',zorder=1):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(min_height, height, 50)
    U, V = np.meshgrid(u, v)

    if orientation == 'x':
        x = V + center[2]
        y = ry * np.sin(U) + center[1]
        z = rx * np.cos(U) + center[0]
    elif orientation == 'y':
        x = rx * np.cos(U) + center[0]
        y = V + center[2]
        z = ry * np.sin(U) + center[1]
    elif orientation == 'z':
        x = rx * np.cos(U) + center[0]
        y = ry * np.sin(U) + center[1]
        z = V + center[2]
    # x = rx * np.cos(U) * orientation[0] + center[0]
    # y = ry * np.sin(U) * orientation[1] + center[1]
    # z = V + center[2]

    ax.plot_surface(x, y, z, alpha=alpha, color=color_surface, zorder=zorder)
    ax.plot_wireframe(x, y, z, color=color_wireframe, linewidth=line_wireframe, alpha=alpha/2,rstride=4, cstride=4, zorder=zorder)

    return ax

def plot_line3D(p_start=[0,0,0], p_end=[0,1,0], color='red', alpha=1):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]], color=color, alpha=alpha)
    return ax

def plot_ellipse(center, rx, ry, angle=0, ax=None, Npoints=100, color='blue', alpha=1.0):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    t = np.linspace(0, 2*np.pi, Npoints)
    x = center[0] + (rx) * np.cos(t) * np.cos(angle) - (ry) * np.sin(t) * np.sin(angle)
    y = center[1] + (rx) * np.cos(t) * np.sin(angle) + (ry) * np.sin(t) * np.cos(angle)

    ax.fill(x, y, color=color, alpha=alpha)