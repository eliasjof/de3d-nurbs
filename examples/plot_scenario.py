#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example to plot the results

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8

Usage:
- Select the pickle file

"""
import sys
import os

# Add the scripts directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from de3dnurbs import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import pickle
# from robots_models import *
from __utils import *
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from geomdl import knotvector

print('plotting')
filepath_name = './results/'

seq = [1,2,3,4,5]
curves_points = []
ctrl_points = []
weights = []
knot = []
colors = randomColor(5)
markers = ['o', 's', '^', 'D', 'd', 'p', 'x', '.', '*']
list_waypoints2 = []
for seqi in seq:
    filedata = './results/paths/experiment_algorithm_LSHADE-COP_p396_7th_nurbs_Complex_'+ str(seqi) +'_S0.pkl'
    with open(filedata, 'rb') as f:  
        problem_params, algorithm_params, solution, time_solved, fitness, experiment_data,list_waypoints,obstacles_cylinders, \
            dict_parameters = pickle.load(f)
    plannerT = de3dnurbs(type='DE-NURBS', degree=problem_params['degree'], debug=0)      
    dt = problem_params['dt'] 
    scale = problem_params['scale']
    # scale = 1
    fx, fy, fz, curve_nurbs = plannerT.get_curve_points(solution, problem_params, dt=problem_params['dt'], scale=scale)    
    knot.append(curve_nurbs.knotvector)
    
    weights.append(curve_nurbs.weights)
    ctrl_points.append(curve_nurbs.ctrlpts)
    
    degree_plus1 = curve_nurbs.degree + 1
    curve_points_i = np.array([fx, fy, fz]).T
    print(seqi, len(fx), curve_points_i[0:2], curve_points_i[-1:-3:-1])
    list_waypoints2.append(list_waypoints)
    curves_points.append(curve_points_i)
waypoints = []
for w0, w1 in zip(list_waypoints2, list_waypoints2[1:]):
    w0 = w0[0]
    w1 = w1[0]
    waypoints.append([w0[0], w0[1], w0[2], w1[0], w1[1], w1[2]])    
waypoints#, list_waypoints

degree = problem_params['degree']
curve1_nurbs, curve1 = generate_curve(*join_curves(knot, ctrl_points, weights, smooth_beta=400, degree=degree), dt=1/(400*len(seq)), degree=degree)
penalties = check_penalties_problem_params(curve1, problem_params, tol=5/100)

##############
fig = plt.figure(figsize=(32,18))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

for iob, ob in enumerate(obstacles_cylinders):
    ob = np.array(ob)*scale
    if iob == 10:
         ax.text(ob[0]-3,ob[1]-1,r'$ob_{10}$', alpha=0.8)
    else:
        ax.text(ob[0]-3,ob[1]-1,rf'$ob_{iob}$', alpha=0.8)
    plot_cylinder([ob[0], ob[1],0], rx=ob[2], ry=ob[3], height=problem_params['space_limit'][0]*scale, ax=ax2, color_surface='green', \
                color_wireframe='green', line_wireframe=0.25, alpha=0.15)
for ob in obstacles_cylinders:
        ob = np.array(ob)*scale
        plot_ellipse([ob[0], ob[1]], rx=ob[2], ry=ob[3], ax=ax, color='green')


ax.plot(curve1.T[0], curve1.T[1], '-', color='k', linewidth=4, markersize=7, zorder=25, alpha=0.3, label=r'$C(s)$')
ax2.plot(curve1.T[0], curve1.T[1], curve1.T[2],'-', color='k',  linewidth=9, markersize=14, zorder=25, alpha=0.2)

colors = randomColor(5)
ii = 0
for curve_i, color in zip(curves_points, colors):
    ax.plot(curve_i.T[0], curve_i.T[1], '--', color=color, linewidth=2, markersize=14, label=fr'$C_{ii}(s)$')
    ax2.plot(curve_i.T[0], curve_i.T[1], curve_i.T[2],'--', color=color,  linewidth=2, markersize=14)
    ii+=1
     

## Topography
if problem_params['topography_index'] > 0:
    # Define custom colormap colors
    min_color = np.array([85,87,83])/255  # RGB values for minimum intensity (blue)
    max_color = np.array([193,125,17])/255  # RGB values for maximum intensity (red)
    # Create a custom colormap
    my_cmap = LinearSegmentedColormap.from_list('custom', [min_color, max_color], N=256)
    x, y, Z = generate_topography(N_points=2500, scale=scale*1.1,topography_index=problem_params['topography_index'])
    ax2.plot_surface(x, y, Z, cmap=my_cmap, alpha=0.27, zorder=1)#,edgecolors=min_color*2, zorder=0)    
    contour = ax.contour(x, y, Z, cmap=my_cmap, levels=75, zorder=0, offset=np.max(Z))
    # Label the contour lines with their values
    ax.clabel(contour, inline=True, fontsize=8)
###########

ax2.set_zlim(0,200)

for ii, waypoint in enumerate(waypoints):
    # print(waypoint)
    pinit = np.array(waypoint[0])
    pgoal = np.array(waypoint[3])
    
    yaw_init = ((waypoint[1] + 180) % 360 - 180)*np.pi/180.
    yaw_goal = ((waypoint[4] + 180) % 360 - 180)*np.pi/180.
    pitch_init = ((waypoint[2] + 180) % 360 - 180)*np.pi/180.
    pitch_goal = ((waypoint[5] + 180) % 360 - 180)*np.pi/180.
    
    print(ii, pinit, pitch_init, pgoal, pitch_goal)
    ax.plot(pinit[0], pinit[1], '*k', markersize=14)
    ax.plot(pgoal[0], pgoal[1], '*k', markersize=14)
    ax.text(pinit[0]-23, pinit[1]+5,rf'$W_{ii}$', fontsize=30)
    ax.quiver(pinit[0], pinit[1], np.cos(yaw_init), np.sin(yaw_init), scale_units='xy', scale=1/15, width=0.005, color='k', linewidth=2,zorder=30)
    ax.quiver(pgoal[0], pgoal[1], np.cos(yaw_goal), np.sin(yaw_goal), scale_units='xy', scale=1/15, width=0.005, color='k', linewidth=2,zorder=30)

    ax2.plot(pinit[0], pinit[1], pinit[2], '*k', markersize=14)
    ax2.plot(pgoal[0], pgoal[1], pgoal[2], '*k', markersize=14)
    ax2.quiver(pinit[0], pinit[1], pinit[2], np.cos(yaw_init), np.sin(yaw_init), np.tan(pitch_init), length=0.1*scale, normalize=True, color='k')
    ax2.quiver(pgoal[0], pgoal[1], pgoal[2], np.cos(yaw_goal), np.sin(yaw_goal), np.tan(pitch_goal), length=0.1*scale,  normalize=True, color='k')
    ax2.text(pinit[0]-15, pinit[1]+5, pinit[2],rf'$W_{ii}$',  fontsize=30)
ax.text(pgoal[0], pgoal[1]-15,rf'$W_{ii+1}$', fontsize=30)
ax2.text(pgoal[0], pgoal[1]-15, pgoal[2]-10, rf'$W_{ii+1}$', fontsize=30)

 

ax.minorticks_on()
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$y$ (m)')
ax.set_xlim(-problem_params['space_limit'][0]*scale, problem_params['space_limit'][0]*scale)
ax.set_ylim(-problem_params['space_limit'][0]*scale, problem_params['space_limit'][0]*scale)
ax.legend(fontsize="14", loc ="upper left")#, ncol = len(yawr)) 
ax2.view_init(elev=9, azim=-73) 
ax2.set_xlabel(r'$x$ (m)')
ax2.set_ylabel(r'$y$ (m)')
ax2.set_zlabel(r'$z$ (m)')
ax2.set_zlim(0,problem_params['space_limit'][0]*scale)
ax2.set_xlim(-problem_params['space_limit'][0]*scale, problem_params['space_limit'][0]*scale)
ax2.set_ylim(-problem_params['space_limit'][0]*scale, problem_params['space_limit'][0]*scale)

fig.set_tight_layout(True)
########################


ax.annotate(f'(a) 2D view', xy=(.5, -0.1), xycoords='axes fraction', ha='center')
ax2.annotate(f'(b) 3D view', xy=(0.5, 0.1), xycoords='axes fraction', ha='center')




fig3 = plt.figure(figsize=(12,10))
fig4 = plt.figure(figsize=(12,10))
ax = fig3.add_subplot(111)
ax2 = fig4.add_subplot(111)

   
############################
kappa1, climb_angle1 = get_kappa_climb_functions(curve1)

kappamax = 1/40#problem_params['kappamax']#/scale/1000
thetamax = problem_params['thetamax']*180/np.pi
thetamin = problem_params['thetamin']*180/np.pi

ax.plot(1/(len(kappa1))*np.array(range(len(kappa1))), kappa1, markersize=4, color='b', label=r'$C(s)$')#, marker=".", markersize=markersize)

ax.hlines(kappamax,0, 1/(len(kappa1))*len(kappa1), 'r', linestyles=['dashed'])
ax.set_ylabel(r'$\kappa(s) $')
ax.set_xlabel('s')

ax.legend()

ax2.plot(1/(len(climb_angle1))*np.array(range(len(climb_angle1))), climb_angle1, markersize=4, color='b', label=r'$C(s)$')


ax2.set_ylabel(r'$\theta(s)$')

knotvectori = [0,0.2,0.4,0.6, 0.8,1]
idx = 0
for ii, wp in zip(knotvectori, list_waypoints2):
    wp = wp[0]
    
    ax.vlines(ii, 0, kappamax*1.2, 'k',linestyles=['dashed'])
    ax2.plot(ii, wp[-1], 'xk')
    ax2.text(ii, wp[-1]+1, rf'$\theta_{idx}$', fontsize=16, color='k',zorder=30)
    ax.text(ii+0.05, kappamax*1.1, rf'$W_{idx} \mapsto W_{idx+1}$', fontsize=16, color='k',zorder=30)
    idx +=1
ax2.hlines(thetamax,0, 1, 'r', linestyles=['dashed'])
ax2.hlines(thetamin,0, 1, 'r', linestyles=['dashed'])
ax2.set_xlim(0,1)
ax.set_xlim(0,1)
ax2.set_xlabel('s')
ax2.legend()

fig.savefig(filepath_name+'scenario5.png', bbox_inches = 'tight', pad_inches = 0, dpi=300)
fig3.savefig(filepath_name+'scenario5_kappa.png', bbox_inches = 'tight', pad_inches = 0, dpi=300)
fig4.savefig(filepath_name+'scenario5_climb.png', bbox_inches = 'tight', pad_inches = 0, dpi=300)
plt.show()


