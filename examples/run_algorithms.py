#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example to run our planner with differents optimization algorithms

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8

Usage:
- Select the algoirthm in the "list_algorithms"

"""

import sys
import os

# Add the scripts directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from de3dnurbs import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
# from robots_models import *
from __utils import *
from topography import *

from joblib import Parallel, delayed
from __experiment_setup import SAVE

print("running...")

################################ Problem's definition
nurbs = True
degree = 7
MAX_EVALS = 600000
SCENARIO = 5 # Complex scenario


# list_algorithms = ["CDE", "CDE", "LSHADE-COP", "PSO", "FFO", "SADE", "LSHADE", "HPSO", "LSHADE-COP"]
# list_population_size = [50, 396, 396, 100, 100, 100, 100, 100,100]
list_algorithms = ["LSHADE-COP"]
list_population_size = [396]

memory_size = 40


start_statistical = 0
N_statistical = 1
N_statistical += start_statistical

###########################

filepath_config = "./results/"
#### Only the first time to run
if SAVE:
    for algo in list_algorithms: 
        with open(filepath_config + 'result_algorithms.csv', "w") as arquivo:
            arquivo.write("algo\tname\tlength\tfeasible\ttime\n")
print("algo\tinstance\tlength\tfeasible\ttime")
##########

#######################################Scenarios
####### Scenario 1
if SCENARIO == 1:
    scale = 400
    space_scale = 10
    kappa_scale = 1
    topography_index = 0
    obstacles_cylinders = []
    hole = []
    list_instances = ['Long_1', 'Long_2', 'Long_3', 'Long_4', 'Long_5', 'Short_1', 'Short_2', 'Short_3', 'Short_4', 'Short_5']#, 'Short 6']
    list_waypoints = [
                    [[200,500,200] ,   180  ,  -5 ], \
                    [[500,350,100] ,  0 ,  -5 ] ,\
                    
                    [[100,-400,100] ,  30 ,  0 ],\
                    [[500,-700,0],  150 ,  0] ,\

                    [[-200,200,250] ,   240  ,  15 ],\
                    [[500,800,0] ,  45 ,  15 ] ,\

                    [[-300,1200,350] ,   160  ,  0 ],\
                    [[1000,200,0] ,  30,  0 ] ,\

                    [[-500,-300,600] ,   150  ,  10 ],\
                    [[1200,900,100] ,  -60 ,  10 ] ,\

                    [[120,-30,250] ,   100  ,  -10 ],\
                    [[220,150,100] ,  -60,  -10 ] ,\

                    [[380,230,200] ,   30  ,  0 ],\
                    [[280,150,100] ,  200 ,  0 ] ,\

                    [[-80,10,250] ,   20  ,  0 ],\
                    [[50,70,0] ,  240 ,  0 ] ,\

                    [[400,-250,600],   -10  ,  0 ],\
                    [[600,-150,300] ,  150 ,  0 ] ,\
                    
                    [[-200,-200,450] ,   -20  ,  0 ],\
                    [[-300,-80,100] ,  100 ,  0 ] ,\
                    # [[0,40,100] ,   180  ,  0 ],\
                    # [[0.7, 39.99, 100] ,  179 ,  0 ] ,\
                    [],\
                    ]
####### Scenario 1.1 (Circle)
elif SCENARIO == 1.1:
    scale = 400
    space_scale = 1
    kappa_scale = 1
    topography_index = 0
    hole = []
    obstacles_cylinders = []
    list_instances = ['Short_6']
    list_waypoints = [                    
                    [[0,40,100] ,   180  ,  0 ],\
                    [[0.7, 39.99, 100] ,  179 ,  0 ] ,\
                    ]
########## Scenario 2 
elif SCENARIO == 2:
    scale = 400
    space_scale = 1
    kappa_scale = 1
    topography_index = 0
    list_waypoints = [
                    [[-190 , -80 , 80] ,   0  ,  0 ], \
                    [[ 190 , -80 , 80] ,  0 ,   -5 ] ,\
                    
                    [],\
                    ]
    list_instances = ['Short_9']    
    obstacles_cylinders = []
    hole = [[0.,0.3,0.3], [0.1,0.0075,0.0075], 0]
###################
########## Scenario 3 (changes in environment)
elif SCENARIO == 3.1:
    scale = 400
    space_scale = 1
    kappa_scale = 1
    topography_index = 2
    hole = []
    list_waypoints = [
                    [[-145 , 0 , 6] ,   0  ,  0 ], \
                    [[ 145 , 0 , 20] ,  0 ,  -5 ] ,\
                    # [[ 145 , 0 , 20] ,  0 ,  -5 ] ,\
                    # [[-145 , 0 , 6] ,   0  ,  0 ],\
                    [],\
                    ]
    list_instances = ['Short_7_1', 'Short_8_1']
    obstacles_cylinders = []
###################
########## Scenario 3 (changes in environment obstacle)
elif SCENARIO == 3.2:
    scale = 400
    space_scale = 1
    kappa_scale = 1
    topography_index = 2
    hole = []
    list_waypoints = [
                    [[-145 , 0 , 6] ,   0  ,  0 ], \
                    [[ 145 , 0 , 20] ,  0 ,  -5 ] ,\
                    [[ 145 , 0 , 20] ,  0 ,  -5 ] ,\
                    [[-145 , 0 , 6] ,   0  ,  0 ],\
                    [],\
                    ]
    list_instances = ['Short_7_2', 'Short_8_2']
    obstacles_cylinders = np.array([
        [   0. ,    0. ,   50. ,   50. ],
        ])
    obstacles_cylinders = list(obstacles_cylinders/scale)    
###################
########## Scenario 4 CHAI
elif SCENARIO == 4:
    scale = 100
    space_scale = 1
    kappa_scale = 1000
    hole = []
    topography_index = 7
    list_waypoints = [
                  [[-45 , -47 , 1.0] , 60  , 0 ], \
                  [[-10 , 20 , 3.0] , 0 , -10] ,\
                  [[-45 , -47 ,1.0] , 60  , 0 ], \
                  [[20, 0 , 4], 135 , -15],\
                  [[-45 , -47 ,1.0] , 60  , 0 ], \
                  [[-10 , -10 , 1.5] , 180  , 5 ], \
                  [[-45 , -47 ,1.0] , 60  , 0 ], \
                  [[ 40 ,  45 , 1.0] , 45  , 0 ], \
                  [],\
                 ]   
    list_instances = ['Long_6', 'Long_7', 'Long_8', 'Long_9']
    obstacles_cylinders = np.array([[-15.,  10.,   7.,   7.],
       [ 35.,  10.,  10.,  10.],
       [-30.,  20.,   7.,   7.],
       [-20., -20.,   8.,   8.],
       [  0., -20.,   7.,   7.],
       [  0.,  30.,   7.,   7.],
       [  0.,  -2.,   7.,   7.],
       [ 20.,  25.,   8.,   8.],
       [ 10.,  10.,  10.,   5.],
       [ 19., -15.,   8.,   8.]])
    obstacles_cylinders = list(obstacles_cylinders/scale)
    
########## Scenario 5 - Complex
elif SCENARIO == 5:
    scale = 400
    space_scale = 1.1
    kappa_scale = 1
    hole = []
    topography_index = 8
    list_waypoints = [
                  [[150 , -150 , 2.8] , 180  , 0 ], \
                  [[-170 , -50 , 180.0] , 90 , 10] ,\
                  
                  [[-170 , -50 , 180.0] , 90 , 10] ,\
                  [[0, 125, 120], 45   , -10] ,\
                  
                  [[0, 125, 120], 45   , -10] ,\
                  [[155, -50 , 120], 180 , -5],\
                  
                  [[155, -50 , 120], 180 , -5],\
                  [[-60, 80 ,  40], -135  , -5] ,\
                  
                  [[-60, 80 ,  40], -135  , -5] ,\
                  [[150 , -150 , 2.8] , 0  , 0 ], \
                  [],\
                 ]   
    
    
    list_instances = ['Complex_1', 'Complex_2', 'Complex_3', 'Complex_4', 'Complex_5']
    obstacles_cylinders = np.array([
       [   0. ,    0. ,   80. ,   80. ],
       [-100. , -100. ,   59.4,   59.4],
       [ 100. ,  100. ,   59.4,   59.4],
       [-125. ,    0. ,   12.5,   12.5],
       [   0. ,  100. ,   12.5,   12.5],
       [   0. , -100. ,   12.5,   12.5],
       [ 125. ,    0. ,   12.5,   12.5],
       [ 100. ,  -75. ,   40. ,   12.5],
       [-125. ,   75. ,   40. ,   40. ],
       [-160. , -160. ,   10. ,   35. ],
       [ 160. ,  160. ,   10. ,   35. ]])
    obstacles_cylinders = list(obstacles_cylinders/scale)
###################
elif SCENARIO == 5.5:
    scale = 400
    space_scale = 1
    kappa_scale = 1
    hole = []
    topography_index = 8
    list_waypoints = [
                    # [[150 , -150 , 2.8] , 180  , 0 ], \
                    # [[-170 , -50 , 180.0] , 90 , 10] ,\
                    # [[-170 , -50 , 180.0] , 90 , 10] ,\
                    # [[0, 125, 120], 45   , -10] ,\
                    # [[0, 125, 120], 45   , -10] ,\
                    # [[150, -50 , 120], 180 , -5],\
                    # [[150, -50 , 120], 180 , -5],\
                    # [[-75, 50 ,  40], -135  , -5] ,\
                    [[-20, 83 ,  40], -150  , -5] ,\
                    [[150 , -150 , 2.8] , 0  , 0 ], \
                    [],\
                    ]   


    list_instances = ['Complex_5']
    obstacles_cylinders = np.array([
        [   0. ,    0. ,   80. ,   80. ],
        [-100. , -100. ,   59.4,   59.4],
        [ 100. ,  100. ,   59.4,   59.4],
        [-125. ,    0. ,   12.5,   12.5],
        [   0. ,  100. ,   12.5,   12.5],
        [   0. , -100. ,   12.5,   12.5],
        [ 125. ,    0. ,   12.5,   12.5],
        [ 100. ,  -75. ,   40. ,   12.5],
        [-125. ,   75. ,   40. ,   40. ],
        [-160. , -160. ,   10. ,   35. ],
        [ 160. ,  160. ,   10. ,   35. ]])
    obstacles_cylinders = list(obstacles_cylinders/scale)
###################
########## Scenario  - Hole
elif SCENARIO == 6.1:
    scale = 400
    space_scale = 3
    kappa_scale = 1
    hole = [np.array([0.,240,240])/scale, np.array([1.5*30,3,3])/scale, 0]
    topography_index = 0
    list_waypoints = [
                  [[-380 , -160 , 180] , -60  , 0 ], \
                  [[ 380 , -160 , 45] ,  60  , -5] ,\
                  [],\
                 ]       
    list_instances = ['hole']    
    obstacles_cylinders = []
###################    
########## Scenario  - Big Hole
elif SCENARIO == 6.2:
    scale = 400
    space_scale = 3
    kappa_scale = 1    
    hole = [np.array([0.,240,240])/scale, np.array([1.5*150,3,3])/scale, 0]    
    topography_index = 0
    list_waypoints = [
                  [[-380 , -160 , 180] , -60  , 0 ], \
                  [[ 380 , -160 , 45] ,  60  , -5] ,\
                  [],\
                 ]   
    list_instances = ['bighole_150'] 
    obstacles_cylinders = []
###################
    
seq_waypoints = []
for idx, waypoint in zip(list_instances, zip(list_waypoints[0:-1:2], list_waypoints[1:-1:2])):
    seq_waypoints.append(waypoint)
    #print(f'{idx:>10}: {waypoint}')

def run_experiment_waypoints(idx, waypoint0, waypoint1, obstacles_cylinders, scale = 1, nurbs=True, algorithm="LSHADE-COP", topography_index=1, id_statistical=0, population_size=396, **kwargs):
    global degree, space_scale
    #print(f'instance = {idx:>10} \t algo = {algorithm:>10} \t population = {population_size} \t statistical = {id_statistical}')
    planner = de3dnurbs(type='DE-NURBS', degree=degree, debug=0)  
    #########################################################################    
    
    if nurbs:
        low_space_search = np.array([-1,-1,-1,    -1,-1,-1,    -1,-1,-1,    -1,-1,-1,    -1,-1,-1,    \
                        0.000001, 0.000001,0.000001, 0.000001, 0.000001,  \
                        0.001, 0.001])*space_scale
        up_space_search = np.array([    1,1,1,       1,1,1,       1,1,1,      1,1,1,        1,1,1,       \
                                1,           1,           1,          1,            1,        \
                                0.25, 0.25])*space_scale
    else:
        low_space_search = np.array([-1,-1,-1,    -1,-1,-1,    -1,-1,-1,    -1,-1,-1,    -1,-1,-1,    \
                        1, 1,1, 1, 1,  \
                        0.01, 0.01])*space_scale
        up_space_search = np.array([    1,1,1,       1,1,1,       1,1,1,      1,1,1,        1,1,1,       \
                                1,           1,           1,          1,            1,        \
                                0.125, 0.125])*space_scale
    if SCENARIO > 1.1:
        r =  0.75 / scale / kappa_scale
    else:
        r = 0
           
    pinit = np.array(waypoint0[0])/scale
    pgoal = np.array(waypoint1[0])/scale
    
    yaw_init = ((waypoint0[1] + 180) % 360 - 180)*np.pi/180.
    yaw_goal = ((waypoint1[1] + 180) % 360 - 180)*np.pi/180.
    pitch_init = ((waypoint0[2] + 180) % 360 - 180)*np.pi/180.
    pitch_goal = ((waypoint1[2] + 180) % 360 - 180)*np.pi/180.    
    dt = 0.0025
    space_limit = np.array([0.5, 0.5, 0.5]) *space_scale
    torsionmax = 1
    phomin = 40/scale/kappa_scale
    kappamax = 1/phomin
    thetamax = 20*np.pi/180
    thetamin = -15*np.pi/180
    alpha = [5000.,  0.0, 0.001, 1.0, 2.0, 0.50, 0.50,  0.00, 0.001, 0.001, 5.0, 0.0, 10., 0.0]
              
                 
    #########################################################################
    problem_params = {
        "degree": degree,
        "scale": scale,
        "r": r,
        "pinit": pinit,
        "pgoal": pgoal,
        "yaw_init": yaw_init,
        "yaw_goal": yaw_goal,
        "pitch_init": pitch_init,
        "pitch_goal": pitch_goal,
        "obs_ellipsoids" : [],             
        "obs_cylinders" : obstacles_cylinders,  
        "topography_index": topography_index,
        "topography_z": get_topography,
        "hole": hole,
        "dt" : dt, 
        "lb": low_space_search,
        "ub": up_space_search,
        "space_limit": space_limit,
        "torsionmax": torsionmax,
        "kappamax": kappamax,
        "thetamax": thetamax,
        "thetamin": thetamin,
        "external_costfunction": None,
        "name_test": idx, 
        "alpha": alpha,
    }
    
    smart_population = smart_init_population(pinit, pgoal, yaw_init, yaw_goal, pitch_init, pitch_goal, \
                                                 scale, population_size, up_space_search, low_space_search,  std=250, vi=0.01, vf=0.01)
    
    if algorithm == "LSHADE-COP":
        algorithm_params = {
            "library": "based_on_pyade",
            "save_data": False,
            "algorithm": "LSHADE",                
            "population_size" : population_size, 
            "memory_size": memory_size,          
            "max_iterations" : MAX_EVALS,  
            "opts": None,  
            "start_line":None,#[line_points],   
            "COP": True,         
            "smart_init_population": smart_population #None, smart_population
        }
    elif algorithm == "LSHADE":        
        algorithm_params = {
            "library": "based_on_pyade",
            "save_data": False,
            "algorithm": "LSHADE",                
            "population_size" : population_size, 
            "memory_size": memory_size,          
            "max_iterations" : MAX_EVALS,  
            "opts": None,  
            "start_line":None,#[line_points],   
            "COP": False,         
            "smart_init_population": None #None, smart_population
        }
    elif algorithm == "CDE":        
        algorithm_params = {
            "library": "based_on_pyade",
            "save_data": False,
            "algorithm": "CDE",                
            "population_size" : population_size,                 
            "max_iterations" : MAX_EVALS,  
            "opts": None,  
            "start_line":None,#[line_points],   
            "COP": False,         
            "smart_init_population": None #None, smart_population
        }
    else:
        algorithm_params = {
            "library": "mealpy",
            "save_data": True,
            "algorithm": algorithm,           #P_PSO, SADE, FFO
            "population_size" : population_size, 
            "memory_size": memory_size,          
            "max_iterations" : MAX_EVALS,  
            "opts": None,  
            "start_line":None,#[line_points],   
            "COP": None,         
            "smart_init_population": None #None, smart_population
        }
    
    #########################################################################
    planner.set_problem(problem_params)
    planner.set_algorithm(algorithm_params)
    best_solution = planner.run()

    res = problem_params, algorithm_params, best_solution, planner.log_time[-1], planner.log_fit[-1], planner.log_experiment[-1]
    name = res[0]['name_test']
    if not nurbs:
        name += '_bs'
    if SAVE:
        if nurbs:
            path2save = filepath_config+'paths/experiment_algorithm_'+ str(algorithm) + '_p' + str(population_size) + '_' +str(degree) + 'th_' +  'nurbs_'  + str(name) + '_S' + str(id_statistical) +  '.pkl'
        else:
            path2save = filepath_config+'paths/experiment_algorithm_'+ str(algorithm) + '_p' + str(population_size) + '_' +str(degree) + 'th_' + 'bspline_'  + str(name) + '_S' + str(id_statistical) +  '.pkl'

        parameters = ["problem_params", "algorithm_params", "solution", "time_solved","fitness", "experiment_data","list_waypoints","obstacles_cylinders",\
                    
        ]
        dict_parameters = dict(zip(parameters, range(len(parameters))))
        
        with open(path2save, 'wb') as f: 
            pickle.dump([res[0], res[1], res[2], res[3], res[4], res[5], [waypoint0, waypoint1],obstacles_cylinders,
                dict_parameters], f)
    
    problem_params = res[0]
    solution = res[2]
    experiment_data = res[5]
    plannerT = de3dnurbs(type='DE-NURBS', degree=degree, debug=0)  
    problem_params['dt'] = problem_params['dt']
    dt = problem_params['dt'] 
    scale = problem_params['scale']
    fx, fy, fz, curve_nurbs = plannerT.get_curve_points(solution, problem_params, dt=problem_params['dt'], scale=scale)
    curve_pts = np.array([fx,fy,fz])
    diff_curve = np.diff(curve_pts.T, axis=0)  
    norm_diff_curve = np.linalg.norm(diff_curve, axis=1)
    """ Length curve """        
    length = np.sum(norm_diff_curve)
    
    penalties = check_penalties_problem_params(curve_pts.T*kappa_scale, problem_params, tol=3/100, km=kappa_scale)   
    
    print(str(algorithm)+ '_p' + str(population_size) + "\t" + str(name) + "\t" +str(length) + "\t" + str(penalties == (0,0,0,0,0)) + "\t" + str(res[3]))
    if SAVE:
        with open(filepath_config + 'result_algorithms.csv', "a") as arquivo:          
            arquivo.write( str(algorithm)+ '_p' + str(population_size) + "\t" + str(name) + "\t" +str(length) + "\t" + str(penalties == (0,0,0,0,0)) + "\t" + str(res[3]) + "\n")




    return problem_params, algorithm_params, best_solution, planner.log_time[-1], planner.log_fit[-1], planner.log_experiment[-1]


### RUNNING


result = Parallel(n_jobs=-1)(delayed(run_experiment_waypoints)(idx, waypoint[0], waypoint[1], obstacles_cylinders,scale=scale, nurbs=nurbs, \
                                                                algorithm=algo, population_size=population_size, topography_index=topography_index, id_statistical=id_statistical) \
                                                                for id_statistical in range(start_statistical, N_statistical)\
                                                                for idx, waypoint in zip(list_instances, seq_waypoints) \
                                                                for algo, population_size in zip(list_algorithms, list_population_size) \
                                                                )
