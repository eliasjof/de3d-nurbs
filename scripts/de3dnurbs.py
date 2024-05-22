#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DE3D-NURBS library

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8


Usage:
1. Set the global variables to use ROS, SAVE datalog. 
2. Import this module into your Python script.
2. Initialize the optimization problem by defining the objective function, constraints, and other parameters.
3. Call the `l_shade` function to perform optimization. Pass the problem parameters as arguments.
4. Retrieve the optimized solution and other relevant information.

Example:
    $ planner = de3dnurbs(type='DE-NURBS', degree=5)

"""
import lshade_cop
import cde
import pyade_commons
import numpy as np
import time


from __utils import *
from __debug import *
from topography import *


from geomdl import NURBS
from geomdl import knotvector

#import inspect
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# others algorithm to test
from mealpy import FloatVar
# from mealpy.bio_based import BBO
from mealpy.evolutionary_based.DE import JADE, SADE
# from mealpy.evolutionary_based.MA import OriginalMA
# from mealpy.swarm_based.MSA import OriginalMSA
from mealpy.swarm_based.PSO import CL_PSO, P_PSO, OriginalPSO, HPSO_TVAC
from mealpy.swarm_based.FFO import OriginalFFO
from mealpy.swarm_based.POA import OriginalPOA
from mealpy.swarm_based.FFA import OriginalFFA
from mealpy.utils.problem import Problem

############################################################################################################
### UTILS

def check_penalties_problem_params(curve_pts_scaled, problem_params, kappamax = 1/40, tol = 0.01, km=1):
    """ penalty_space_limit, penalty_obs, penalty_climb_angle, penalty_kappa, topography """
    fx = curve_pts_scaled[:,0]
    fy = curve_pts_scaled[:,1]
    fz = curve_pts_scaled[:,2]
    scale = problem_params['scale']
    obs_cylinders = problem_params['obs_cylinders']
    r = problem_params['r']
    space = np.array(problem_params['space_limit'])*scale*km    
    penalty_obs = 0
    penalty_space_limit = 0
    tol = 1 + tol
    ### OBSTACLES    
    d = check_collision_cylinder2line(curve_pts_scaled/scale/km, obs_cylinders, r, tol=tol)  
    penalty_obs = len(d) #+ np.sum(1/d**2))

    ## SPACE
    
    penalty_space_limit = np.sum(fx < -space[0]*tol + r*tol) + \
                  np.sum(fy < -space[1]*tol + r*tol) + \
                  np.sum(fx > space[0]*tol - r*tol) + \
                  np.sum(fy > space[1]*tol - r*tol) + \
                  np.sum(fz > space[2]*tol - r*tol)
    
    kappa, climb_angle = get_kappa_climb_functions(curve_pts_scaled)
    # kappa1 = kappa1[0:-1:4]
    # climb_angle1 = climb_angle1[0:-1:4]

    
    thetamax = problem_params['thetamax']*180/np.pi
    thetamin = problem_params['thetamin']*180/np.pi

    penalty_kappa = len(kappa[kappa > kappamax*tol])
    g2a =  climb_angle[climb_angle > thetamax*tol]
    g2b =  climb_angle[climb_angle < thetamin*tol]
    penalty_climb_angle = len(g2a)+len(g2b)

    """ topography """
    if problem_params['topography_index'] > 0:
        if callable(problem_params['topography_z']):            
            z_mountain = problem_params['topography_z'](fx/scale, fy/scale, problem_params['topography_index'])
            # print(len(z_mountain), len(fz))
            penalty_topography = np.sum(z_mountain >= fz/scale - 2*r)
    else:
        penalty_topography = 0
       

    return penalty_space_limit, penalty_obs, penalty_climb_angle, penalty_kappa, penalty_topography


def check_penalties(curve, yaw_init, yaw_goal, pitch_init, pitch_goal, obs_cylinders, r, kappamax, thetamax, thetamin, space, scale=1, tol = 0.01):
    """ penalty_space_limit, penalty_obs, penalty_climb_angle, penalty_kappa """
    fx = curve[:,0]
    fy = curve[:,1]
    fz = curve[:,2]
    penalty_obs = 0
    penalty_space_limit = 0
    tol = 1 + tol
    ### OBSTACLES    
    d = check_collision_cylinder2line(curve/scale, obs_cylinders/scale,r*tol/scale)  
    penalty_obs = len(d) #+ np.sum(1/d**2))

    ## SPACE
    
    penalty_space_limit = np.sum(fx < -space[0]*tol + r*tol) + \
                  np.sum(fy < -space[1]*tol + r*tol) + \
                  np.sum(fx > space[0]*tol - r*tol) + \
                  np.sum(fy > space[1]*tol - r*tol) + \
                  np.sum(fz > space[2]*tol - r*tol)
    # np.sum(fz > space[2]*tol - r*tol)
    # penalty_space_limit += len(fz[fz > space[2]- r])#*current_generation
    

    """  curvature function """     
    rl = np.diff(curve, axis=0)#/self.dt
    norm_rl = np.linalg.norm( rl, axis=1 )[:-1]
    rl2 = np.diff(rl, axis=0)#/self.dt
    # printD(f"grad = {grad_curve.shape}, nrl = {norm_rl.shape}, rl2 = {rl2.shape}")
    cross_rl_rl2 = np.cross(rl[:-1], rl2)        
    kappa = np.linalg.norm(cross_rl_rl2, axis=1)/norm_rl**3
    

    kappa_initial = 0#kappamax#0
    diff_kappa = np.diff(kappa)
    zero_kappa = np.sum(np.abs(kappa[0:1]- kappa_initial)) + np.sum(np.abs(kappa[-1:]-kappa_initial))
    diff_zero_kappa = np.sum(np.abs(rl[0:1]- kappa_initial)) + np.sum(np.abs(rl[-1:]-kappa_initial)) # initial and final curvature desired to be zero
    kappa_problem = kappa[kappa>kappamax*tol]
    # kappa_problem2 = kappa[kappa>kappamax/2]
    e_kappa= np.abs(kappa_problem - kappamax)
    
    """climb angle penalty """ 
    diff_curve = np.diff(curve, axis=0)    
    climb_angle = np.arctan2(diff_curve[:,2], (diff_curve[:,0]**2 + diff_curve[:,1]**2)**0.5)
    g2a =  climb_angle[climb_angle > thetamax*tol]
    g2b =  climb_angle[climb_angle < thetamin*tol]

    grad_climb_angle = np.diff(climb_angle, axis=0)
    

    length = np.sum(np.linalg.norm(np.diff(curve.T), axis=0))

    
    N_points = 3
    initial_points = np.diff(curve[0:N_points], axis=0)
    final_points = np.diff(curve[len(curve)-N_points:len(curve)], axis=0)
    penalty_e_init = 0
    penalty_e_goal = 0
    yi=[]
    yg=[]
    x = 1
    for i, point in enumerate(initial_points):
        # printD(point, point.shape)
        yaw = np.arctan2(point[1], point[0])
        pitch = np.arctan2(point[2],(point[0]**2 + point[1]**2)**0.5)
        # printD(pitch*180/np.pi)
        yi.append(yaw)            
        penalty_e_init += x*(yaw_init - yaw)**2 + x*(pitch_init - pitch)**2
        x /= 10
    x = 1
    for i, point in enumerate(final_points):
        # printD(point, point.shape)
        yaw = np.arctan2(point[1], point[0])
        pitch = np.arctan2(point[2],(point[0]**2 + point[1]**2)**0.5)


        yg.append(yaw)
        penalty_e_goal += x*(yaw_goal - yaw )**2 + x*(pitch_goal - pitch)**2
        x /= 10

    penalty_climb_angle = len(g2a)+len(g2b) #(np.sum((g2a-thetamax)**2)+np.sum((g2b-thetamin)**2)
    penalty_kappa = (len(kappa_problem))
    penalty_dkappa = diff_zero_kappa 
    penalty_kappazero = zero_kappa

    return penalty_space_limit, penalty_obs, penalty_climb_angle, penalty_kappa# penalty_kappazero, penalty_dkappa



def join_curves(list_of_knots, list_of_ctrl_points, list_of_weights, degree = 7, smooth_beta = 0):
    """ proposed Joint curve algorithm """
    degree_plus1 = degree + 1
    final_knot = list_of_knots[0]
    final_ctrl_points = list_of_ctrl_points[0]
    final_weights = list_of_weights[0]

    for knot_i, ctrl_points_i, weights_i in zip(list_of_knots[1:], list_of_ctrl_points[1:], list_of_weights[1:]):
        end_knot1 = final_knot[-1]
        knot_i = list(np.array(knot_i)+end_knot1)

        if smooth_beta:
            smooth_i = (knot_i[degree_plus1] - 1 )/smooth_beta
            final_knot[-2] = final_knot[-2]  + smooth_i
            final_knot[-3] = final_knot[-3]  + smooth_i/2
            final_knot[-4] = final_knot[-4]  + smooth_i/3
        
        final_knot = final_knot[:-1] + knot_i[degree_plus1:]        
        final_ctrl_points = final_ctrl_points + ctrl_points_i[1:]
        
        final_weights = final_weights + weights_i[1:]
    final_knot = list(np.array(final_knot)/final_knot[-1])
    printD(len(final_knot),final_knot)
    return final_knot, final_ctrl_points, final_weights

def generate_curve(knot, ctrl_points, weights, dt = 0.001, degree=7):
    curve_nurbs = NURBS.Curve()
    # Set degree
    curve_nurbs.degree = degree

    ctrlpts = []
    for pt in ctrl_points:
        ctrlpts.append(pt)
    curve_nurbs.ctrlpts = ctrlpts
    printD(knot)
    curve_nurbs.knotvector = knot
    curve_nurbs.delta = dt

    curve_nurbs.weights = weights

    # num_points = int(1/dt)
    # start_param = curve_nurbs.knotvector[curve_nurbs.degree]
    # end_param = curve_nurbs.knotvector[-(curve_nurbs.degree + 1)]
    # uniform_params = np.linspace(start_param, end_param, num_points)

    # # Evaluate the curve at the uniform parameter values
    # f = np.array(curve_nurbs.evaluate_list(uniform_params))
    curve_nurbs.evaluate()        
    f = np.array(curve_nurbs.evalpts)
    fx = np.array([fi[0] for fi in f])
    fy = np.array([fi[1] for fi in f])
    fz = np.array([fi[2] for fi in f])
    curve_points = np.array([fx,fy,fz]).T
    

    return curve_nurbs, curve_points

def get_kappa_climb_functions(curve):
    rl = np.diff(curve, axis=0)
    norm_rl = np.linalg.norm( rl, axis=1 )[:-1]
    rl2 = np.diff(rl, axis=0)    
    cross_rl_rl2 = np.cross(rl[:-1], rl2)        
    kappa = np.linalg.norm(cross_rl_rl2, axis=1)/norm_rl**3

    # rl = np.gradient(curve, axis=0, edge_order=2)
    # norm_rl = np.linalg.norm( rl, axis=1 )
    # rl2 = np.gradient(rl, axis=0, edge_order=2)    
    # cross_rl_rl2 = np.cross(rl, rl2)        
    # kappa = np.linalg.norm(cross_rl_rl2, axis=1)/norm_rl**3
    
    

    diff_curve = np.diff(curve, axis=0)               
    climb_angle = np.arctan2(diff_curve[:,2], (diff_curve[:,0]**2 + diff_curve[:,1]**2)**0.5)*180/np.pi

    return kappa, climb_angle


def smart_init_population(pinit, pgoal, yaw_init, yaw_goal, pitch_init, pitch_goal, scale, population_size, ub, lb, std=5, vi=0.1/4, vf=0.1/4):  

    n_ctrl_pts = (len(lb)-2)//4   
    line_points, _, _, _ = generate_line_points(pinit, pgoal, yaw_init, yaw_goal, pitch_init, pitch_goal, n_ctrl_pts, vi=vi, vf=vf)  
    # printD(len(lb), n_ctrl_pts)
    for _ in range(len(lb)-n_ctrl_pts*3):
        line_points.append(1)
    # printD(len(line_points))
    line_points[-1] = 10*1/scale#*np.linalg.norm(np.array(pgoal) - np.array(pinit))
    line_points[-2] = 10*1/scale#*np.linalg.norm(np.array(pgoal) - np.array(pinit))
    parameters = line_points
    points = []   
    for i in range(0,n_ctrl_pts*3,3):                
        points.append([parameters[i], parameters[i+1], parameters[i+2]])
    weigths = []
    weigths.append(1) # P0             
    weigths.append(1) # P1
    weigths.append(1) # P2
    for ii in range(-3,-n_ctrl_pts-3,-1):
        weigths.append(parameters[ii])        
    weigths.append(1) # P_last-2   
    weigths.append(1) # P_last-1
    weigths.append(1) # P_last
    stds = std*(np.array(ub) - np.array(lb))
    population = []  
    base = line_points.copy()
    for _ in range(population_size//3-1):
        pop = np.random.normal(base, stds)        
        pop[:9] = base[:9]
        pop[9:n_ctrl_pts*3] = base[9:n_ctrl_pts*3]
        pop_c = np.random.normal(stds[n_ctrl_pts*3:], 1)
        pop[n_ctrl_pts*3:] = pop_c

        num = np.random.choice(len(base))
        for _ in range(num):
            if np.random.uniform(0,1) > 0.5:
                id = np.random.choice(len(base))
                id2 = np.random.choice(len(base))
                aux = pop[id]
                pop[id] = pop[id2]
                pop[id2] = aux
        
        population.append(pop)
    
    
    # population.append(base)
    population[0] = line_points
    # population = np.clip(population, lb, ub)
    
    boundaries = []
    for lbi, ubi in zip(lb, ub):
        boundaries.append([lbi, ubi])
    population2 = pyade_commons.init_population(population_size-len(population), individual_size=len(ub), bounds=boundaries)# np.random.randn(population_size-len(population), len(ub))
    population.extend(population2)
    population = np.array(population)    
    printD(f'len pop = {population.shape}')
    
    return np.clip(population, lb, ub)

def generate_line_points(pti, ptf, gammai, gammaf, thi, thf, num_points, vi=0.01, vf=0.01):
    """
    Generate a list of 3D points that form a line between two given points.

    Args:
        point1 (list): The coordinates of the first point [x, y, z].
        point2 (list): The coordinates of the second point [x, y, z].
        num_points (int): The number of points to generate along the line.

    Returns:
        list: A list of 3D points forming a line between the two input points.
    """
    # gammai = 0
    # gammaf = 0
    distT = np.linalg.norm(np.array(ptf) - np.array(pti))
    # vi = 0.01 #porcentagem do comprimento total da distância entre os dois pontos
    # vf = 0.01
    dxi =  np.cos(gammai)*vi
    dyi =  np.sin(gammai)*vi
    pti2x = pti[0] + dxi*2/3
    pti2y = pti[1] + dyi*2/3
    # printD(thi)
    pti2z = pti[2] + np.tan(thi)*vi*2/3#(dxi**2 + dyi**2)**0.5

    pti3x = pti2x + dxi*1/3
    pti3y = pti2y + dyi*1/3        
    pti3z = pti2z + np.tan(thi)*vi*1/3#(dxi**2 + dyi**2)**0.5



    dxf = np.cos(gammaf)*vf
    dyf = np.sin(gammaf)*vf
    ptf2x = ptf[0] - dxf*2/3
    ptf2y = ptf[1] - dyf*2/3
    ptf2z = ptf[2] - np.tan(thf)*vf*2/3#(dxf**2 + dyf**2)**0.5


    ptf3x = ptf2x - dxf*1/3
    ptf3y = ptf2y - dyf*1/3        
    ptf3z = ptf2z - np.tan(thf)*vf*1/3
    
    line_points = []# [pti2x,  pti2y , pti2z, pti3x,  pti3y , pti3z]
    
    point1 = [ pti3x,  pti3y , pti3z]
    point2 = [ ptf2x,  ptf2y , ptf2z]
    point3 = [ ptf3x,  ptf3y , ptf3z]

    x = np.linspace(point1[0], point3[0], num_points+2)
    y = np.linspace(point1[1], point3[1], num_points+2)
    z = np.linspace(point1[2], point3[2], num_points+2)

    
    for xi, yi, zi in zip(x, y, z):
        line_points.extend([xi, yi, zi])
    
    # printD(len(line_points))
    # remove initial points
    line_points = line_points[1*3:]
    line_points = line_points[:-1*3]
    
    return line_points, x, y, z

############################################################################################################

class de3dnurbs:

    def __init__(self, type = "DE-NURBS", degree=7, debug=0 ):
        '''
        type = DE-NURBS        
        use set_problem and set_algorithm to customize
        '''   
        self.type = type        

        self.problem_defined = False
        self.algorithm_defined = False
        

        # log variables
        self.log_fit = []
        self.log_experiment = []
        self.log_time = []
        self.log_path = [] 
        self.debug = debug
        
        # log variables
        self.log_fit = []
        self.log_time = []
        self.log_path = [] 
        self.current_generation = 1
        self.degree = degree
    
    def set_problem(self, problem_params = None):
        ''' Set problem params         
        '''
        if problem_params is None:            
            printD("Define the problem params....")
            return None
        self.problem_params = problem_params       
        
        self.scale = problem_params["scale"]
        self.space_limit = np.array(self.problem_params["space_limit"])
        self.r = self.problem_params["r"]
        self.obs_ellipsoids = self.problem_params["obs_ellipsoids"]
        self.obs_cylinders = self.problem_params["obs_cylinders"]        
        self.topography_index = self.problem_params["topography_index"]
        self.topography_z = self.problem_params["topography_z"]
        self.center_obs_cylinders = np.array([np.array([ob[0], ob[1],0.0]) for ob in self.obs_cylinders])
        self.radius_obs_cylinders = np.array([np.array([ob[2], ob[3]]) for ob in self.obs_cylinders])
        self.pinit = self.problem_params["pinit"]
        self.pgoal = self.problem_params["pgoal"]
        self.yaw_init = self.problem_params["yaw_init"]
        self.yaw_goal = self.problem_params["yaw_goal"]
        self.pitch_init = self.problem_params["pitch_init"]
        self.pitch_goal = self.problem_params["pitch_goal"]

        self.yaw_init = ((self.yaw_init + np.pi) % (2*np.pi) - np.pi)
        self.pitch_init = ((self.pitch_init + np.pi) % (2*np.pi) - np.pi)
        self.yaw_goal  = ((self.yaw_goal  + np.pi) % (2*np.pi) - np.pi)
        self.pitch_goal = ((self.pitch_goal + np.pi) % (2*np.pi) - np.pi)        
        self.dt = self.problem_params["dt"]                
        self.kappamax = self.problem_params["kappamax"]        
        self.thetamax = self.problem_params["thetamax"]
        self.thetamin = self.problem_params["thetamin"]


        try:
            self.hole = self.problem_params["hole"]
        except:
            self.hole = []

        if problem_params["external_costfunction"] is not None: 
            self.costfunction = problem_params["external_costfunction"]              
        self.alpha = problem_params["alpha"]
        
        printD(f'de problem = {self.problem_params}')
        printD(f'dt = {self.dt}')
        printD(f'kappa_max = {self.kappamax}')

        self.problem_defined = True        
        
    def get_problem_params(self):
        return self.problem_params
    
    def get_algorithm_params(self):
        return self.algorithm_params
    
    def set_algorithm(self,algorithm_params=None):
        '''
        Set algorithm params:
        
        '''     
        if not self.problem_defined:
            printD("Problem is not defined.")
            return None
        if algorithm_params is None:            
            printD("Define the algorithm params")
            return None
        
        self.algorithm_params = algorithm_params

        if self.type in "DE-NURBS":            
            self.dim = len(self.problem_params["lb"])
            printD(f'optimization dimension = {self.dim}')

        self.algorithm = algorithm_params["algorithm"]

        # List of algorithms
        if algorithm_params["library"] in "based_on_pyade":
            if self.algorithm in "LSHADE":
                self.model = lshade_cop
                self.problem_dict = self.model.get_default_params(dim=self.dim)  
                self.problem_dict['COP'] = algorithm_params['COP']  
                if self.problem_dict['COP']:                    
                    printD(f"LSHADE-COP, COP={algorithm_params['COP']}")            
                else:
                    printD(f"LSHADE, COP={algorithm_params['COP']}")                  
                if algorithm_params["save_data"]: self.model.save_data() 
                self.problem_dict['smart_init_population'] = algorithm_params['smart_init_population']                                 
                """ We define the boundaries of the variables """
                boundaries = []
                for lb, ub in zip(self.problem_params["lb"], self.problem_params["ub"]):
                    boundaries.append([lb, ub])                
                self.problem_dict['bounds'] = np.array(boundaries)
                self.problem_dict['population_size'] = algorithm_params["population_size"]
                self.problem_dict['memory_size'] = algorithm_params["memory_size"]
                self.problem_dict['max_evals'] = algorithm_params["max_iterations"]
                self.problem_dict['opts'] = algorithm_params["opts"]#None
                
                if self.problem_dict['opts'] is None:                
                    self.costfunction = self.costfunction_nurbs
                    printD("LSHADE running normal NURBS...")
                elif len(self.problem_dict['opts']) > 1:                
                    self.costfunction = self.costfunction_nurbs_opt
                    printD("LSHADE running opts NURBS...")
                else:
                    self.costfunction = self.costfunction_nurbs
                    printD("LSHADE running add line NURBS...")

                self.problem_dict['func'] = self.costfunction
                if algorithm_params["start_line"] is not None:
                    printD("Add start line in the population...")
                self.problem_dict["start_line"] = algorithm_params["start_line"]                        
            elif self.algorithm in "CDE":
                printD(f"CDE")
                self.model = cde
                self.problem_dict = self.model.get_default_params(dim=self.dim)  
                """ We define the boundaries of the variables """
                boundaries = []
                for lb, ub in zip(self.problem_params["lb"], self.problem_params["ub"]):
                    boundaries.append([lb, ub])                
                self.problem_dict['bounds'] = np.array(boundaries)
                self.problem_dict['population_size'] = algorithm_params["population_size"]                
                self.problem_dict['max_evals'] = algorithm_params["max_iterations"]
                self.problem_dict['opts'] = algorithm_params["opts"]#None
                self.costfunction = self.costfunction_nurbs
                self.problem_dict['func'] = self.costfunction              


        
        elif algorithm_params["library"] in "mealpy":            
            self.costfunction = self.costfunction_nurbs_mealpy
            self.problem_dict = {
            "obj_func": self.costfunction,
            "n_dims": self.dim,
            "bounds": FloatVar(lb=self.problem_params["lb"], ub=self.problem_params["ub"]),
            "minmax": "min",
            "log_to": None,
            }      
            if self.algorithm in "SADE":
                self.model = SADE(epoch=100000, pop_size=algorithm_params["population_size"])
            elif self.algorithm in "JADE":
                self.model = JADE(epoch=100000, pop_size=algorithm_params["population_size"])    
            elif self.algorithm in "CL_PSO":
                self.model = CL_PSO(epoch=100000, pop_size=algorithm_params["population_size"])
            elif self.algorithm in "P_PSO":
                self.model = P_PSO(epoch=100000, pop_size=algorithm_params["population_size"])
            elif self.algorithm in "FFA":
                self.model = OriginalFFA(epoch=100000, pop_size=algorithm_params["population_size"])                
            elif self.algorithm in "FFO":
                self.model = OriginalFFO(epoch=100000, pop_size=algorithm_params["population_size"])
            elif self.algorithm in "PSO":
                self.model = OriginalPSO(epoch=100000, pop_size=algorithm_params["population_size"])
            elif self.algorithm in "HPSO":
                self.model = HPSO_TVAC(epoch=100000, pop_size=algorithm_params["population_size"])
                

        self.algorithm_defined = True

    def print_params(self) :
        printD(self.type)
        printD(self.algorithm_params)
        printD(self.problem_params)

    def run(self, repeat_test=1):

        if not self.problem_defined:
            printD("Problem is not defined.")
            return None
        if not self.algorithm_defined:
            printD("Algorithm is not defined.")
            return None

        self.log_fit  = []
        self.log_time = []
        self.log_path = []
        self.log_length = []
        
        for i in range(repeat_test):                        
            start = time.time()
            # We run the algorithm and obtain the results                        
            if self.algorithm_params["library"] in "based_on_pyade":  
                solution, fit_value, experiment_data = self.model.apply(**self.problem_dict)
            elif self.algorithm_params["library"] in "mealpy":  
                term_dict = {
                "max_fe": self.algorithm_params["max_iterations"]    # number of function evaluation
                }
                gbest = self.model.solve(self.problem_dict, termination=term_dict)               
                solution = gbest.solution 
                fit_value = gbest.target.fitness
                printD(solution)
                experiment_data = []
            now = time.time()
            time_elapse = now - start 

            best_solution = solution                         
            self.log_experiment.append(experiment_data)            
            self.log_path.append(best_solution)
            self.log_fit.append(fit_value)
            self.log_time.append(time_elapse)
            fx, fy, fz, _ = self.get_curve_points(best_solution, self.problem_params)
            curve = np.array([fx, fy, fz]).T*self.problem_params['scale']
            length = np.sum(np.linalg.norm(np.diff(curve.T).T, axis=1))
            self.log_length.append(length)
        return best_solution
         
            
    def printlog(self, msg):
        if self.debug:
            printD(msg)    

    
    def set_obstacles_ellipsoids(self, obs):
        self.obs_ellipsoids = obs   

    def set_obstacles_cylinders(self, obs):
        self.obs_cylinders = obs    
    
    def nurbs(self, pti, ptf, gammai, gammaf, thi, thf, points, weigths = [1.0,1.0,1.0,1.0,1.0], \
              alpha1=0.5, alpha2=0.5, angle_format='r', dt=0.01,scale=1):        
        
        if angle_format=='d':
            gammai = ((gammai*180/np.pi + 180) % 360 - 180)*np.pi/180.
            gammaf = ((gammaf*180/np.pi) % 360 - 180)*np.pi/180.
            thi = ((thi*180/np.pi + 180) % 360 - 180)*np.pi/180.
            thf = ((thf*180/np.pi) % 360 - 180)*np.pi/180.        
       
        # start = time.time()
        curve = NURBS.Curve()
        """Set degree"""
        curve.degree =  self.degree 
        
        """ Non-free control points"""
        distT = np.linalg.norm(np.array(ptf) - np.array(pti))        
        vi = alpha1
        vf = alpha2
        dxi =  np.cos(gammai)*vi
        dyi =  np.sin(gammai)*vi
        pti2x = pti[0] + dxi*2/3
        pti2y = pti[1] + dyi*2/3
        
        pti2z = pti[2] + np.tan(thi)*vi*2/3
        pti3x = pti2x + dxi*1/3
        pti3y = pti2y + dyi*1/3        
        pti3z = pti2z + np.tan(thi)*vi*1/3
        dxf = np.cos(gammaf)*vf
        dyf = np.sin(gammaf)*vf
        ptf2x = ptf[0] - dxf*2/3
        ptf2y = ptf[1] - dyf*2/3
        ptf2z = ptf[2] - np.tan(thf)*vf*2/3
        ptf3x = ptf2x - dxf*1/3
        ptf3y = ptf2y - dyf*1/3        
        ptf3z = ptf2z - np.tan(thf)*vf*1/3

        pti2 = [ pti2x,  pti2y , pti2z]
        pti3 = [ pti3x,  pti3y , pti3z]
        ptf2 = [ ptf2x,  ptf2y , ptf2z]
        ptf3 = [ ptf3x,  ptf3y , ptf3z]
        
        # Set control points
        ctrlpts = []
        ctrlpts.append(pti)
        ctrlpts.append(pti2)
        ctrlpts.append(pti3)
        for pt in points:
            ctrlpts.append(pt)
        ctrlpts.append(ptf3)
        ctrlpts.append(ptf2)
        ctrlpts.append(ptf)        
        curve.ctrlpts = list(np.array(ctrlpts)*scale)
        
        curve.knotvector = knotvector.generate(curve.degree, len(ctrlpts))        
        """Set evaluation delta"""
        curve.delta = dt 

        curve.weights = weigths
        """ Evaluate curve """
        try:
            curve.evaluate()       
            f = np.array(curve.evalpts)
            fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]            
            # now = time.time()
            # tt = now - start 
            # printD("t = ", tt)
            return fx, fy, fz, curve
        except:
            a = np.zeros((1000,1))
            return a,a,a, curve


   
    def costfunction_nurbs(self, parameters):
        """ return the cost function """           

        """  Create the NURBS curve """
        # each ctrl_point requires 4 parameters: 3 parameters for x,y,z + 1 parameter for weight
        n_ctrl_pts = (len(parameters)-2)//4    
        # printD(f'ctr = {n_ctrl_pts}')     
        points = []   
        for i in range(0,n_ctrl_pts*3,3):    
            # printD(i)            
            points.append([parameters[i], parameters[i+1], parameters[i+2]])
        weigths = []
        weigths.append(1) # P0             
        weigths.append(1) # P1
        weigths.append(1) # P2
        for ii in range(-3,-n_ctrl_pts-3,-1):
            # printD(ii)
            weigths.append(parameters[ii])        
        weigths.append(1) # P_last-2   
        weigths.append(1) # P_last-1
        weigths.append(1) # P_last
        # t = time.time()        
        fx, fy, fz, curve_nurbs = self.nurbs(alpha1=parameters[-1],alpha2=parameters[-2],\
                                             pti=self.pinit, ptf=self.pgoal, gammai=self.yaw_init, \
                                             gammaf=self.yaw_goal, thi=self.pitch_init, thf=self.pitch_goal, \
                                             points=points, weigths=weigths,dt=self.dt, scale=1)
        # t1 = time.time() - t        
        # t = time.time()                
        curve =  np.array([fx,fy, fz]).T
        diff_curve = np.diff(curve, axis=0)  
        norm_diff_curve = np.linalg.norm(diff_curve, axis=1)


        """ Length curve """        
        length = np.sum(norm_diff_curve)
        f1 = length
        

        """  curvature function """        
        rl = np.diff(curve, axis=0)
        norm_rl = np.linalg.norm( rl, axis=1 )[:-1]
        rl2 = np.diff(rl, axis=0)
        
        cross_rl_rl2 = np.cross(rl[:-1], rl2)        
        kappa = np.linalg.norm(cross_rl_rl2, axis=1)/norm_rl**3
        kappa_grad = np.gradient(kappa, edge_order=1)

        kappa_initial = 0
        zero_kappa = np.sum(np.abs(kappa[0:1]- kappa_initial)) + np.sum(np.abs(kappa[-1:]-kappa_initial))
        diff_zero_kappa = np.sum(np.abs(rl[0:1]- kappa_initial)) + np.sum(np.abs(rl[-1:]-kappa_initial)) # initial and final curvature desired to be zero
        f12 = zero_kappa + diff_zero_kappa        
        f2 = np.sum(((kappa)**2))

        """climb angle """
        climb_angle = np.arctan2(diff_curve[:,2], (diff_curve[:,0]**2 + diff_curve[:,1]**2)**0.5)
        grad_climb_angle = np.diff(climb_angle, axis=0)
        grad2_climb_angle = np.diff(grad_climb_angle, axis=0)
        f3 = np.sum(np.power(climb_angle, 2))
        f14 = np.sum(np.power(grad_climb_angle, 2))

        """ initial and final orientation """
        yaw_i = np.arctan2(diff_curve[0,1], diff_curve[0,0])
        pitch_i = np.arctan2(diff_curve[0,2],(diff_curve[0,0]**2 + diff_curve[0,1]**2)**0.5)    
        yaw_f = np.arctan2(diff_curve[-1,1], diff_curve[-1,0])
        pitch_f = np.arctan2(diff_curve[-1,2],(diff_curve[-1,0]**2 + diff_curve[-1,1]**2)**0.5)  
        diff_yi = self.yaw_init - yaw_i
        diff_yi =   ((diff_yi + np.pi) % (2*np.pi) - np.pi)

        diff_yg = self.yaw_goal- yaw_f
        diff_yg =   ((diff_yg + np.pi) % (2*np.pi) - np.pi)

        diff_pi = self.pitch_init - pitch_i
        diff_pi =   ((diff_pi + np.pi) % (2*np.pi) - np.pi)
        
        diff_pg = self.pitch_goal - pitch_f        
        diff_pg =   ((diff_pg + np.pi) % (2*np.pi) - np.pi)
        f4 = (diff_yi)**2 + (diff_yg)**2 + (diff_pi)**2 + (diff_pg)**2
        
        """ penalty climb angle"""
        f5a =  climb_angle[climb_angle > self.thetamax]
        f5b =  climb_angle[climb_angle < self.thetamin]
        f5 = len(f5a) + len(f5b)

        f6 = 0
        if len(f5a):
            f6  = np.sum((f5a)**2) + np.sum((f5b)**2)
        
        f7 = 0
        if len(f5b):
            f7 = np.sum((f5b-self.thetamin)**2)
        
        """ penalty kappa """        
        f8a = kappa[kappa>self.kappamax]        
        f8 = len(f8a)
        f9 = 0
        if f8 > 0:
            f9 = np.sum((f8a-self.kappamax)**2)
        
        """  check collision with elliptical cylinders """      
        d = check_collision_cylinder2line(curve, self.obs_cylinders,self.r)  
        f10 = len(d)        
        if f10 > 0:
            f10 *= (np.sum(1/(d)**2))  
        
        
        if len(self.hole) > 0:
            # Parameters        
            center = self.hole[0]    
            dx = self.hole[1][0]
            dy = self.hole[1][1]
            dz = self.hole[1][2]  # Width and height of the plane and cylinder
            
            

            if self.hole[-1] == 0: #'x'                    
                # Create a meshgrid for the 3D space
                # in this case: 
                # dx is the height of the hole, dy and dz radii            
                condition_in_plane = (fx - self.r >= center[0]-dx/2) & (fx + self.r <= center[0]+dx/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fy-center[1])**2/(dy+self.r/2)**2 + (fz-center[2])**2/(dz+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)

            elif self.hole[-1] == 1: #'y'     
                condition_in_plane = (fy - self.r >= center[1]-dy/2) & (fy + self.r <= center[1]+dy/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fx-center[0])**2/(dx+self.r/2)**2 + (fz-center[2])**2/(dz+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)
   
            elif self.hole[-1] == 2: #'z'        
                condition_in_plane = (fz - self.r >= center[2]-dy/2) & (fz + self.r <= center[2]+dy/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fx-center[0])**2/(dx+self.r/2)**2 + (fy-center[2])**2/(dy+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)
            
            f10b = np.sum(condition_out_cylinder)            
            f10 += 1*(10*f10b + 1000*f10a)
        

        # t3 = time.time() - t
        """ space limit """
        f11 = 0
        if self.alpha[10] > 0:
            f11 = np.sum(fx < -self.space_limit[0] + self.r) + \
                  np.sum(fy < -self.space_limit[1] + self.r) + \
                  np.sum(fx > self.space_limit[0] - self.r) + \
                  np.sum(fy > self.space_limit[1] - self.r) + \
                  np.sum(fz > self.space_limit[2] - self.r) + \
                  np.sum(fz < self.r)  \
                #   np.sum(fz < -self.space_limit[2] + self.r)  \ #  to consider fz <0 possible
                  

        """ topography """
        if self.topography_index > 0:
            if callable(self.topography_z):            
                z_mountain = self.topography_z(fx, fy, self.topography_index)
                # print(len(z_mountain), len(fz))
                f13 = np.sum(z_mountain >= fz - 2*self.r)
        else:
            f13 = 0            
            if len(self.alpha) < 13:
                self.alpha.append(0)
        
        cost_total = self.alpha[0]*f1 + self.alpha[1]*f2+ self.alpha[2]*f3 + self.alpha[3]*f4 +\
                     length*(self.alpha[4]*f5  +\
                     self.alpha[5]*f6 + self.alpha[6]*f7 + \
                     self.alpha[7]*f8+ self.alpha[8]*f9 + self.alpha[9]*f10 +\
                     self.alpha[10]*f11 + self.alpha[12]*f13) + self.alpha[13]*f14#self.alpha[11]*f12
        
        
        g =length*(self.alpha[4]*f5 + self.alpha[5]*f6 + self.alpha[6]*f7 + \
                     self.alpha[7]*f8+ self.alpha[8]*f9 + self.alpha[9]*f10 +\
                     self.alpha[10]*f11 + self.alpha[12]*f13)
        
                
        return cost_total,g==0, g

    
    def costfunction_nurbs_mealpy(self, parameters):        
        # each ctrl_point requires 4 parameters: 3 parameters for x,y,z + 1 parameter for weight
        n_ctrl_pts = (len(parameters)-2)//4             
        points = []   
        for i in range(0,n_ctrl_pts*3,3):    
            # printD(i)            
            points.append([parameters[i], parameters[i+1], parameters[i+2]])
        weigths = []
        weigths.append(1) # P0             
        weigths.append(1) # P1
        weigths.append(1) # P2
        for ii in range(-3,-n_ctrl_pts-3,-1):
            # printD(ii)
            weigths.append(parameters[ii])        
        weigths.append(1) # P_last-2   
        weigths.append(1) # P_last-1
        weigths.append(1) # P_last
        # t = time.time()        
        fx, fy, fz, curve_nurbs = self.nurbs(alpha1=parameters[-1],alpha2=parameters[-2],\
                                             pti=self.pinit, ptf=self.pgoal, gammai=self.yaw_init, \
                                             gammaf=self.yaw_goal, thi=self.pitch_init, thf=self.pitch_goal, \
                                             points=points, weigths=weigths,dt=self.dt, scale=1)
        # t1 = time.time() - t        
        # t = time.time()                
        curve =  np.array([fx,fy, fz]).T
        diff_curve = np.diff(curve, axis=0)  
        norm_diff_curve = np.linalg.norm(diff_curve, axis=1)


        """ Length curve """        
        length = np.sum(norm_diff_curve)
        f1 = length
        

        """  curvature function """        
        rl = np.diff(curve, axis=0)
        norm_rl = np.linalg.norm( rl, axis=1 )[:-1]
        rl2 = np.diff(rl, axis=0)
        
        cross_rl_rl2 = np.cross(rl[:-1], rl2)        
        kappa = np.linalg.norm(cross_rl_rl2, axis=1)/norm_rl**3
        kappa_grad = np.gradient(kappa, edge_order=1)

        kappa_initial = 0
        zero_kappa = np.sum(np.abs(kappa[0:1]- kappa_initial)) + np.sum(np.abs(kappa[-1:]-kappa_initial))
        diff_zero_kappa = np.sum(np.abs(rl[0:1]- kappa_initial)) + np.sum(np.abs(rl[-1:]-kappa_initial)) # initial and final curvature desired to be zero
        f12 = zero_kappa + diff_zero_kappa
        f2 = np.sum(((kappa)**2))

        """climb angle """
        climb_angle = np.arctan2(diff_curve[:,2], (diff_curve[:,0]**2 + diff_curve[:,1]**2)**0.5)
        grad_climb_angle = np.diff(climb_angle, axis=0)
        grad2_climb_angle = np.diff(grad_climb_angle, axis=0)
        f3 = np.sum(np.power(climb_angle, 2))
        f14 = np.sum(np.power(grad_climb_angle, 2))        

        """ initial and final orientation """
        yaw_i = np.arctan2(diff_curve[0,1], diff_curve[0,0])
        pitch_i = np.arctan2(diff_curve[0,2],(diff_curve[0,0]**2 + diff_curve[0,1]**2)**0.5)    
        yaw_f = np.arctan2(diff_curve[-1,1], diff_curve[-1,0])
        pitch_f = np.arctan2(diff_curve[-1,2],(diff_curve[-1,0]**2 + diff_curve[-1,1]**2)**0.5)  
        diff_yi = self.yaw_init - yaw_i
        diff_yi =   ((diff_yi + np.pi) % (2*np.pi) - np.pi)

        diff_yg = self.yaw_goal- yaw_f
        diff_yg =   ((diff_yg + np.pi) % (2*np.pi) - np.pi)

        diff_pi = self.pitch_init - pitch_i
        diff_pi =   ((diff_pi + np.pi) % (2*np.pi) - np.pi)
        
        diff_pg = self.pitch_goal - pitch_f        
        diff_pg =   ((diff_pg + np.pi) % (2*np.pi) - np.pi)
        f4 = (diff_yi)**2 + (diff_yg)**2 + (diff_pi)**2 + (diff_pg)**2
        
        """ penalty climb angle"""
        f5a =  climb_angle[climb_angle > self.thetamax]
        f5b =  climb_angle[climb_angle < self.thetamin]
        f5 = len(f5a) + len(f5b)

        f6 = 0
        if len(f5a):
            f6  = np.sum((f5a)**2) + np.sum((f5b)**2)
            # f6 = np.sum(((climb_angle[climb_angle > self.thetamax*1.4])-self.thetamax)**2) + np.sum((climb_angle[climb_angle < self.thetamin*1.4]-self.thetamax)**2)
        
        f7 = 0
        if len(f5b):
            f7 = np.sum((f5b-self.thetamin)**2)
        
        """ penalty kappa """        
        f8a = kappa[kappa>self.kappamax]        
        f8 = len(f8a)
        f9 = 0
        if f8 > 0:
            f9 = np.sum((f8a-self.kappamax)**2)
        
        
        
        # t = time.time()        
        """  check collision with elliptical cylinders """      
        d = check_collision_cylinder2line(curve, self.obs_cylinders,self.r)  
        f10 = len(d)        
        if f10 > 0:
            f10 *= (np.sum(1/(d)**2))  
        
        
        if len(self.hole) > 0:
            # Parameters        
            center = self.hole[0]    
            dx = self.hole[1][0]
            dy = self.hole[1][1]
            dz = self.hole[1][2]  # Width and height of the plane and cylinder
            # print(d)
            

            if self.hole[-1] == 0: #'x'                    
                # Create a meshgrid for the 3D space
                # in this case: 
                # dx is the height of the hole, dy and dz radii            
                condition_in_plane = (fx - self.r >= center[0]-dx/2) & (fx + self.r <= center[0]+dx/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fy-center[1])**2/(dy+self.r/2)**2 + (fz-center[2])**2/(dz+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)

            elif self.hole[-1] == 1: #'y'     
                condition_in_plane = (fy - self.r >= center[1]-dy/2) & (fy + self.r <= center[1]+dy/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fx-center[0])**2/(dx+self.r/2)**2 + (fz-center[2])**2/(dz+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)
   
            elif self.hole[-1] == 2: #'z'        
                condition_in_plane = (fz - self.r >= center[2]-dy/2) & (fz + self.r <= center[2]+dy/2)  #& (fx < -self.space_limit[0] + self.r) & (fx > self.space_limit[0] - self.r)    
                dist_in_cylinder =  (fx-center[0])**2/(dx+self.r/2)**2 + (fy-center[2])**2/(dy+self.r/2)**2
                condition_in_cylinder = (dist_in_cylinder <= 1) #& (fx >= center[0]-dx/2) & (fx <= center[2]+dx/2)        
                condition_out_cylinder = (~condition_in_cylinder)  &  condition_in_plane
                f10a = np.sum(dist_in_cylinder[condition_out_cylinder]) #+ 1/(np.sum(dist_in_cylinder[condition_in_cylinder])+1e-18)
            
            f10b = np.sum(condition_out_cylinder)            
            f10 += 1*(10*f10b + 1000*f10a)
        
        """ space limit """
        f11 = 0
        if self.alpha[10] > 0:
            f11 = np.sum(fx < -self.space_limit[0] + self.r) + \
                  np.sum(fy < -self.space_limit[1] + self.r) + \
                  np.sum(fz < self.r) + \
                  np.sum(fx > self.space_limit[0] - self.r) + \
                  np.sum(fy > self.space_limit[1] - self.r) + \
                  np.sum(fz > self.space_limit[2] - self.r)
        

        """ topography """
        if callable(self.topography_z):            
            z_mountain = self.topography_z(fx, fy, self.topography_index)            
            f13 = np.sum(z_mountain >= fz - 2*self.r)
        else:
            f13 = 0
            if len(self.alpha) < 13:
                self.alpha.append(0)

        cost_total = self.alpha[0]*f1 + self.alpha[1]*f2+ self.alpha[2]*f3 + self.alpha[3]*f4 +\
                     length*(self.alpha[4]*f5  +\
                     self.alpha[5]*f6 + self.alpha[6]*f7 + \
                     self.alpha[7]*f8+ self.alpha[8]*f9 + self.alpha[9]*f10 +\
                     self.alpha[10]*f11 + self.alpha[12]*f13) + self.alpha[13]*f14#self.alpha[11]*f12
        
        
        g =length*(self.alpha[4]*f5 + self.alpha[5]*f6 + self.alpha[6]*f7 + \
                     self.alpha[7]*f8+ self.alpha[8]*f9 + self.alpha[9]*f10 +\
                     self.alpha[10]*f11 + self.alpha[12]*f13)
        
        return cost_total

    def get_curve_points(self, path_params, problem_params, dt=None, scale=1):
        """
        input: path_params, problem_params
        return: fx, fy (curve points x, y)
        """        
        pinit = problem_params["pinit"]
        pgoal = problem_params["pgoal"]
        yaw_init = problem_params["yaw_init"]
        yaw_goal = problem_params["yaw_goal"]
        pitch_init = problem_params["pitch_init"]
        pitch_goal = problem_params["pitch_goal"]
        if dt is None:
            dt = problem_params['dt']
        
        
        
        if self.type in "DE-NURBS":
            

            n_ctrl_pts = (len(path_params)-2)//4 # each ctrl_point requires 4 parameters: 3 parameters for x,y,z + 1 parameter for weight
# 
            points = []   
            for i in range(0,n_ctrl_pts*3,3):                
                points.append([path_params[i], path_params[i+1], path_params[i+2]])
            weigths = []
            weigths.append(1) # P0             
            weigths.append(1) # P1
            weigths.append(1) # P2

            for ii in range(-3,-n_ctrl_pts-3    ,-1):
                weigths.append(path_params[ii])                
            
            weigths.append(1) # P_last-2   
            weigths.append(1) # P_last-1
            weigths.append(1) # P_last            
            fx, fy, fz, curve_nurbs = self.nurbs(alpha1=path_params[-1],alpha2=path_params[-2], pti=pinit, ptf=pgoal, \
                                                 gammai=yaw_init, gammaf=yaw_goal, thi=pitch_init, thf=pitch_goal, \
                                                 points=points, weigths=weigths, dt=dt, scale=scale)
            
        return fx, fy, fz, curve_nurbs
    
    
    def plot_curve(self, path_params, problem_params, fig=None, viewpoint = (16, 45)):
        fx, fy, fz, _ = self.get_curve_points(path_params = path_params,problem_params=problem_params)
        curve = np.array([fx, fy, fz]).T*problem_params['scale']
        if fig is None:
            fig = plt.figure(figsize=(40,16))

        yaw_init = problem_params['yaw_init']
        yaw_goal = problem_params['yaw_goal']
        pitch_init = problem_params['pitch_init']
        pitch_goal = problem_params['pitch_goal']
        pinit = np.array(problem_params['pinit'])*problem_params['scale']
        pgoal = np.array(problem_params['pgoal'])*problem_params['scale']

        ax = fig.add_subplot(121)
        ax.plot(curve.T[0], curve.T[1], '.b', markersize=12)
        ax.plot(curve.T[0], curve.T[1], '-m',  linewidth=4)
        ax.quiver(pinit[0], pinit[1], np.cos(yaw_init), np.sin(yaw_init), color='indigo')
        ax.quiver(pgoal[0], pgoal[1], np.cos(yaw_goal), np.sin(yaw_goal), color='indigo')

        for iob, ob in enumerate(problem_params['obs_cylinders']):
            ob = np.array(ob)*problem_params['scale']
            plot_ellipse([ob[0], ob[1]], rx=ob[2], ry=ob[3], ax=ax, color='green')
            ax.text(ob[0], ob[1],f'ob{iob}', color='k', fontsize=12)

        ax2 = fig.add_subplot(122, projection='3d')
        for ob in problem_params['obs_cylinders']:
            ob = np.array(ob)*problem_params['scale']
            plot_cylinder([ob[0], ob[1],0], rx=ob[2], ry=ob[3], height=2*problem_params['scale'], ax=ax2, color_surface='green', \
                            color_wireframe='green', line_wireframe=0.05, alpha=0.3)
            
        ax2.plot(curve.T[0], curve.T[1], curve.T[2],'-m',  linewidth=4)
        ax2.quiver(pinit[0], pinit[1], pinit[2], np.cos(yaw_init), np.sin(yaw_init), np.tan(pitch_init), length=0.001*problem_params['scale'], normalize=True, color='indigo', linewidth=4)
        ax2.quiver(pgoal[0], pgoal[1], pgoal[2], np.cos(yaw_goal), np.sin(yaw_goal), np.tan(pitch_goal), length=0.001*problem_params['scale'],  normalize=True, color='indigo', linewidth=4)
        ax.axis('equal')
            
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')

        ax2.view_init(viewpoint[0], viewpoint[1]) 
        ax2.set_xlabel(r'$x$ (m)')
        ax2.set_ylabel(r'$y$ (m)')
        ax2.set_zlabel(r'$z$ (m)')
        fig.tight_layout()
        return fig, curve