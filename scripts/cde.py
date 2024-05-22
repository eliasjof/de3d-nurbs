#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CDE (Constrained Differential Evolution)
X. Yu, C. Li, J. Zhou, A constrained differential evolution algorithm to solve uav path planning in disaster scenarios, Knowledge-Based
Systems 204 (2020) 106209

This library is based on the code obtained from PyADE: https://github.com/xKuZz/pyade/tree/master

Author: Elias J R Freitas
Date Created: 2023
Python Version: >3.8

"""

from __debug import *
from __experiment_setup import SAVE

import numpy as np
import pyade_commons
from typing import Callable, Union, Dict, Any

experiment_data = []


def get_default_params(dim: int) -> dict:
    """
    Returns the default parameters of the Differential Evolution Algorithm
    :param dim: Size of the problem (or individual).
    :type dim: int
    :return: Dict with the default parameters of the Differential
    Evolution Algorithm.
    :rtype dict
    """
    return {'callback': None, 'max_evals': 10000 * dim, 'seed': None, 'cross': 'bin',
            'f': 0.5, 'cr': 0.95, 'individual_size': dim, 'population_size': 18 * dim, 'opts': None}


def apply(population_size: int, individual_size: int, f: Union[float, int],
          cr: Union[float, int], bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          callback: Callable[[Dict], Any],
          cross: str,
          max_evals: int, seed: Union[int, None]):
    """
    Applies the standard differential evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param f: Mutation parameter. Must be in [0, 2].
    :type f: Union[float, int]
    :param cr: Crossover Ratio. Must be in [0, 1].
    :type cr: Union[float, int]
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param cross: Indicates whether to use the binary crossover('bin') or the exponential crossover('exp').
    :type cross: str
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """
    # 0. Check parameters are valid
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if (type(f) is not int and type(f) is not float) or not 0 <= f <= 2:
        raise ValueError("f (mutation parameter) must be a "
                         "real number in [0,2].")

    if (type(cr) is not int and type(cr) is not float) or not 0 <= cr <= 1:
        raise ValueError("cr (crossover ratio) must be a "
                         "real number in [0,1].")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(cross) is not str and cross not in ['bin', 'exp']:
        raise ValueError("cross must be a string and must be one of \'bin\' or \'cross\'")
    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # 1. Initialization
    np.random.seed(seed)
    population = pyade_commons.init_population(population_size,
                                               individual_size, bounds)
    
    # print(f'pop init = {population.shape}, {len(population)}')
    
    return_func = pyade_commons.apply_fitness(population, func, opts)
    return_func = np.array(return_func)
    num_evals = population_size
    
    
    cr_inicial = 0.4
    cr_final = 0.7
    f_inicial = 1.0
    f_final = 0.7

    cr_incremento = (cr_final - cr_inicial) / max_evals
    f_decremento = (f_inicial - f_final) / max_evals

    
        

    population, fitness, population_size_fe, _, _, _ = pyade_commons.sort_indiviuals(population, return_func)
    current_generation = 0
    
    while num_evals < max_evals:
        

        # cr_current = cr_inicial + num_evals * cr_incremento
        # f_current = f_inicial - num_evals * f_decremento

        cr_current = cr
        f_current = f

        mutated = pyade_commons.cde_mutation(population, fitness, f_current, bounds)        
        crossed = pyade_commons.crossover(population, mutated, cr_current)   

        c_return_func = pyade_commons.apply_fitness(crossed, func, opts)
        c_return_func = np.array(c_return_func)
        num_evals += population_size

        c_fitness = c_return_func[:,0]
        population, indexes = pyade_commons.selection_constraints(population, crossed,
                                                        return_func, c_return_func, return_indexes=True)
        return_func[indexes] = c_return_func[indexes].copy()
        fitness[indexes] = c_fitness[indexes].copy()

        population, fitness, population_size_fe, best_solution, best_fitness, best_constraint = pyade_commons.sort_indiviuals(population, return_func)
        

        if callback is not None:
            callback(**(locals()))
        current_generation += 1
        

        printD(f"\r Num_evals CDE: {num_evals}, gen:{current_generation}, s:{population_size}, sfe:{population_size_fe}, f:{best_fitness:.4f}, mf:{np.mean(fitness):.2f}, g={best_constraint:.2f}                  ",end="")#, 

    
    return best_solution, best_fitness, experiment_data
