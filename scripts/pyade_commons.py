import numpy as np
from typing import Callable, Union, List, Tuple, Any
import time


def sort_indiviuals(population, return_func):
    
    return_func_fe, return_func_un, indexes_fe, indexes_un = divide_feasibility(return_func)

    
    fitness = return_func[:,0].copy()
    fitness_fe = return_func_fe[:,0].copy()
    constraints_fe = return_func_fe[:,2].copy()
    
    fitness_un = return_func_un[:,0].copy()
    constraints_un = return_func_un[:, 2].copy()    
    # print("\ntem que ser zero", constraints_un[-1,1], "se g = 0", len(population_un) )

    population_fe = population[indexes_fe].copy()
    sorted_indices_fe = np.argsort((fitness_fe))
    population_fe = population_fe[sorted_indices_fe]
    fitness_fe = fitness_fe[sorted_indices_fe]
    

    population_un = population[indexes_un].copy()
    sorted_indices_un = np.argsort(constraints_un)
    constraints_un = constraints_un[sorted_indices_un]
    population_un = population_un[sorted_indices_un]
    fitness_un = fitness_un[sorted_indices_un]  
    
    

    if len(population_fe) == 0: # only unfeasible solutions      
        
        best_solution = population_un[0]
        best_fitness = fitness_un[0]
        best_constraint = constraints_un[0] 
        # print(len(population_un), best_fitness, np.min(return_func[:,2]), np.min(fitness_un)) 
        
        return population_un, fitness_un, 0, best_solution, best_fitness, best_constraint
    
    elif len(population_un) == 0: #only feasible solutions       
        best_solution = population_fe[0]
        best_fitness = fitness_fe[0]
        best_constraint = 0
        return population_fe, fitness_fe, len(population_fe), best_solution, best_fitness, best_constraint
    
    elif len(population_fe) < len(population):
        new_population = np.vstack((population_fe, population_un))
        fitness = np.append(fitness_fe, fitness_un)
        sorted_indices = np.lexsort((return_func[:, 0], return_func[:, 2])) #sort by the minimum g and after min f
        best_solution = population[sorted_indices[0]]
        best_fitness = fitness[sorted_indices[0]]
        best_constraint = 0
        return new_population, fitness, len(population_fe), best_solution, best_fitness, best_constraint


    else:

        population = np.vstack((population_fe, population_un))
        # print(f'\n\nsort pop before =  {fitness_un.shape}, {fitness_fe.shape}')
        fitness = np.append(fitness_fe, fitness_un)
        best_solution = population[0]
        best_fitness = fitness[0]
        best_constraint = 0

        # print(f'sort pop after= {population.shape}')
        return population, fitness, len(population_fe), best_solution, best_fitness, best_constraint

def ranking_selection(i, N):
    p = [((N - j) / N) for j in range(1, N + 1)]

    r1 = np.random.randint(1, N)
    while np.random.random() > p[r1 - 1] or r1 == i:
        r1 = np.random.randint(1, N)

    r2 = np.random.randint(1, N)
    while np.random.random() > p[r2 - 1] or r2 == r1 or r2 == i:
        r2 = np.random.randint(1, N)

    r3 = np.random.randint(1, N)
    while r3 == r1 or r3 == r2 or r3 == i:
        r3 = np.random.randint(1, N)

    return r1, r2, r3


def divide_feasibility(np_array):
    feasible_mask = (np_array[:, 1].astype(int) == 1)
    feasible_solutions = np_array[feasible_mask]
    unfeasible_solutions = np_array[~feasible_mask]

    feasible_indices = np.where(feasible_mask)[0]
    unfeasible_indices = np.where(~feasible_mask)[0]

    return feasible_solutions, unfeasible_solutions, feasible_indices, unfeasible_indices

def keep_bounds(population: np.ndarray,
                bounds: np.ndarray) -> np.ndarray:
    """
    Constrains the population to its proper limits.
    Any value outside its bounded ranged is clipped.
    :param population: Current population that may not be constrained.
    :type population: np.ndarray
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype np.ndarray
    :return: Population constrained within its bounds.
    """
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    return np.clip(population, minimum, maximum)


def init_population(population_size: int, individual_size: int,
                    bounds: Union[np.ndarray, list]) -> np.ndarray:
    """
    Creates a random population within its constrained bounds.
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens.
    :type individual_size: int
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Initialized population.
    """
    minimum = np.array([bound[0] for bound in bounds])
    maximum = np.array([bound[1] for bound in bounds])
    range = maximum-minimum
    population = np.random.rand(population_size, individual_size) * range + minimum

    return keep_bounds(population, bounds)


    # population = np.random.randn(population_size, individual_size)
    # return keep_bounds(population, bounds)


def apply_fitness(population: np.ndarray,
                  func: Callable[[np.ndarray], float],
                  opts: Any) -> np.ndarray:
    """
    Applies the given fitness function to each individual of the population.
    :param population: Population to apply the current fitness function.
    :type population: np.ndarray
    :param func: Function that is used to calculate the fitness.
    :type func: np.ndarray
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :rtype np.ndarray
    :return: Numpy array of fitness for each individual.
    """

    # if opts is None:     
        
    #     return_func = np.array([func(individual) for individual in population])   
    #     print(return_func.shape, population.shape)
    #     return np.hstack((return_func,population))
    # else:
    #     return_func = np.array([func(individual, opts) for individual in population])
    #     print(return_func.shape, population.shape)
    #     return np.hstack((return_func,population))
    if opts is None:        
        return np.array([func(individual) for individual in population])
    else:
        return np.array([func(individual, opts) for individual in population])


def __parents_choice(population: np.ndarray, n_parents: int) -> np.ndarray:
    pob_size = population.shape[0]
    choices = np.indices((pob_size, pob_size))[1]
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    choices = choices[mask].reshape(pob_size, pob_size - 1)
    parents = np.array([np.random.choice(row, n_parents, replace=False) for row in choices])

    return parents


def binary_mutation(population: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    """
    Calculate the binary mutation of the population. For each individual (n),
    3 random parents (x,y,z) are selected. The parents are guaranteed to not
    be in the same position than the original. New individual are created by
    n = z + F * (x-y)
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents = __parents_choice(population, 3)
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated += population[parents[:, 2]]

    return keep_bounds(mutated, bounds)


def current_to_best_2_binary_mutation(population: np.ndarray,
                                      population_fitness: np.ndarray,
                                      f: Union[int, float],
                                      bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to best/2/bin" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 3:
        return population

    # 1. We find the best parent
    best_index = np.argmin(population_fitness)

    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f * (population[best_index] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)


def current_to_pbest_mutation(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to p-best" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))

    p_best = np.array(p_best)
    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)

def cde_mutation(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f,                              
                              bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the CDE. This is
    V_{i, G} = X_{r1, G} + F * (X_{r2. G} - X_{r3, G})
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    mutated = []
    for i in range(len(population)):
        # print(f'pp = {population.shape}, {population_fitness.shape}')
        r1, r2, r3 = ranking_selection(i, len(population))

        # print(r1, r2, r3, f)

        xr1 = population[r1]
        xr2 = population[r2]
        xr3 = population[r3]

        f_xr1 = population_fitness[r1]
        f_xr2 = population_fitness[r2]
        f_xr3 = population_fitness[r3]
        
    
        if f_xr1 <= f_xr2 and f_xr1 <= f_xr3:
            mutated_i = xr1 + f * (xr2 - xr3)
        elif f_xr2 <= f_xr1 and f_xr2 <= f_xr3:
            mutated_i = xr2 + f * (xr1 - xr3)
        else:
            mutated_i = xr3 + f * (xr1 - xr2)

        mutated.append(mutated_i)
    mutated = np.array(mutated)    

    return keep_bounds(mutated, bounds)


def current_to_rand_1_mutation(population: np.ndarray,
                              population_fitness: np.ndarray,
                              k: List[float],
                              f: List[float],
                              bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to rand/1" mutation. This is
    U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G} + F * (X_{r2. G} - X_{r3, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents = __parents_choice(population, 3)
    # 2. Apply the formula to each set of parents
    mutated = k * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])

    return keep_bounds(mutated, bounds)


def current_to_pbest_weighted_mutation(population: np.ndarray,
                                       population_fitness: np.ndarray,
                                       f: np.ndarray,
                                       f_w: np.ndarray,
                                       p: float,
                                       bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to p-best weighted" mutation. This is
    V_{i, G} = X_{i, G} + F_w * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: np.ndarray
    :param f_w: NumPy Array with the weighted version of the mutation array
    :type f_w: np.ndarray
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    best_index = np.argsort(population_fitness)[:max(2, round(p*len(population)))]

    p_best = np.random.choice(best_index, len(population))
    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f_w * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)


def crossover(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    """
    Crosses gens from individuals of the last generation and the mutated ones
    based on the crossover rate. Binary crossover
    :param population: Previous generation population.
    :type population: np.ndarray
    :param mutated: Mutated population.
    :type population: np.ndarray
    :param cr: Crossover rate. Must be in [0,1].
    :type population: Union[int, float]
    :rtype: np.ndarray
    :return: Current generation population.
    """
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1])
    chosen[j_rand::population.shape[1]] = 0
    return np.where(chosen <= cr, mutated, population)

def exponential_crossover(population: np.ndarray, mutated: np.ndarray,
                          cr: Union[int, float]) -> np.ndarray:
    """
        Crosses gens from individuals of the last generation and the mutated ones
        based on the crossover rate. Exponential crossover.
        :param population: Previous generation population.
        :type population: np.ndarray
        :param mutated: Mutated population.
        :type population: np.ndarray
        :param cr: Crossover rate. Must be in [0,1].
        :type population: Union[int, float]
        :rtype: np.ndarray
        :return: Current generation population.
    """
    if type(cr) is int or float:
        cr = np.array([cr] * len(population))
    else:
        cr = cr.flatten()

    def __exponential_crossover_1(x: np.ndarray, y: np.ndarray, cr: Union[int, float]) -> np.ndarray:
        z = x.copy()
        n = len(x)
        k = np.random.randint(0, n)
        j = k
        l = 0
        while True:
            z[j] = y[j]
            j = (j + 1) % n
            l += 1
            if np.random.randn() >= cr or l == n:
                return z

    return np.array([__exponential_crossover_1(population[i], mutated[i], cr.flatten()[i]) for i in range(len(population))])


def current_to_pbest_constraints_mutation(population, population_fe, archive,\
                                        fitness, fitness_fe,return_func,\
                                        f, f_fe, p, p_fe, bounds):
    """
    Calculates the mutation of the entire population based on the
    "current to p-best" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population
    
    # We find the best parent between all solutions
    p_best = []
    best_index = np.lexsort((return_func[:, 0], return_func[:, 2]))
    for p_i in p:        
        best_index = best_index[:max(2, int(round(p_i*len(population))))]
        # best_index = np.argsort(fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))
    p_best = np.array(p_best)
    
    # We find the best parent between feasible solutions
    p_best_fe = []
    if len(p_fe) > 0.0*len(population):
        # print("pfe", len(p_fe), len(population_fe))
        best_index_fe = np.argsort(fitness_fe)[:max(2, int(round(p_i*len(population_fe))))]
        for p_i in p_fe:
            # print(p_i)            
            p_best_fe.append(np.random.choice(best_index_fe))    
    p_best_fe = np.array(p_best_fe)
    
    # We choose one random parents from all solutions
    p_U_archive = population.copy()
    if len(archive)> 0:
        # print('aqui', p_U_archive.shape, np.array(archive).shape)
        p_U_archive = np.vstack((p_U_archive,archive))        
        random_indices = np.random.choice(p_U_archive.shape[0], size=population.shape[0], replace=False)
        p_U_archive = p_U_archive[random_indices]
    parents = __parents_choice(p_U_archive, 2)
    
    # print(p_U_archive[parents[:, 0]].shape, p_U_archive[parents[:, 1]].shape)
    # mutated = population + f * (population[p_best] - population)
    # mutated += f * (p_U_archive[parents[:, 0]] - p_U_archive[parents[:, 1]])
    # print(f'{0.1*len(population)}, {len(p_fe)}, {len(population_fe)}')
    mutated = population + f * (p_U_archive[parents[:, 0]] - p_U_archive[parents[:, 1]])
    if len(p_fe) > 0.0*len(population):
        # print(f'fe = {f_fe}')
        extended_array = np.tile(population_fe[p_best_fe], (population.shape[0] // population_fe[p_best_fe].shape[0] + 1, 1))
        extended_array = extended_array[:population.shape[0], :population.shape[1]]
        # print(f'{extended_array.shape}, {population.shape}, {population_fe[p_best_fe].shape}')
        mutated +=  f*(extended_array - population)
    else:        
        mutated += f * (population[p_best] - population)

    # print(f'f={np.max(f)}')
    return keep_bounds(mutated, bounds)


def current_to_pbest_archive_mutation(population: np.ndarray,
                                      archive: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray):
    """
    Calculates the mutation of the entire population based on the
    "current to p-best" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))

    p_best = np.array(p_best)
    # 2. We choose two random parents
    p_U_archive = population.copy()
    if len(archive)> 0:
        # print('aqui', p_U_archive.shape, np.array(archive).shape)
        p_U_archive = np.vstack((p_U_archive,archive))
        random_indices = np.random.choice(p_U_archive.shape[0], size=population.shape[0], replace=False)
        p_U_archive = p_U_archive[random_indices]
        # p_U_archive = np.random.choice(p_U_archive, size=(population.shape[0],population.shape[1]))
        # print(p_U_archive.shape, np.array(archive).shape)
    parents = __parents_choice(p_U_archive, 2)
    # print(p_U_archive[parents[:, 0]].shape, p_U_archive[parents[:, 1]].shape)
    mutated = population + f * (population[p_best] - population)
    mutated += f * (p_U_archive[parents[:, 0]] - p_U_archive[parents[:, 1]])

    return keep_bounds(mutated, bounds)

def current_to_pbest_mean_archive_mutation(population: np.ndarray,
                                      archive: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray):
    """
    Calculates the mutation of the entire population based on the
    "current to p-best" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))

    p_best = np.array(p_best)
    # 2. We choose two random parents
    p_U_archive = population.copy()
    if len(archive)> 0:
        # print('aqui', p_U_archive.shape, np.array(archive).shape)
        p_U_archive = np.vstack((p_U_archive,archive))
        random_indices = np.random.choice(p_U_archive.shape[0], size=population.shape[0], replace=False)
        p_U_archive = p_U_archive[random_indices]
        # p_U_archive = np.random.choice(p_U_archive, size=(population.shape[0],population.shape[1]))
        # print(p_U_archive.shape, np.array(archive).shape)
    parents = __parents_choice(p_U_archive, 1)
    mean_vector = np.mean(p_U_archive, axis=0)
    # print(p_U_archive[parents[:, 0]].shape, p_U_archive[parents[:, 1]].shape)
    random_vector = np.random.randn(len(population[p_best]),1)
    # random_vector = keep_bounds(random_vector, bounds)    
    mutated = population + f * (population[p_best] - population)
    mutated += f * (p_U_archive[parents[:, 0]] - random_vector)

    return keep_bounds(mutated, bounds)


# def keep_bounds(population: np.ndarray,
#                 bounds: np.ndarray) -> np.ndarray:
#     """
#     Constrains the population to its proper limits.
#     Any value outside its bounded ranged is clipped.
#     :param population: Current population that may not be constrained.
#     :type population: np.ndarray
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype np.ndarray
#     :return: Population constrained within its bounds.
#     """
#     minimum = [bound[0] for bound in bounds]
#     maximum = [bound[1] for bound in bounds]
#     return np.clip(population, minimum, maximum)


# def init_population(population_size: int, individual_size: int,
#                     bounds: Union[np.ndarray, list]) -> np.ndarray:
#     """
#     Creates a random population within its constrained bounds.
#     :param population_size: Number of individuals desired in the population.
#     :type population_size: int
#     :param individual_size: Number of features/gens.
#     :type individual_size: int
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: Union[np.ndarray, list]
#     :rtype: np.ndarray
#     :return: Initialized population.
#     """
#     # min_x = bounds[:,0]
#     # max_x = bounds[:,1]
#     # population = min_x + (max_x - min_x)/2*np.random.normal(0, 1,size=(population_size, individual_size))
#     population = np.random.randn(population_size, individual_size)
#     # print(population.shape)
#     return keep_bounds(population, bounds)

def selection_constraints(population: np.ndarray, new_population: np.ndarray,
              return_func: np.ndarray, c_return_func: np.ndarray,
              return_indexes: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Selects the best individuals based on their fitness.
    :param population: Last generation population.
    :type population: np.ndarray
    :param new_population: Current generation population.
    :type new_population: np.ndarray
    :param fitness: Last generation fitness.
    :type fitness: np.ndarray
    :param new_fitness: Current generation fitness
    :param return_indexes: When active the function also returns the individual indexes that have been modified
    :type return_indexes: bool
    :rtype: ndarray
    :return: The selection of the best of previous generation
     and mutated individual for the entire population and optionally, the indexes changed
    """
    # Extract fitness and penalties from 'return_func' and 'c_return_func'
    fitness = return_func[:, 0].copy()
    penalties = return_func[:, 2].copy()
    new_fitness = c_return_func[:, 0].copy()
    new_penalties = c_return_func[:, 2].copy()

    # Create boolean masks for the three conditions
    condition1 = (new_fitness < fitness) & (new_penalties == 0) & (penalties == 0)
    condition2 = (penalties > new_penalties)&(penalties > 0) & (new_penalties >= 0)
    # condition2 = (~condition1)&(condition2a)
    # condition3 = (~condition1)&(~condition2a)&(new_penalties == 0 ) & (penalties > 0) 

    # Combine the conditions with logical OR to find all elements that satisfy any of the conditions
    selected_indices = np.array(np.where(condition1 | condition2)[0])    

    # Update 'population' for the selected indices using boolean indexing
    population[selected_indices] = new_population[selected_indices].copy()

    # if np.any(np.array(penalties[selected_indices]) - np.array(new_penalties[selected_indices]) < 0):
    #     print(f'!!!!!!!!!!penal !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
    #     time.sleep(10)

    # Optionally return the indexes
    if return_indexes:
        return population, selected_indices
    else:
        return population
    # indexes = []    
    # for id, old, new in zip(range(len(return_func)), return_func, c_return_func):
    #     fitness, _, penalties = old
    #     new_fitness, _, new_penalties = new
    #     # print(f'id={id}, f{fitness:.2E}, nf{new_fitness:.2E}, g{penalties:.4E}, ng{new_penalties:.4E}')
    #     if ((new_fitness <= fitness) and new_penalties==0 and penalties==0):
    #         # print('choice 1')
    #         indexes.append(id)   
    #         population[id] = new_population[id]
    #     elif ((new_penalties < penalties) and penalties>0):
    #         # print('choice 2')
    #         indexes.append(id)   
    #         population[id] = new_population[id]
    #     else:     
    #         # indexes.append(id)   
    #         # print('choice 3')    
    #         pass

    # if return_indexes:
    #     return population, indexes
    # else:
    #     return population
    



    # c_return_func_fe, c_return_func_un, indexes_fe, indexes_un = divide_feasibility(c_return_func)

    # sorted_array = my_array[np.lexsort((my_array[:, 0], my_array[:, 2]))]
    # indexes = np.where(fitness > new_fitness)[0]



    # indexes = np.where(fitness > new_fitness)[0]
    # population[indexes] = new_population[indexes]
    # if return_indexes:
    #     return population, indexes
    # else:
    #     return population
    
def selection(population: np.ndarray, new_population: np.ndarray,
              fitness: np.ndarray, new_fitness: np.ndarray,
              return_indexes: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Selects the best individuals based on their fitness.
    :param population: Last generation population.
    :type population: np.ndarray
    :param new_population: Current generation population.
    :type new_population: np.ndarray
    :param fitness: Last generation fitness.
    :type fitness: np.ndarray
    :param new_fitness: Current generation fitness
    :param return_indexes: When active the function also returns the individual indexes that have been modified
    :type return_indexes: bool
    :rtype: ndarray
    :return: The selection of the best of previous generation
     and mutated individual for the entire population and optionally, the indexes changed
    """
    indexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    if return_indexes:
        return population, indexes
    else:
        return population
