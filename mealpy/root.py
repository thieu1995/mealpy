#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:58, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, ptp, median
from numpy import ndarray, array, min, sum, ceil, multiply, mean, sqrt, sign, ones, dot
from numpy.random import uniform, random, normal, choice
from numpy import max as np_max
from math import gamma
from copy import deepcopy


class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    ## Assumption the A solution with format: [position, [target, [obj1, obj2, ...]]]
    ID_POS = 0  # Index of position/location of solution/agent
    ID_FIT = 1  # Index of fitness value of solution/agent

    ID_TAR = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in fitness

    ## To get the position, fitness wrapper, target and obj list
    ##      A[self.ID_POS]                  --> Return: position
    ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
    ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
    ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]


    EPSILON = 10E-10

    DEFAULT_BATCH_IDEA = False
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_LB = -1
    DEFAULT_UB = 1

    def __init__(self, obj_func=None, lb=None, ub=None, minmax="min", verbose=True, kwargs=None):
        """
        Parameters
        ----------
        obj_func : function
        lb : list
        ub : list
        minmax: "min" or "max" problem
        verbose : bool
        """
        if kwargs is None:
            kwargs = {}
        self.problem_size, self.lb, self.ub, self.batch_size, self.batch_idea = None, None, None, None, None
        self.n_objs, self.obj_weight, self.multi_objs, self.obj_is_list = None, None, None, True
        self.verbose = verbose
        self.minmax = minmax
        self.obj_func = obj_func
        self.__check_parameters__(lb, ub, kwargs)
        self.__check_optional_parameters__(kwargs)
        self.__check_objective_function__(kwargs)
        self.epoch, self.pop_size = None, None
        self.solution, self.loss_train = None, []
        self.g_best_list = []           # List of global best solution found so far in all previous generations
        self.c_best_list = []           # List of current best solution in each previous generations
        self.epoch_time_list = []       # List of runtime for each generation
        self.g_fit_best_list = []       # List of global best fitness found so far in all previous generations
        self.c_fit_best_list = []       # List of current best fitness in each previous generations
        self.pop_list = []              # List of population in each generations
        self.div_list = None            # List of diversity of swarm in all generations
        self.exploit_list = None        # List of exploitation percentages for all generations
        self.explore_list = None        # List of exploration percentanges for all generations

    def __check_parameters__(self, lb, ub, kwargs):
        if (lb is None) or (ub is None):
            if "problem_size" in kwargs:
                print(f"Default lb={self.DEFAULT_LB}, ub={self.DEFAULT_UB}.")
                self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                self.lb = self.DEFAULT_LB * ones(self.problem_size)
                self.ub = self.DEFAULT_UB * ones(self.problem_size)
            else:
                print("If lb, ub are undefined, then you must set problem size to be an integer.")
                exit(0)
        else:
            if isinstance(lb, list) and isinstance(ub, list):
                if len(lb) == len(ub):
                    if len(lb) == 0:
                        if "problem_size" in kwargs:
                            print(f"Default lb={self.DEFAULT_LB}, ub={self.DEFAULT_UB}.")
                            self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                            self.lb = self.DEFAULT_LB * ones(self.problem_size)
                            self.ub = self.DEFAULT_UB * ones(self.problem_size)
                        else:
                            print("Wrong lower bound and upper bound parameters.")
                            exit(0)
                    elif len(lb) == 1:
                        if "problem_size" in kwargs:
                            self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                            self.lb = lb[0] * ones(self.problem_size)
                            self.ub = ub[0] * ones(self.problem_size)
                    else:
                        self.problem_size = len(lb)
                        self.lb = array(lb)
                        self.ub = array(ub)
                else:
                    print("Lower bound and Upper bound need to be same length")
                    exit(0)
            elif type(lb) in [int, float] and type(ub) in [int, float]:
                self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                self.lb = lb * ones(self.problem_size)
                self.ub = ub * ones(self.problem_size)
            else:
                print("Lower bound and Upper bound need to be a list.")
                exit(0)

    def __check_problem_size__(self, problem_size):
        if problem_size is None:
            print("Problem size must be an int number")
            exit(0)
        elif problem_size <= 0:
            print("Problem size must > 0")
            exit(0)
        return int(ceil(problem_size))

    def __check_optional_parameters__(self, kwargs):
        if "batch_idea" in kwargs:
            batch_idea = kwargs["batch_idea"]
            if type(batch_idea) == bool:
                self.batch_idea = batch_idea
            else:
                self.batch_idea = self.DEFAULT_BATCH_IDEA
            if "batch_size" in kwargs:
                batch_size = kwargs["batch_size"]
                if type(batch_size) == int:
                    self.batch_size = batch_size
                else:
                    self.batch_size = self.DEFAULT_BATCH_SIZE
            else:
                self.batch_size = self.DEFAULT_BATCH_SIZE
        else:
            self.batch_idea = self.DEFAULT_BATCH_IDEA

    def __check_objective_function__(self, kwargs):
        tested_solution = uniform(self.lb, self.ub)
        try:
            result = self.obj_func(tested_solution)
        except Exception as err:
            print(f"Error: {err}\n")
            print("Please check your defined objective function!")
            exit(0)
        if isinstance(result, list) or isinstance(result, ndarray):
            self.n_objs = len(result)
            if self.n_objs > 1:
                self.multi_objs = True
                if "obj_weight" in kwargs:
                    self.obj_weight = kwargs["obj_weight"]
                    if isinstance(self.obj_weight, list) or isinstance(self.obj_weight, ndarray):
                        if self.n_objs != len(self.obj_weight):
                            print(f"Please check your objective function/weight. N objs = {self.n_objs}, but N weights = {len(self.obj_weight)}")
                            exit(0)
                        if self.verbose:
                            print(f"N objs = {self.n_objs} with weights = {self.obj_weight}")
                    else:
                        print(f"Please check your objective function/weight. N objs = {self.n_objs}, weights must be a list or numpy array with same length.")
                        exit(0)
                else:
                    self.obj_weight = ones(self.n_objs)
                    if self.verbose:
                        print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weight = ones(1)
                if self.verbose:
                    print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            else:
                print(f"Please check your objective function. It returns nothing!")
                exit(0)
        else:
            if type(result) in (int, float):
                self.multi_objs = False
                self.obj_is_list = False
                self.obj_weight = ones(1)

    def create_solution(self):
        """ Return the position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def get_fitness_position(self, position=None):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :return:
        """
        objs = self.obj_func(position)
        if not self.obj_is_list:
            objs = [objs]
        fit = dot(objs, self.obj_weight)
        fit = fit if self.minmax == "min" else 1.0 / (fit + self.EPSILON)
        return [fit, objs]

    def get_fitness_solution(self, solution=None):
        return self.get_fitness_position(solution[self.ID_POS])

    def get_global_best_solution(self, pop=None):
        """ Sort a copy of population and return the copy of the best position """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        return deepcopy(sorted_pop[0]) if self.minmax == "min" else deepcopy(sorted_pop[-1])

    def update_global_best_solution(self, pop=None):
        """ Sort the copy of population and update the current best position. Return the new current best position """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        if self.minmax == "min":
            current_best = sorted_pop[0]
            self.c_best_list.append(deepcopy(current_best))
            if current_best[self.ID_FIT][self.ID_TAR] < self.g_best_list[-1][self.ID_FIT][self.ID_TAR]:
                self.g_best_list.append(deepcopy(current_best))
            else:
                self.g_best_list.append(deepcopy(self.g_best_list[-1]))
        else:
            current_best = sorted_pop[-1]
            self.c_best_list.append(current_best)
            better = deepcopy(current_best) if current_best[self.ID_FIT][self.ID_TAR] > self.g_best_list[-1][self.ID_FIT][self.ID_TAR] \
                else deepcopy(self.g_best_list[-1])
            self.g_best_list.append(better)

    def print_epoch(self, epoch, runtime):
        if self.verbose:
            print(f"> Epoch: {epoch}, Current best: {self.c_best_list[-1][self.ID_FIT][self.ID_TAR]}, "
                  f"Global best: {self.g_best_list[-1][self.ID_FIT][self.ID_TAR]}, Runtime: {runtime:.5f} seconds")

    def save_data(self):
        # Draw the convergence line with this data
        self.g_fit_best_list = [agent[self.ID_FIT][self.ID_TAR] for agent in self.g_best_list]
        self.c_fit_best_list = [agent[self.ID_FIT][self.ID_TAR] for agent in self.c_best_list]

        # Draw the exploration and exploitation line with this data
        self.div_list = ones(self.epoch)
        for idx, pop in enumerate(self.pop_list):
            pos_matrix = array([agent[self.ID_POS] for agent in pop])
            div = mean(abs((median(pos_matrix, axis=0) - pos_matrix)), axis=0)
            self.div_list[idx] = mean(div, axis=0)
        div_max = np_max(self.div_list)
        self.explore_list = 100 * (self.div_list / div_max)
        self.exploit_list = 100 - self.explore_list

    def get_global_best_global_worst_solution(self, pop=None, id_fit=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        if id_best == self.ID_MIN_PROB:
            return deepcopy(sorted_pop[id_best]), deepcopy(sorted_pop[self.ID_MAX_PROB])
        elif id_best == self.ID_MAX_PROB:
            return deepcopy(sorted_pop[id_best]), deepcopy(sorted_pop[self.ID_MIN_PROB])

    def update_global_best_global_worst_solution(self, pop=None, id_best=None, id_worst=None, g_best=None):
        """ Sort the copy of population and update the current best position. Return the new current best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return g_best, sorted_pop[id_worst]

    def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        """ Sort population and return the sorted population and the best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return sorted_pop, deepcopy(sorted_pop[id_best])

    def amend_position(self, position=None):
        return maximum(self.lb, minimum(self.ub, position))

    def amend_position_faster(self, position=None):
        return clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return sorted_pop, g_best

    def create_opposition_position(self, position=None, g_best=None):
        return self.lb + self.ub - g_best[self.ID_POS] + uniform() * (g_best[self.ID_POS] - position)

    def levy_flight(self, epoch=None, position=None, g_best_position=None, step=0.001, case=0):
        """
        Parameters
        ----------
        epoch (int): current iteration
        position : 1-D numpy array
        g_best_position : 1-D numpy array
        step (float, optional): 0.001
        case (int, optional): 0, 1, 2

        """
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = power(gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = normal(0, sigma_muy ** 2)
        v = normal(0, sigma_v ** 2)
        s = muy / power(abs(v), 1 / beta)
        levy = uniform(self.lb, self.ub) * step * s * (position - g_best_position)

        if case == 0:
            return levy
        elif case == 1:
            return position + 1.0 / sqrt(epoch + 1) * sign(random() - 0.5) * levy
        elif case == 2:
            return position + normal(0, 1, len(self.lb)) * levy
        elif case == 3:
            return position + 0.01 * levy

    def levy_flight_2(self, position=None, g_best_position=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * sin(pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (normal(0, xichma_u ** 2)) / (sqrt(abs(normal(0, xichma_v ** 2))) ** (1.0 / 1.5))
        return position + alpha * levy_b * (position - g_best_position)

    def step_size_by_levy_flight(self, multiplier=0.001, beta=1.0, case=0):
        """
        Parameters
        ----------
        multiplier (float, optional): 0.01
        beta: [0-2]
            + 0-1: small range --> exploit
            + 1-2: large range --> explore
        case: 0, 1, -1
            + 0: return multiplier * s * uniform()
            + 1: return multiplier * s * normal(0, 1)
            + -1: return multiplier * s
        """
        # u and v are two random variables which follow normal distribution
        # sigma_u : standard deviation of u
        sigma_u = power(gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        u = normal(0, sigma_u ** 2)
        v = normal(0, sigma_v ** 2)
        s = u / power(abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * uniform()
        elif case == 1:
            step = multiplier * s * normal(0, 1)
        else:
            step = multiplier * s
        return step

    def get_index_roulette_wheel_selection(self, list_fitness=None):
        """ It can handle negative also. Make sure your list fitness is 1D-numpy array"""
        scaled_fitness = (list_fitness - min(list_fitness)) / (ptp(list_fitness) + self.EPSILON)
        minimized_fitness = 1.0 - scaled_fitness
        total_sum = sum(minimized_fitness)
        r = uniform(low=0, high=total_sum)
        for idx, f in enumerate(minimized_fitness):
            r = r + f
            if r > total_sum:
                return idx

    def get_parent_kway_tournament_selection(self, pop=None, k_way=0.2, output=2):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = choice(range(len(pop)), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[:output]

    ### Crossover
    def crossover_arthmetic_recombination(self, dad_pos=None, mom_pos=None):
        r = uniform()           # w1 = w2 when r =0.5
        w1 = multiply(r, dad_pos) + multiply((1 - r), mom_pos)
        w2 = multiply(r, mom_pos) + multiply((1 - r), dad_pos)
        return w1, w2

    ### Mutation
    ### This method won't be used in any algorithm because of it's slow performance
    ### Using numpy vector for faster performance
    def mutation_flip_point(self, parent_pos, idx):
        w = deepcopy(parent_pos)
        w[idx] = uniform(self.lb[idx], self.ub[idx])
        return w


    #### Improved techniques can be used in any algorithms: 1
    ## Based on this paper: An efficient equilibrium optimizer with mutation strategy for numerical optimization (but still different)
    ## This scheme used after the original and including 4 step:
    ##  s1: sort population, take p1 = 1/2 best population for next round
    ##  s2: do the mutation for p1, using greedy method to select the better solution
    ##  s3: do the search mechanism for p1 (based on global best solution and the updated p1 above), to make p2 population
    ##  s4: construct the new population for next generation
    def improved_ms(self, pop=None, g_best=None):    ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]
        ## Mutation scheme
        for i in range(0, pop_len):
            pos_new = pop_s1[i][self.ID_POS] * (1 + normal(0, 1, self.problem_size))
            fit = self.get_fitness_position(pos_new)
            if fit < pop_s1[i][self.ID_FIT]:        ## Greedy method --> improved exploitation
                pop_s1[i] = [pos_new, fit]
        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = mean(pos_s1_list, axis=0)
        for i in range(0, pop_len):
            pos_new = (g_best[self.ID_POS] - pos_s1_mean) - random() * (self.lb + random() * (self.ub - self.lb))
            fit = self.get_fitness_position(pos_new)
            pop_s2[i] = [pos_new, fit]              ## Keep the diversity of populatoin and still improved the exploration

        ## Construct a new population
        pop = pop_s1 + pop_s2
        pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        return pop, g_best

    def train(self):
        pass
