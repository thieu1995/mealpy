#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:58, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy


class Problem:

    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    ID_TAR = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in fitness

    DEFAULT_BATCH_IDEA = False
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_LB = -1
    DEFAULT_UB = 1

    def __init__(self, problem: dict):
        """
        Args:
            problem (dict): Dict properties of your problem

        Examples:
             problem = {
                "obj_func": your objective function,
                "lb": list of value
                "ub": list of value
                "minmax": "min" or "max"
                "verbose": True or False
                "problem_size": int (Optional)
                "batch_idea": True or False (Optional)
                "batch_size": int (Optional, smaller than population size)
                "obj_weight": list weights for all your objectives (Optional, default = [1, 1, ...1])
             }
        """
        self.minmax = "min"
        self.verbose = True
        self.batch_size = 10
        self.batch_idea = False
        self.n_objs = 1
        self.obj_weight = None
        self.multi_objs = False
        self.obj_is_list = False
        self.problem_size, self.lb, self.ub = None, None, None
        self.__set_parameters__(problem)
        self.__check_parameters__(problem)
        self.__check_optional_parameters__(problem)
        self.__check_objective_function__(problem)

    def __set_parameters__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __check_parameters__(self, kwargs):
        if "lb" in kwargs and "ub" in kwargs:
            lb, ub = kwargs["lb"], kwargs["ub"]
            if (lb is None) or (ub is None):
                if "problem_size" in kwargs:
                    print(f"Default lb={self.DEFAULT_LB}, ub={self.DEFAULT_UB}.")
                    self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                    self.lb = self.DEFAULT_LB * np.ones(self.problem_size)
                    self.ub = self.DEFAULT_UB * np.ones(self.problem_size)
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
                                self.lb = self.DEFAULT_LB * np.ones(self.problem_size)
                                self.ub = self.DEFAULT_UB * np.ones(self.problem_size)
                            else:
                                print("Wrong lower bound and upper bound parameters.")
                                exit(0)
                        elif len(lb) == 1:
                            if "problem_size" in kwargs:
                                self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                                self.lb = lb[0] * np.ones(self.problem_size)
                                self.ub = ub[0] * np.ones(self.problem_size)
                        else:
                            self.problem_size = len(lb)
                            self.lb = np.array(lb)
                            self.ub = np.array(ub)
                    else:
                        print("Lower bound and Upper bound need to be same length")
                        exit(0)
                elif type(lb) in [int, float] and type(ub) in [int, float]:
                    self.problem_size = self.__check_problem_size__(kwargs["problem_size"])
                    self.lb = lb * np.ones(self.problem_size)
                    self.ub = ub * np.ones(self.problem_size)
                else:
                    print("Lower bound and Upper bound need to be a list.")
                    exit(0)
        else:
            print("Please define lb and ub values!")
            exit(0)

    def __check_problem_size__(self, problem_size):
        if problem_size is None:
            print("Problem size must be an int number")
            exit(0)
        elif problem_size <= 0:
            print("Problem size must > 0")
            exit(0)
        return int(np.ceil(problem_size))

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
        if "obj_func" in kwargs:
            obj_func = kwargs["obj_func"]
            if callable(obj_func):
                self.obj_func = obj_func
            else:
                print("Please check your function. It needs to return value!")
                exit(0)
        tested_solution = np.random.uniform(self.lb, self.ub)
        try:
            result = self.obj_func(tested_solution)
        except Exception as err:
            print(f"Error: {err}\n")
            print("Please check your defined objective function!")
            exit(0)
        if isinstance(result, list) or isinstance(result, np.ndarray):
            self.n_objs = len(result)
            if self.n_objs > 1:
                self.multi_objs = True
                if "obj_weight" in kwargs:
                    self.obj_weight = kwargs["obj_weight"]
                    if isinstance(self.obj_weight, list) or isinstance(self.obj_weight, np.ndarray):
                        if self.n_objs != len(self.obj_weight):
                            print(f"Please check your objective function/weight. N objs = {self.n_objs}, but N weights = {len(self.obj_weight)}")
                            exit(0)
                        if self.verbose:
                            print(f"N objs = {self.n_objs} with weights = {self.obj_weight}")
                    else:
                        print(f"Please check your objective function/weight. N objs = {self.n_objs}, weights must be a list or numpy np.array with same length.")
                        exit(0)
                else:
                    self.obj_weight = np.ones(self.n_objs)
                    if self.verbose:
                        print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weight = np.ones(1)
                if self.verbose:
                    print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            else:
                print(f"Please check your objective function. It returns nothing!")
                exit(0)
        else:
            if isinstance(result, np.floating) or type(result) in (int, float):
                self.multi_objs = False
                self.obj_is_list = False
                self.obj_weight = np.ones(1)
            else:
                print("Please check your objective function. It needs to return value!")
                exit(0)


class Optimizer(Problem):
    """ This is base class of all Algorithms """

    ## Assumption the A solution with format: [position, [target, [obj1, obj2, ...]]]
    ID_POS = 0  # Index of position/location of solution/agent
    ID_FIT = 1  # Index of fitness value of solution/agent

    EPSILON = 10E-10

    def __init__(self, problem: dict):
        """
        Args:
            problem (dict): Dict properties of your problem

        Examples:
            problem = {
                "obj_func": your objective function,
                "lb": list of value
                "ub": list of value
                "minmax": "min" or "max"
                "verbose": True or False
                "problem_size": int (Optional)
                "batch_idea": True or False (Optional)
                "batch_size": int (Optional, smaller than population size)
                "obj_weight": list weights for all your objectives (Optional, default = [1, 1, ...1])
             }
        """
        super(Optimizer, self).__init__(problem)
        self.epoch, self.pop_size = None, None
        self.solution, self.loss_train = None, []
        self.history_list_g_best = []           # List of global best solution found so far in all previous generations
        self.history_list_c_best = []           # List of current best solution in each previous generations
        self.history_list_epoch_time = []       # List of runtime for each generation
        self.history_list_g_best_fit = []       # List of global best fitness found so far in all previous generations
        self.history_list_c_best_fit = []       # List of current best fitness in each previous generations
        self.history_list_pop = []              # List of population in each generations
        self.history_list_div = None            # List of diversity of swarm in all generations
        self.history_list_exploit = None        # List of exploitation percentages for all generations
        self.history_list_explore = None        # List of exploration percentages for all generations

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def get_fitness_position(self, position=None):
        """
        Args:
            position (nd.array): 1-D numpy array

        Returns:
            [target, [obj1, obj2, ...]]
        """
        objs = self.obj_func(position)
        if not self.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.obj_weight)
        # fit = fit if self.minmax == "min" else 1.0 / (fit + self.EPSILON)
        return [fit, objs]

    def get_fitness_solution(self, solution=None):
        """
        Args:
            solution (list): A solution with format [position, [target, [obj1, obj2, ...]]]

        Returns:
            [target, [obj1, obj2, ...]]
        """
        return self.get_fitness_position(solution[self.ID_POS])

    def get_global_best_solution(self, pop: list):
        """
        Sort population and return the sorted population and the best solution

        Args:
            pop (list): The population of pop_size individuals

        Returns:
            Sorted population and global best solution
        """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])  # Already returned a new sorted list
        if self.minmax == "min":
            return sorted_pop, sorted_pop[0].copy()
        else:
            return sorted_pop, sorted_pop[-1].copy()

    def get_better_solution(self, agent1: list, agent2: list):
        """
        Args:
            agent1 (list): A solution
            agent2 (list): Another solution

        Returns:
            The better solution between them
        """
        if self.minmax == "min":
            if agent1[self.ID_FIT][self.ID_TAR] < agent2[self.ID_FIT][self.ID_TAR]:
                return agent1.copy()
            return agent2.copy()
        else:
            if agent1[self.ID_FIT][self.ID_TAR] < agent2[self.ID_FIT][self.ID_TAR]:
                return agent2.copy()
            return agent1.copy()

    def update_global_best_solution(self, pop=None):
        """
        Update the global best solution saved in variable named: self.history_list_g_best
        Args:
            pop (list): The population of pop_size individuals

        Returns:
            sorted population
        """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        current_best = sorted_pop[0] if self.minmax == "min" else sorted_pop[-1]
        self.history_list_c_best.append(current_best)
        better = self.get_better_solution(current_best, self.history_list_g_best[-1])
        self.history_list_g_best.append(better)
        return sorted_pop.copy()

    def print_epoch(self, epoch, runtime):
        """
        Print out the detailed information of training process
        Args:
            epoch (int): current iteration
            runtime (float): the runtime for current iteration
        """
        if self.verbose:
            print(f"> Epoch: {epoch}, Current best: {self.history_list_c_best[-1][self.ID_FIT][self.ID_TAR]}, "
                  f"Global best: {self.history_list_g_best[-1][self.ID_FIT][self.ID_TAR]}, Runtime: {runtime:.5f} seconds")

    def save_data(self):
        """
        Detail: Save important data for later use such as:
            + history_list_g_best_fit
            + history_list_c_best_fit
            + history_list_div
            + history_list_explore
            + history_list_exploit
        """
        self.history_list_g_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history_list_g_best]
        self.history_list_c_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history_list_c_best]

        # Draw the exploration and exploitation line with this data
        self.history_list_div = np.ones(self.epoch)
        for idx, pop in enumerate(self.history_list_pop):
            pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
            div = np.mean(abs((np.median(pos_matrix, axis=0) - pos_matrix)), axis=0)
            self.history_list_div[idx] = np.mean(div, axis=0)
        div_max = np.max(self.history_list_div)
        self.history_list_explore = 100 * (self.history_list_div / div_max)
        self.history_list_exploit = 100 - self.history_list_explore

    ## Crossover techniques

    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        """
        This method can handle min/max problem, and negative or positive fitness value.
        Args:
            list_fitness (nd.array): 1-D numpy array

        Returns:
            Index of selected solution
        """
        scaled_fitness = (list_fitness - np.min(list_fitness)) / (np.ptp(list_fitness) + self.EPSILON)
        if self.minmax == "min":
            final_fitness = 1.0 - scaled_fitness
        else:
            final_fitness = scaled_fitness
        total_sum = sum(final_fitness)
        r = np.random.uniform(low=0, high=total_sum)
        for idx, f in enumerate(final_fitness):
            r = r + f
            if r > total_sum:
                return idx

    def get_solution_kway_tournament_selection(self, pop: list, k_way=0.2, output=2):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        k_way = round(k_way)
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        if self.minmax == "min":
            return list_parents[:output]
        else:
            return list_parents[-output:]







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

    def amend_position(self, position=None):
        return np.maximum(self.lb, np.minimum(self.ub, position))

    def amend_position_faster(self, position=None):
        return np.clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return np.where(np.logical_and(self.lb <= position, position <= self.ub), position, np.random.uniform(self.lb, self.ub))

    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return sorted_pop, g_best

    def create_opposition_position(self, position=None, g_best=None):
        return self.lb + self.ub - g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - position)

    def levy_flight(self, epoch=None, position=None, g_best_position=None, step=0.001, case=0):
        """
        Parameters
        ----------
        epoch (int): current iteration
        position : 1-D numpy np.array
        g_best_position : 1-D numpy np.array
        step (float, optional): 0.001
        case (int, optional): 0, 1, 2

        """
        beta = 1
        # muy and v are two random variables which follow np.random.normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = muy / np.power(abs(v), 1 / beta)
        levy = np.random.uniform(self.lb, self.ub) * step * s * (position - g_best_position)

        if case == 0:
            return levy
        elif case == 1:
            return position + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * levy
        elif case == 2:
            return position + np.random.normal(0, 1, len(self.lb)) * levy
        elif case == 3:
            return position + 0.01 * levy

    def levy_flight_2(self, position=None, g_best_position=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (np.random.normal(0, xichma_u ** 2)) / (np.sqrt(abs(np.random.normal(0, xichma_v ** 2))) ** (1.0 / 1.5))
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
            + 0: return multiplier * s * np.random.uniform()
            + 1: return multiplier * s * np.random.normal(0, 1)
            + -1: return multiplier * s
        """
        # u and v are two random variables which follow np.random.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        u = np.random.normal(0, sigma_u ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = u / np.power(abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * np.random.uniform()
        elif case == 1:
            step = multiplier * s * np.random.normal(0, 1)
        else:
            step = multiplier * s
        return step

    def get_parent_kway_tournament_selection(self, pop=None, k_way=0.2, output=2):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[:output]

    ### Crossover
    def crossover_arthmetic_recombination(self, dad_pos=None, mom_pos=None):
        r = np.random.uniform()           # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
        return w1, w2

    ### Mutation
    ### This method won't be used in any algorithm because of it's slow performance
    ### Using numpy vector for faster performance
    def mutation_flip_point(self, parent_pos, idx):
        w = deepcopy(parent_pos)
        w[idx] = np.random.uniform(self.lb[idx], self.ub[idx])
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
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem_size))
            fit = self.get_fitness_position(pos_new)
            if fit < pop_s1[i][self.ID_FIT]:        ## Greedy method --> improved exploitation
                pop_s1[i] = [pos_new, fit]
        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        for i in range(0, pop_len):
            pos_new = (g_best[self.ID_POS] - pos_s1_mean) - np.random.random() * (self.lb + np.random.random() * (self.ub - self.lb))
            fit = self.get_fitness_position(pos_new)
            pop_s2[i] = [pos_new, fit]              ## Keep the diversity of populatoin and still improved the exploration

        ## Construct a new population
        pop = pop_s1 + pop_s2
        pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        return pop, g_best

    def train(self):
        pass
