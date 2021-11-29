#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:27, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseEP(Optimizer):
    """
        The original version of: Evolutionary Programming (EP)
            (Clever Algorithms: Nature-Inspired Programming Recipes - Evolutionary Programming)
        Link:
            http://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, problem, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            n_child (float/int): if float number --> percentage of child agents, int --> number of child agents
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        if bout_size < 1:                   # Number of tried with tournament selection (5% of pop_size)
            self.bout_size = int(bout_size * self.pop_size)
        else:
            self.bout_size = int(bout_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], strategy, times_win]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        strategy = np.random.uniform(0, self.distance, self.problem.n_dims)
        times_win = 0
        return [position, fitness, strategy, times_win]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position_faster(pos_new)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
        child = self.update_fitness_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1
        pop = sorted(pop, key=lambda item: item[self.ID_WIN], reverse=True)
        self.pop = pop[:self.pop_size]


class LevyEP(BaseEP):
    """
        The levy version of: Evolutionary Programming (EP)
        Noted:
            + Applied levy-flight
            + Change the flow and add more equations
    """

    def __init__(self, problem, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            n_child (float/int): if float number --> percentage of child agents, int --> number of child agents
        """
        super().__init__(problem, epoch, pop_size, bout_size, **kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            epoch (int): The current iteration
        """

        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position_faster(pos_new)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
        child = self.update_fitness_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1

        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        pop = sorted(pop, key=lambda agent: agent[self.ID_WIN], reverse=True)
        pop = deepcopy(pop[:self.pop_size])
        pop_left = deepcopy(pop[self.pop_size:])

        ## Choice random 50% of population left
        pop_comeback = []
        idx_list = np.random.choice(range(0, len(pop_left)), int(0.5 * len(pop_left)), replace=False)
        for idx in idx_list:
            levy = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = pop_left[idx][self.ID_POS] + 0.01 * levy
            strategy = self.distance = 0.05 * (self.problem.ub - self.problem.lb)
            pop_comeback.append([pos_new, None, strategy, 0])
        pop_comeback = self.update_fitness_population(pop_comeback)
        self.nfe_per_epoch = self.pop_size + int(0.5 * len(pop_left))
        self.pop = self.get_sorted_strim_population(pop + pop_comeback, self.pop_size)