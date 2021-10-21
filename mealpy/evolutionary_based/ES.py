#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:14, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseES(Optimizer):
    """
        The original version of: Evolution Strategies (ES)
            (Clever Algorithms: Nature-Inspired Programming Recipes - Evolution Strategies)
    Link:
        http://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2      # strategy

    def __init__(self, problem, epoch=10000, pop_size=100, n_child=0.75, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            n_child (float/int): if float number --> percentage of child agents, int --> number of child agents
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        if n_child < 1:             # lamda, 75% of pop_size
            self.n_child = int(n_child * self.pop_size)
        else:
            self.n_child = int(n_child)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

        self.nfe_per_epoch = pop_size + self.n_child
        self.sort_flag = True

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
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        strategy = np.random.uniform(0, self.distance)
        return [position, fitness, strategy]

    def create_child(self, agent_i):
        pos_new = agent_i[self.ID_POS] + agent_i[self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
        tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
        strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem.n_dims) + tau * np.random.normal(0, 1.0, self.problem.n_dims))
        return [pos_new, fit_new, strategy]

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        pop_copy = pop[:self.n_child].copy()

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(self.create_child, pop_copy)
            children = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(self.create_child, pop_copy)
            children = [x for x in pop_child]
        else:
            children = [self.create_child(agent) for agent in pop_copy]

        pop = sorted(pop + children, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        return pop[:self.pop_size]


class LevyES(BaseES):
    """
        The levy version of: Evolution Strategies (ES)
        Noted:
            + Applied levy-flight
            + Change the flow of algorithm
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_child=0.75, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            n_child (float/int): if float number --> percentage of child agents, int --> number of child agents
        """
        super().__init__(problem, epoch, pop_size, n_child, **kwargs)
        self.nfe_per_epoch = 2 * pop_size

    def create_levy_child(self, agent_i, g_best):
        levy = self.get_levy_flight_step(multiplier=0.01, case=-1)
        pos_new = agent_i[self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * \
                  levy * (agent_i[self.ID_POS] - g_best[self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
        tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
        stdevs = np.array([np.exp(tau_p * np.random.normal(0, 1.0) + tau * np.random.normal(0, 1.0)) for _ in range(self.problem.n_dims)])
        return [pos_new, fit_new, stdevs]

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        pop_copy = pop[:self.n_child].copy()
        pop_copy_levy = pop[self.n_child:].copy()

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(self.create_child, pop_copy)
            with parallel.ThreadPoolExecutor() as executor:
                pop_child_levy = executor.map(partial(self.create_levy_child, g_best=g_best), pop_copy_levy)
            children = [x for x in pop_child] + [x for x in pop_child_levy]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(self.create_child, pop_copy)
            with parallel.ProcessPoolExecutor() as executor:
                pop_child_levy = executor.map(partial(self.create_levy_child, g_best=g_best), pop_copy_levy)
            children = [x for x in pop_child] + [x for x in pop_child_levy]
        else:
            pop_child = [self.create_child(agent) for agent in pop_copy]
            pop_child_levy = [self.create_levy_child(agent, g_best) for agent in pop_copy_levy]
            children = pop_child + pop_child_levy

        pop = sorted(pop + children, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        return pop[:self.pop_size]

