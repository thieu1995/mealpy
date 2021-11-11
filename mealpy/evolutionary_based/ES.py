#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:14, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

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

        self.nfe_per_epoch = self.n_child
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position_faster(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem.n_dims) + tau * np.random.normal(0, 1.0, self.problem.n_dims))
            child.append([pos_new, None, strategy])
        child = self.update_fitness_population(child)
        self.pop = self.get_sorted_strim_population(child + self.pop, self.pop_size)


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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        self.nfe_per_epoch = 2 * self.n_child
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position_faster(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem.n_dims) + tau * np.random.normal(0, 1.0, self.problem.n_dims))
            child.append([pos_new, None, strategy])
        child = self.update_fitness_population(child)

        child_levy = []
        for idx in range(0, self.n_child):
            levy = self.get_levy_flight_step(multiplier=0.01, case=-1)
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * \
                      levy * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            stdevs = np.array([np.exp(tau_p * np.random.normal(0, 1.0) + tau * np.random.normal(0, 1.0)) for _ in range(self.problem.n_dims)])
            child.append([pos_new, None, stdevs])
        child_levy = self.update_fitness_population(child_levy)
        self.pop = self.get_sorted_strim_population(child + child_levy + self.pop, self.pop_size)

