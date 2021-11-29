#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSSDO(Optimizer):
    """
    The original version of: Social Ski-Driver Optimization (SSDO)
        (Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm)
    Noted:
        https://doi.org/10.1007/s00521-019-04159-z
        https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019
    """
    ID_VEL = 2
    ID_LOC = 3

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], velocity, best_local_position]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        pos_local = deepcopy(position)
        return [position, fitness, velocity, pos_local]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

        ## Calculate the mean of the best three solutions in each dimension. Eq 9
        _, pop_best3, _ = self.get_special_solutions(self.pop, best=3)
        pos_mean = np.mean(np.array([item[self.ID_POS] for item in pop_best3]))

        pop_new = deepcopy(self.pop)
        # Updating velocity vectors
        for i in range(0, self.pop_size):
            r1 = np.random.uniform()  # r1, r2 is a random number in [0,1]
            r2 = np.random.uniform()
            if r2 <= 0.5:  ## Use Sine function to move
                vel_new = c * np.sin(r1) * (self.pop[i][self.ID_LOC] - self.pop[i][self.ID_POS]) + np.sin(r1) * (pos_mean - self.pop[i][self.ID_POS])
            else:  ## Use Cosine function to move
                vel_new = c * np.cos(r1) * (self.pop[i][self.ID_LOC] - self.pop[i][self.ID_POS]) + np.cos(r1) * (pos_mean - self.pop[i][self.ID_POS])
            pop_new[i][self.ID_VEL] = vel_new

        ## Reproduction
        for idx in range(0, self.pop_size):
            pos_new = pop_new[idx][self.ID_POS] + pop_new[idx][self.ID_VEL]
            pos_new = self.amend_position_faster(pos_new)
            pop_new[idx][self.ID_POS] = pos_new
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
