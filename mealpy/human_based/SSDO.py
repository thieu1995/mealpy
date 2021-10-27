#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
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
        pos_local = position.copy()
        return [position, fitness, velocity, pos_local]

    def create_child(self, idx, pop):
        pos_new = pop[idx][self.ID_POS] + pop[idx][self.ID_VEL]
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new, pop[idx][self.ID_VEL].copy(), pos_new.copy()]
        return [pos_new, fit_new, pop[idx][self.ID_VEL], pop[idx][self.ID_LOC]]

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value, velocity, best_local_position]
        """
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

        ## Calculate the mean of the best three solutions in each dimension. Eq 9
        _, pop_best3, _ = self.get_special_solutions(pop, best=3)
        pos_mean = np.mean(np.array([item[self.ID_POS] for item in pop_best3]))

        # Updating velocity vectors
        for i in range(0, self.pop_size):
            r1 = np.random.uniform()  # r1, r2 is a random number in [0,1]
            r2 = np.random.uniform()
            if r2 <= 0.5:  ## Use Sine function to move
                vel_new = c * np.sin(r1) * (pop[i][self.ID_LOC] - pop[i][self.ID_POS]) + np.sin(r1) * (pos_mean - pop[i][self.ID_POS])
            else:  ## Use Cosine function to move
                vel_new = c * np.cos(r1) * (pop[i][self.ID_LOC] - pop[i][self.ID_POS]) + np.cos(r1) * (pos_mean - pop[i][self.ID_POS])
            pop[i][self.ID_VEL] = vel_new

        ## Reproduction
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop_copy) for idx in pop_idx]
        return child

