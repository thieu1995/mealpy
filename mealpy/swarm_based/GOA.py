#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:53, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseGOA(Optimizer):
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)
        (Grasshopper Optimisation Algorithm: Theory and Application Advances in Engineering Software)
    Link:
        http://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa
    Notes:
        + I added np.random.normal() component to Eq, 2.7
        + Changed the way to calculate distance between two location
    """
    def __init__(self, problem, epoch=10000, pop_size=100, c_minmax=(0.00004, 1), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_minmax (list): coefficient c
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax

    def _s_function__(self, r_vector=None):
        f = 0.5
        l = 1.5
        # Eq.(2.3) in the paper
        return f * np.exp(-r_vector / l) - np.exp(-r_vector)

    def create_child(self, idx, pop, g_best, c):
        S_i_total = np.zeros(self.problem.n_dims)
        for j in range(0, self.pop_size):
            dist = np.sqrt(np.sum((pop[idx][self.ID_POS] - pop[j][self.ID_POS]) ** 2))
            r_ij_vector = (pop[idx][self.ID_POS] - pop[j][self.ID_POS]) / (dist + self.EPSILON)  # xj - xi / dij in Eq.(2.7)
            xj_xi = 2 + np.remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
            ## The first part inside the big bracket in Eq. (2.7)   16 955 230 764    212 047 193 643
            ran = (c / 2) * (self.problem.ub - self.problem.lb)
            s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
            S_i_total += s_ij
        x_new = c * np.random.normal() * S_i_total + g_best[self.ID_POS]  # Eq. (2.7) in the paper
        pos_new = self.amend_position_faster(x_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

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
        # Eq.(2.8) in the paper
        c = self.c_minmax[1] - epoch * ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, c=c), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, c=c), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best=g_best, c=c) for idx in pop_idx]
        return child

