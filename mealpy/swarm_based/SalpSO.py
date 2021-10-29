#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:38, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseSalpSO(Optimizer):
    """
    The original version of: Salp Swarm Optimization (SalpSO)
    Link:
        https://doi.org/10.1016/j.advengsoft.2017.07.002
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best, c1):
        if idx < self.pop_size / 2:
            c2_list = np.random.random(self.problem.n_dims)
            c3_list = np.random.random(self.problem.n_dims)
            pos_new_1 = g_best[self.ID_POS] + c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
            pos_new_2 = g_best[self.ID_POS] - c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
            pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
        else:
            # Eq. (3.4) in the paper
            pos_new = (pop[idx][self.ID_POS] + pop[idx - 1][self.ID_POS]) / 2

        # Check if salps go out of the search space and bring it back then re-calculate its fitness value
        # for i in range(0, self.pop_size):
        #     pos_new = self.amend_position_faster(pop[i][self.ID_POS])
        #     fit_new = self.get_fitness_position(pos_new)
        #     pop[i] = [pos_new, fit_new]

        pos_new = self.amend_position_faster(pos_new)
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
        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * (epoch + 1) / self.epoch) ** 2))
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, c1=c1), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, c1=c1), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, c1) for idx in pop_idx]
        return child
