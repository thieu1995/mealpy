#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:56, 07/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAOA(Optimizer):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)
    Link:
        https://doi.org/10.1016/j.cma.2020.113609
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5,
            miu (float): fixed parameter , control parameter to adjust the search process, Default: 0.5,
            moa_min (float): range min of Math Optimizer Accelerated, Default: 0.2,
            moa_max (float): range max of Math Optimizer Accelerated, Default: 0.9,
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.miu = miu
        self.moa_min = moa_min
        self.moa_max = moa_max

    def create_child(self, idx, pop, g_best, moa, mop):
        pos_new = pop[idx][self.ID_POS].copy()
        for j in range(0, self.problem.n_dims):
            r1, r2, r3 = np.random.rand(3)
            if r1 > moa:  # Exploration phase
                if r2 < 0.5:
                    pos_new[j] = g_best[self.ID_POS][j] / (mop + self.EPSILON) * \
                                 ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
                else:
                    pos_new[j] = g_best[self.ID_POS][j] * mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
            else:  # Exploitation phase
                if r3 < 0.5:
                    pos_new[j] = g_best[self.ID_POS][j] - mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
                else:
                    pos_new[j] = g_best[self.ID_POS][j] + mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        moa = self.moa_min + epoch * ((self.moa_max - self.moa_min) / self.epoch)  # Eq. 2
        mop = 1 - (epoch ** (1.0 / self.alpha)) / (self.epoch ** (1.0 / self.alpha))  # Eq. 4

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, moa=moa, mop=mop), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, moa=moa, mop=mop), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, g_best, moa, mop) for idx in pop_idx]
        return pop_new


