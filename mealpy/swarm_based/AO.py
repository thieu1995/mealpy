#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:53, 07/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from math import gamma
from mealpy.optimizer import Optimizer


class OriginalAO(Optimizer):
    """
    The original version of: Aquila Optimization (AO)
    Link:
        Aquila Optimizer: A novel meta-heuristic optimization Algorithm
        https://doi.org/10.1016/j.cie.2021.107250
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
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = 0.1
        self.delta = 0.1

    def create_child(self, idx, pop, epoch, g_best, x, y, g1, g2, QF):
        x_mean = np.mean(np.array([item[self.ID_FIT][self.ID_TAR] for item in pop]), axis=0)
        if (epoch + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
            if np.random.rand() < 0.5:
                pos_new = g_best[self.ID_POS] * (1 - (epoch + 1) / self.epoch) + np.random.rand() * (x_mean - g_best[self.ID_POS])
            else:
                idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                pos_new = g_best[self.ID_POS] * self.get_simple_levy_step() + pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
        else:
            if np.random.rand() < 0.5:
                pos_new = self.alpha * (g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                          (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * self.delta  # Eq. 13
            else:
                pos_new = QF * g_best[self.ID_POS] - (g2 * pop[idx][self.ID_POS] *
                                        np.random.rand()) - g2 * self.get_simple_levy_step() + np.random.rand() * g1  # Eq. 14
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
        g1 = 2 * np.random.rand() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17

        dim_list = np.array(list(range(1, self.problem.n_dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = (epoch + 1) ** ((2 * np.random.rand() - 1) / (1 - self.epoch) ** 2)  # Eq.(15)        Quality function

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch,
                                                 g_best=g_best, x=x, y=y, g1=g1, g2=g2, QF=QF), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch,
                                                 g_best=g_best, x=x, y=y, g1=g1, g2=g2, QF=QF), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, epoch, g_best, x, y, g1, g2, QF) for idx in pop_idx]
        return pop + child

    def get_simple_levy_step(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, self.problem.n_dims) * sigma
        v = np.random.normal(1, self.problem.n_dims)
        step = u / abs(v) ** (1 / beta)
        return step

