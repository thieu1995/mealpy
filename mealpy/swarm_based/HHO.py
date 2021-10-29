#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from math import gamma
import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseHHO(Optimizer):
    """
        The original version of: Harris Hawks Optimization (HHO)
            (Harris Hawks Optimization: Algorithm and Applications)
        Link:
            https://doi.org/10.1016/j.future.2019.02.028
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 1.5*pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best, epoch):
        # -1 < E0 < 1
        E0 = 2 * np.random.uniform() - 1
        # factor to show the decreasing energy of rabbit
        E = 2 * E0 * (1 - (epoch + 1) * 1.0 / self.epoch)
        J = 2 * (1 - np.random.uniform())

        # -------- Exploration phase Eq. (1) in paper -------------------
        if (np.abs(E) >= 1):
            # Harris' hawks perch randomly based on 2 strategy:
            if np.random.rand() >= 0.5:  # perch based on other family members
                X_rand = pop[np.random.randint(0, self.pop_size)][self.ID_POS].copy()
                pos_new = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * pop[idx][self.ID_POS])

            else:  # perch on a random tall tree (random site inside group's home range)
                X_m = np.mean([x[self.ID_POS] for x in pop])
                pos_new = (g_best[self.ID_POS] - X_m) - np.random.uniform() * \
                          (self.problem.lb + np.random.uniform() * (self.problem.ub - self.problem.lb))
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            return [pos_new, fit_new]
        # -------- Exploitation phase -------------------
        else:
            # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
            # phase 1: ----- surprise pounce (seven kills) ----------
            # surprise pounce (seven kills): multiple, short rapid dives by different hawks
            if (np.random.rand() >= 0.5):
                delta_X = g_best[self.ID_POS] - pop[idx][self.ID_POS]
                if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                    pos_new = delta_X - E * np.abs(J * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                else:  # Soft besiege Eq. (4) in paper
                    pos_new = g_best[self.ID_POS] - E * np.abs(delta_X)
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                return [pos_new, fit_new]
            else:
                xichma = np.power((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2.0)) /
                                  (gamma((1 + 1.5) * 1.5 * np.power(2, (1.5 - 1) / 2)) / 2.0), 1.0 / 1.5)
                LF_D = 0.01 * np.random.uniform() * xichma / np.power(np.abs(np.random.uniform()), 1.0 / 1.5)
                if np.abs(E) >= 0.5:  # Soft besiege Eq. (10) in paper
                    Y = g_best[self.ID_POS] - E * np.abs(J * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                else:  # Hard besiege Eq. (11) in paper
                    X_m = np.mean([x[self.ID_POS] for x in pop])
                    Y = g_best[self.ID_POS] - E * np.abs(J * g_best[self.ID_POS] - X_m)
                pos_Y = self.amend_position_faster(Y)
                fit_Y = self.get_fitness_position(pos_Y)
                Z = Y + np.random.uniform(self.problem.lb, self.problem.ub) * LF_D
                pos_Z = self.amend_position_faster(Z)
                fit_Z = self.get_fitness_position(pos_Z)

                if self.compare_agent([pos_Y, fit_Y], pop[idx]):
                    return [pos_Y, fit_Y]
                if self.compare_agent([pos_Z, fit_Z], pop[idx]):
                    return [pos_Z, fit_Z]
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
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child(idx, pop=pop, g_best=g_best, epoch=epoch) for idx in pop_idx]
        return pop
