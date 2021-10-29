#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)
        (Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103300
    """

    def __init__(self, problem, epoch=10000, pop_size=100, somersault_range=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (): somersault factor that decides the somersault range of manta rays, default=2
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.somersault_range = somersault_range

    def create_child(self, idx, pop, epoch, g_best):
        # Cyclone foraging (Eq. 5, 6, 7)
        if np.random.rand() < 0.5:
            r1 = np.random.uniform()
            beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

            if (epoch + 1) / self.epoch < np.random.uniform():
                x_rand = np.random.uniform(self.problem.lb, self.problem.ub)
                if idx == 0:
                    x_t1 = x_rand + np.random.uniform() * (x_rand - pop[idx][self.ID_POS]) + \
                           beta * (x_rand - pop[idx][self.ID_POS])
                else:
                    x_t1 = x_rand + np.random.uniform() * (pop[idx - 1][self.ID_POS] - pop[idx][self.ID_POS]) + \
                           beta * (x_rand - pop[idx][self.ID_POS])
            else:
                if idx == 0:
                    x_t1 = g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop[idx][self.ID_POS]) + \
                           beta * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
                else:
                    x_t1 = g_best[self.ID_POS] + np.random.uniform() * (pop[idx - 1][self.ID_POS] - pop[idx][self.ID_POS]) + \
                           beta * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        # Chain foraging (Eq. 1,2)
        else:
            r = np.random.uniform()
            alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
            if idx == 0:
                x_t1 = pop[idx][self.ID_POS] + r * (g_best[self.ID_POS] - pop[idx][self.ID_POS]) + \
                       alpha * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
            else:
                x_t1 = pop[idx][self.ID_POS] + r * (pop[idx - 1][self.ID_POS] - pop[idx][self.ID_POS]) + \
                       alpha * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(x_t1)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, pop, g_best):
        # Somersault foraging   (Eq. 8)
        x_t1 = pop[idx][self.ID_POS] + self.somersault_range * \
               (np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(x_t1)
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
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
            _, g_best = self.update_global_best_solution(child)
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=child, g_best=g_best), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
            _, g_best = self.update_global_best_solution(child)
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=child, g_best=g_best), pop_idx)
            pop = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, epoch, g_best) for idx in pop_idx]
            _, g_best = self.update_global_best_solution(child)
            pop = [self.create_child2(idx, pop, g_best) for idx in pop_idx]
        return pop
