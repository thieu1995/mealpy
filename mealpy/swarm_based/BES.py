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


class BaseBES(Optimizer):
    """
    Original version of: Bald Eagle Search (BES)
        (Novel meta-heuristic bald eagle search optimisation algorithm)
    Link:
        DOI: https://doi.org/10.1007/s10462-019-09732-5
    """

    def __init__(self, problem, epoch=10000, pop_size=100, a=10, R=1.5, alpha=2, c1=2, c2=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a (int): default: 10, determining the corner between point search in the central point, in [5, 10]
            R (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
            alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
            c1 (float): default: 2, in [1, 2]
            c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.a = a
        self.R = R
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = False

    def _create_x_y_x1_y1_(self):
        """ Using numpy vector for faster computational time """
        ## Eq. 2
        phi = self.a * np.pi * np.random.uniform(0, 1, self.pop_size)
        r = phi + self.R * np.random.uniform(0, 1, self.pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)

        ## Eq. 3
        r1 = phi1 = self.a * np.pi * np.random.uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)

        x_list = xr / max(xr)
        y_list = yr / max(yr)
        x1_list = xr1 / max(xr1)
        y1_list = yr1 / max(yr1)
        return x_list, y_list, x1_list, y1_list

    def create_child1(self, idx, pop, g_best, pos_mean):
        pos_new = g_best[self.ID_POS] + self.alpha * np.random.uniform() * (pos_mean - pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, pop, pos_mean, x_list, y_list):
        idx_rand = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
        pos_new = pop[idx][self.ID_POS] + y_list[idx] * (pop[idx][self.ID_POS] - pop[idx_rand][self.ID_POS]) + \
                  x_list[idx] * (pop[idx][self.ID_POS] - pos_mean)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child3(self, idx, pop, g_best, pos_mean, x1_list, y1_list):
        pos_new = np.random.uniform() * g_best[self.ID_POS] + x1_list[idx] * (pop[idx][self.ID_POS] - self.c1 * pos_mean) \
                  + y1_list[idx] * (pop[idx][self.ID_POS] - self.c2 * g_best[self.ID_POS])
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
        pop_idx = np.array(range(0, self.pop_size))
        ## 0. Pre-definded
        x_list, y_list, x1_list, y1_list = self._create_x_y_x1_y1_()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([individual[self.ID_POS] for individual in pop])
        pos_mean = np.mean(pos_list, axis=0)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                child1 = executor.map(partial(self.create_child1, pop=pop, g_best=g_best, pos_mean=pos_mean), pop_idx)
            child = [x for x in child1]

            ## 2. Search in space
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)

            with parallel.ThreadPoolExecutor() as executor:
                child2 = executor.map(partial(self.create_child2, pop=child, pos_mean=pos_mean, x_list=x_list, y_list=y_list), pop_idx)
            child = [x for x in child2]

            ## 3. Swoop
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)

            with parallel.ThreadPoolExecutor() as executor:
                child3 = executor.map(partial(self.create_child3, pop=child, g_best=g_best, pos_mean=pos_mean, x1_list=x1_list, y1_list=y1_list), pop_idx)
            child = [x for x in child3]

        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                child1 = executor.map(partial(self.create_child1, pop=pop, g_best=g_best, pos_mean=pos_mean), pop_idx)
            child = [x for x in child1]

            ## 2. Search in space
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)

            with parallel.ProcessPoolExecutor() as executor:
                child2 = executor.map(partial(self.create_child2, pop=child, pos_mean=pos_mean, x_list=x_list, y_list=y_list), pop_idx)
            child = [x for x in child2]

            ## 3. Swoop
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)

            with parallel.ProcessPoolExecutor() as executor:
                child3 = executor.map(partial(self.create_child3, pop=child, g_best=g_best, pos_mean=pos_mean, x1_list=x1_list, y1_list=y1_list), pop_idx)
            child = [x for x in child3]

        else:
            child = [self.create_child1(idx, pop=pop, g_best=g_best, pos_mean=pos_mean) for idx in pop_idx]
            ## 2. Search in space
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)
            child = [self.create_child2(idx, pop=child, pos_mean=pos_mean, x_list=x_list, y_list=y_list) for idx in pop_idx]
            ## 3. Swoop
            pos_list = np.array([individual[self.ID_POS] for individual in child])
            pos_mean = np.mean(pos_list, axis=0)
            child = [self.create_child3(idx, pop=child, g_best=g_best, pos_mean=pos_mean, x1_list=x1_list, y1_list=y1_list) for idx in pop_idx]
        return child
