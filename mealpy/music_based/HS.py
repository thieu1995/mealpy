#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseHS(Optimizer):
    """
    My version of: Harmony Search (HS)
    Noted:
        - Using global best in the harmony memories
        - Remove third for loop
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_new=50, c_r=0.95, pa_r=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            n_new (int): Number of New Harmonies, default = 0.85
            c_r (float): Harmony Memory Consideration Rate, default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.n_new = n_new
        self.c_r = c_r
        self.pa_r = pa_r
        self.fw = 0.0001 * (self.problem.ub - self.problem.lb)  # Fret Width (Bandwidth)
        self.fw_damp = 0.9995                                   # Fret Width Damp Ratio

        self.dyn_fw = self.fw

    def create_child(self, idx, pop, g_best):
        # Create New Harmony Position
        pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
        delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)

        # Use Harmony Memory
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.c_r, g_best[self.ID_POS], pos_new)
        # Pitch Adjustment
        x_new = pos_new + delta
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pa_r, x_new, pos_new)

        pos_new = self.amend_position_faster(pos_new)   # Check the bound
        fit_new = self.get_fitness_position(pos_new)    # Evaluation
        return [pos_new, fit_new]

        # Batch-size idea
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) % self.pop_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)


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

        ## Reproduction
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, g_best) for idx in pop_idx]

        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp

        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        pop = pop + pop_new
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        return pop[:self.pop_size]


class OriginalHS(BaseHS):
    """
    Original version of: Harmony Search (HS)
        A New Heuristic Optimization Algorithm: Harmony Search
    Link:

    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_new=50, c_r=0.95, pa_r=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            n_new (int): Number of New Harmonies, default = 0.85
            c_r (float): Harmony Memory Consideration Rate), default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(problem, epoch, pop_size, n_new, c_r, pa_r, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def create_child(self, idx, pop, g_best):
        pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
        for j in range(self.problem.n_dims):
            # Use Harmony Memory
            if np.random.uniform() <= self.c_r:
                random_index = np.random.randint(0, self.pop_size)
                pos_new[j] = pop[random_index][self.ID_POS][j]
            # Pitch Adjustment
            if np.random.uniform() <= self.pa_r:
                delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)  # Gaussian(Normal)
                pos_new[j] = pos_new[j] + delta[j]
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]
