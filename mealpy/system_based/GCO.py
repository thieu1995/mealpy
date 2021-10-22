#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseGCO(Optimizer):
    """
    My modified version of: Germinal Center Optimization (GCO)
        (Germinal Center Optimization Algorithm)
    Link:
        https://www.atlantis-press.com/journals/ijcis/25905179/view
    Noted:
        + Using batch-size updating
        + Instead randomize choosing 3 solution, I use 2 random solution and global best solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, cr=0.7, wf=1.25, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.cr = cr
        self.wf = wf

        ## Dynamic variables
        self.dyn_list_cell_counter = np.ones(self.pop_size)         # CEll Counter
        self.dyn_list_life_signal = 70 * np.ones(self.pop_size)     # 70% to duplicate, and 30% to die  # LIfe-Signal

    def create_child(self, idx, pop, g_best):
        if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
            self.dyn_list_cell_counter[idx] += 1
        else:
            self.dyn_list_cell_counter[idx] = 1

        # Mutate process
        r1, r2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
        pos_new = g_best[self.ID_POS] + self.wf * (pop[r2][self.ID_POS] - pop[r1][self.ID_POS])
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new, pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)

        if self.compare_agent([pos_new, fit_new], pop[idx]):
            self.dyn_list_cell_counter[idx] += 10
            return [pos_new, fit_new]
        return pop[idx].copy()

        # ## Update based on batch-size training
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

        ## Dark-zone process    (can be parallelization)
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

        ## Light-zone process   (no needs parallelization)
        for i in range(0, self.pop_size):
            self.dyn_list_cell_counter[i] = 10
            fit_list = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop_new])
            fit_max = max(fit_list)
            fit_min = min(fit_list)
            self.dyn_list_cell_counter[i] += 10 * (pop[i][self.ID_FIT][self.ID_TAR] - fit_max) / (fit_min - fit_max + self.EPSILON)
        return pop_new


class OriginalGCO(BaseGCO):
    """
    Original version of: Germinal Center Optimization (GCO)
        (Germinal Center Optimization Algorithm)
    Link:
        DOI: https://doi.org/10.2991/ijcis.2018.25905179
    """

    def __init__(self, problem, epoch=10000, pop_size=100, cr=0.7, wf=1.25, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(problem, epoch, pop_size, cr, wf, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def create_child(self, idx, pop, g_best):
        if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
            self.dyn_list_cell_counter[idx] += 1
        else:
            self.dyn_list_cell_counter[idx] = 1

        # Mutate process
        r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
        pos_new = pop[r1][self.ID_POS] + self.wf * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new, pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            self.dyn_list_cell_counter[idx] += 10
            return [pos_new, fit_new]
        return pop[idx].copy()
