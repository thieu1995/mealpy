#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
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
            pop_size (int): number of population size, default = 100
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Dark-zone process    (can be parallelization)
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1

            # Mutate process
            r1, r2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = self.g_best[self.ID_POS] + self.wf * (self.pop[r2][self.ID_POS] - self.pop[r1][self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.dyn_list_cell_counter[idx] += 10
                self.pop[idx] = deepcopy(pop_new[idx])

        ## Light-zone process   (no needs parallelization)
        for i in range(0, self.pop_size):
            self.dyn_list_cell_counter[i] = 10
            fit_list = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop_new])
            fit_max = max(fit_list)
            fit_min = min(fit_list)
            self.dyn_list_cell_counter[i] += 10 * (self.pop[i][self.ID_FIT][self.ID_TAR] - fit_max) / (fit_min - fit_max + self.EPSILON)


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
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(problem, epoch, pop_size, cr, wf, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1

            # Mutate process
            r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            pos_new = self.pop[r1][self.ID_POS] + self.wf * (self.pop[r2][self.ID_POS] - self.pop[r3][self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.dyn_list_cell_counter[idx] += 10
                self.pop[idx] = deepcopy(pop_new[idx])
