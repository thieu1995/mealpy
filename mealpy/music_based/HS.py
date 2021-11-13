#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

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
            pop_size (int): number of population size, default = 100
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Create New Harmony Position
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)

            # Use Harmony Memory
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.c_r, self.g_best[self.ID_POS], pos_new)
            # Pitch Adjustment
            x_new = pos_new + delta
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pa_r, x_new, pos_new)
            pos_new = self.amend_position_faster(pos_new)  # Check the bound
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)

        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp

        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)


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
            pop_size (int): number of population size, default = 100
            n_new (int): Number of New Harmonies, default = 0.85
            c_r (float): Harmony Memory Consideration Rate), default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(problem, epoch, pop_size, n_new, c_r, pa_r, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            for j in range(self.problem.n_dims):
                # Use Harmony Memory
                if np.random.uniform() <= self.c_r:
                    random_index = np.random.randint(0, self.pop_size)
                    pos_new[j] = self.pop[random_index][self.ID_POS][j]
                # Pitch Adjustment
                if np.random.uniform() <= self.pa_r:
                    delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)  # Gaussian(Normal)
                    pos_new[j] = pos_new[j] + delta[j]
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)

        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp

        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
