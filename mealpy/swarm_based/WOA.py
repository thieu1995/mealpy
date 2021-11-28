#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseWOA(Optimizer):
    """
        The original version of: Whale Optimization Algorithm (WOA)
            - In this algorithms: Prey means the best position
        Link:
            https://doi.org/10.1016/j.advengsoft.2016.01.008
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.create_solution()
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class HI_WOA(Optimizer):
    """
        The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)
            A hybrid improved whale optimization algorithm
        Link:
            https://ieenp.explore.ieee.org/document/8900003
    """

    def __init__(self, problem, epoch=10000, pop_size=100, feedback_max=10, **kwargs):
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
        self.feedback_max = feedback_max
        # The maximum of times g_best doesn't change -> need to change half of population
        self.n_changes = int(pop_size/2)

        ## Dynamic variable
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))    # Eq. 8
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.create_solution()
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        nfe_epoch += self.pop_size

        ## Feedback Mechanism
        _, current_best = self.get_global_best_solution(pop_new)
        if current_best[self.ID_FIT][self.ID_TAR] == self.g_best[self.ID_FIT][self.ID_TAR]:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = np.random.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_child = self.create_population(self.n_changes)
            nfe_epoch += self.n_changes
            for idx_counter, idx in enumerate(idx_list):
                pop_new[idx] = pop_child[idx_counter]
        self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch
