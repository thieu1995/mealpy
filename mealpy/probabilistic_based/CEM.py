#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:08, 19/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseCEM(Optimizer):
    """
        The original version of: Cross-Entropy Method (CEM)
            https://github.com/clever-algorithms/CleverAlgorithms
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_best=30, alpha=0.7, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best ():
            alpha ():
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.n_best = n_best
        self.means = np.random.uniform(self.problem.lb, self.problem.ub)
        self.stdevs = np.abs(self.problem.ub - self.problem.lb)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Selected the best samples and update means and stdevs
        pop_best = self.pop[:self.n_best]
        pos_list = np.array([item[self.ID_POS] for item in pop_best])

        means_new = np.mean(pos_list, axis=0)
        means_new_repeat = np.repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
        stdevs_new = np.mean((pos_list - means_new_repeat) ** 2, axis=0)
        self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
        self.stdevs = np.abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

        ## Create new population for next generation
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.normal(self.means, self.stdevs)
            pop_new.append([self.amend_position_faster(pos_new), None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)

