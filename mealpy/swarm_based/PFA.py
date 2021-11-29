#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BasePFA(Optimizer):
    """
        The original version of: Pathfinder Algorithm (PFA)
            (A new meta-heuristic optimizer: Pathfinder algorithm)
        Link:
            https://doi.org/10.1016/j.asoc.2019.03.012
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
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        alpha, beta = np.random.uniform(1, 2, 2)
        A = np.random.uniform(self.problem.lb, self.problem.ub) * np.exp(-2 * (epoch + 1) / self.epoch)

        ## Update the position of pathfinder and check the bound
        temp = self.pop[0][self.ID_POS] + 2 * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[0][self.ID_POS]) + A
        temp = self.amend_position_faster(temp)
        fit = self.get_fitness_position(temp)
        pop_new = [[temp, fit], ]

        ## Update positions of members, check the bound and calculate new fitness
        for idx in range(1, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for k in range(1, self.pop_size):
                dist = np.sqrt(np.sum((self.pop[k][self.ID_POS] - self.pop[idx][self.ID_POS]) ** 2)) / self.problem.n_dims
                t2 = alpha * np.random.uniform() * (self.pop[k][self.ID_POS] - self.pop[idx][self.ID_POS])
                ## First stabilize the distance
                t3 = np.random.uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.problem.ub - self.problem.lb))
                pos_new += t2 + t3
            ## Second stabilize the population size
            t1 = beta * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (pos_new + t1) / self.pop_size
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
