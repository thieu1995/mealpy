#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseMSA(Optimizer):
    """
    My modified version of: Moth Search Algorithm (MSA)
        (Moth search algorithm: a bio-inspired metaheuristic algorithm for global optimization problems.)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        http://doi.org/10.1007/s12293-016-0212-3
    Notes:
        + Simply the matlab version above is not working (or bad at convergence characteristics).
        + Need to add normal random number (gaussian) in each updating equation. (Better performance)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_best=5, partition=0.5, max_step_size=1.0, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (): how many of the best moths to keep from one generation to the next
            partition (): The proportional of first partition
            max_step_size ():
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.n_best = n_best
        self.partition = partition
        self.max_step_size = max_step_size
        # np1 in paper
        self.n_moth1 = int(np.ceil(self.partition * self.pop_size))
        # np2 in paper, we actually don't need this variable
        self.n_moth2 = self.pop_size - self.n_moth1
        # you can change this ratio so as to get much better performance
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def _levy_walk(self, iteration):
        beta = 1.5      # Eq. 2.23
        sigma = (gamma(1+beta) * np.sin(np.pi*(beta-1)/2) / (gamma(beta/2) * (beta-1) * 2 ** ((beta-2) / 2))) ** (1/(beta-1))
        u = np.random.uniform(self.problem.lb, self.problem.ub) * sigma
        v = np.random.uniform(self.problem.lb, self.problem.ub)
        step = u / np.abs(v) ** (1.0 / (beta - 1))     # Eq. 2.21
        scale = self.max_step_size / (iteration+1)
        delta_x = scale * step
        return delta_x

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_best = deepcopy(self.pop[:self.n_best])

        pop_new = []
        for idx in range(0, self.pop_size):
            # Migration operator
            if idx < self.n_moth1:
                # scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
                pos_new = self.pop[idx][self.ID_POS] + np.random.normal() * self._levy_walk(epoch)
            else:
                # Flying in a straight line
                temp_case1 = self.pop[idx][self.ID_POS] + np.random.normal() * \
                             self.golden_ratio * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                temp_case2 = self.pop[idx][self.ID_POS] + np.random.normal() * \
                             (1.0 / self.golden_ratio) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = np.where(np.random.uniform(self.problem.n_dims) < 0.5, temp_case2, temp_case1)
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)

        self.pop, _ = self.get_global_best_solution(pop_new)
        # Replace the worst with the previous generation's elites.
        for i in range(0, self.n_best):
            self.pop[-1 - i] = deepcopy(pop_best[i])
