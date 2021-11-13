#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:08, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHC(Optimizer):
    """
    The original version of: Hill Climbing (HC)
    Noted:
        The number of neighbour solutions are equal to user defined
        The step size to calculate neighbour is randomized
    """

    def __init__(self, problem, epoch=10000, pop_size=100, neighbour_size=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 5,
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.neighbour_size = neighbour_size

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        self.nfe_per_epoch = self.neighbour_size
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * (epoch + 1) / self.epoch)
        pop_neighbours = []
        for i in range(0, self.neighbour_size):
            pos_new = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * step_size
            pos_new = self.amend_position_faster(pos_new)
            pop_neighbours.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_neighbours)


class BaseHC(OriginalHC):
    """
    The modified version of: Hill Climbing (HC) based on swarm-of people are trying to climb on the mountain ideas
    Noted:
        The number of neighbour solutions are equal to population size
        The step size to calculate neighbour is randomized and based on ranks of solution.
            + The guys near on top of mountain will move slower than the guys on bottom of mountain.
            + Imagine it is like: exploration when far from global best, and exploitation when near global best
        Who on top of mountain first will be the winner. (global optimal)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, neighbour_size=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 5,
        """
        super().__init__(problem, epoch, pop_size, neighbour_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ranks = np.array(list(range(1, self.pop_size + 1)))
        ranks = ranks / sum(ranks)
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * (epoch + 1) / self.epoch)

        for idx in range(0, self.pop_size):
            ss = step_size * ranks[idx]
            pop_neighbours = []
            for j in range(0, self.neighbour_size):
                pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * ss
                pos_new = self.amend_position_faster(pos_new)
                pop_neighbours.append([pos_new, None])
            pop_neighbours = self.update_fitness_population(pop_neighbours)
            _, agent = self.get_global_best_solution(pop_neighbours)
            self.pop[idx] = agent

