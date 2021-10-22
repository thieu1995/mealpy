#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:08, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
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
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 5,
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.neighbour_size = neighbour_size

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
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * (epoch + 1) / self.epoch)

        if mode != "sequential":
            print("Original HC algorithm only support sequential process!")
            exit(0)
        pop_neighbours = []
        for i in range(0, self.neighbour_size):
            pos_new = pos_new = g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * step_size
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            pop_neighbours.append([pos_new, fit_new])
        pop_neighbours.append(g_best)
        return pop_neighbours


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
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 5,
        """
        super().__init__(problem, epoch, pop_size, neighbour_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.neighbour_size = neighbour_size

    def create_child(self, idx, pop, g_best, step_size, ranks):
        ss = step_size * ranks[idx]
        pop_neighbours = []
        for j in range(0, self.neighbour_size):
            pos_new = pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * ss
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            pop_neighbours.append([pos_new, fit_new])
        pop_neighbours.append(g_best)
        _, agent = self.get_global_best_solution(pop_neighbours)
        return agent

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

        ranks = np.array(list(range(1, self.pop_size + 1)))
        ranks = ranks / sum(ranks)
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * (epoch + 1) / self.epoch)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, step_size=step_size, ranks=ranks), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, step_size=step_size, ranks=ranks), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, g_best, step_size, ranks) for idx in pop_idx]
        return pop_new


