#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from math import gamma
import concurrent.futures as parallel
from functools import partial
import numpy as np
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

    def _levy_walk__(self, iteration):
        beta = 1.5      # Eq. 2.23
        sigma = (gamma(1+beta) * np.sin(np.pi*(beta-1)/2) / (gamma(beta/2) * (beta-1) * 2 ** ((beta-2) / 2))) ** (1/(beta-1))
        u = np.random.uniform(self.problem.lb, self.problem.ub) * sigma
        v = np.random.uniform(self.problem.lb, self.problem.ub)
        step = u / np.abs(v) ** (1.0 / (beta - 1))     # Eq. 2.21
        scale = self.max_step_size / (iteration+1)
        delta_x = scale * step
        return delta_x

    def create_child(self, idx, pop, epoch, g_best):
        # Migration operator
        if idx < self.n_moth1:
            # scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
            pos_new = pop[idx][self.ID_POS] + np.random.normal() * self._levy_walk__(epoch)
        else:
        # Flying in a straight line
            temp_case1 = pop[idx][self.ID_POS] + np.random.normal() * self.golden_ratio * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
            temp_case2 = pop[idx][self.ID_POS] + np.random.normal() * (1.0 / self.golden_ratio) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
            pos_new = np.where(np.random.uniform(self.problem.n_dims) < 0.5, temp_case2, temp_case1)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

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
        pop_best = pop[:self.n_best].copy()
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, epoch, g_best) for idx in pop_idx]
        pop, _ = self.get_global_best_solution(child)
        # Replace the worst with the previous generation's elites.
        for i in range(0, self.n_best):
            pop[-1 - i] = pop_best[i].copy()
        return pop

