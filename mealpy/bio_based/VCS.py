#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:07, 11/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseVCS(Optimizer):
    """
        My version of: Virus Colony Search (VCS)
            A Novel Nature-inspired Algorithm For Optimization: Virus Colony Search
        Link:
            https://doi.org/10.1016/j.advengsoft.2015.11.004
        Notes:
            + Remove all third loop, make algrithm 10 times faster than original
            + In Immune response process, updating whole position instead of updating each variable in position
            + Drop batch-size idea to 3 main process of this algorithm, make it more robust
    """

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            xichma (float): Weight factor, default = 0.3
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.xichma = xichma
        self.lamda = lamda
        if lamda < 1:
            self.n_best = int(lamda * self.pop_size)
        else:
            self.n_best = int(lamda)

    def _calculate_xmean(self, pop):
        ## Calculate the weighted mean of the λ best individuals by
        pop, local_best = self.get_global_best_solution(pop)
        pos_list = [agent[self.ID_POS] for agent in pop[:self.n_best]]
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(np.prod(range(1, self.n_best + 1)))
        weight = np.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        x_mean = weight * np.sum(pos_list, axis=0)
        return x_mean

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
        ## Viruses diffusion
        for i in range(0, self.pop_size):
            xichma = (np.log1p(epoch + 1) / self.epoch) * (pop[i][self.ID_POS] - g_best[self.ID_POS])
            gauss = np.random.normal(np.random.normal(g_best[self.ID_POS], np.abs(xichma)))
            pos_new = gauss + np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * pop[i][self.ID_POS]
            pos_new = self.amend_position_random(pos_new)
            pop[i][self.ID_POS] = pos_new
        pop = self.update_fitness_population(mode, pop)

        ## Host cells infection
        x_mean = self._calculate_xmean(pop)
        xichma = self.xichma * (1 - (epoch + 1) / self.epoch)
        for i in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + xichma * np.random.normal(0, 1, self.problem.n_dims)
            pos_new = self.amend_position_random(pos_new)
            pop[i][self.ID_POS] = pos_new
        pop = self.update_fitness_population(mode, pop)

        ## Calculate the weighted mean of the λ best individuals by
        pop, g_best = self.get_global_best_solution(pop)
        pos_list = [agent[self.ID_POS] for agent in pop[:self.n_best]]

        ## Immune response
        for i in range(0, self.pop_size):
            pr = (self.problem.n_dims - i + 1) / self.problem.n_dims
            id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
            temp = pop[id1][self.ID_POS] - (pop[id2][self.ID_POS] - pop[i][self.ID_POS]) * np.random.uniform()
            pos_new = pop[i][self.ID_POS].copy()
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < pr, pos_new, temp)
            pop[i][self.ID_POS] = self.amend_position_faster(pos_new)
        pop = self.update_fitness_population(mode, pop)
        return pop


class OriginalVCS(BaseVCS):
    """
        The original version of: Virus Colony Search (VCS)
            A Novel Nature-inspired Algorithm For Optimization: Virus Colony Search
            - This is basic version, not the full version of the paper
        Link:
            https://doi.org/10.1016/j.advengsoft.2015.11.004
    """

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            xichma (float): Weight factor, default = 0.3
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, lamda, xichma, **kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

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
        ## Viruses diffusion
        for i in range(0, self.pop_size):
            xichma = (np.log1p(epoch + 1) / self.epoch) * (pop[i][self.ID_POS] - g_best[self.ID_POS])
            gauss = np.array([np.random.normal(g_best[self.ID_POS][idx], np.abs(xichma[idx])) for idx in range(0, self.problem.n_dims)])
            pos_new = gauss + np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * pop[i][self.ID_POS]
            pop[i][self.ID_POS] = self.amend_position_random(pos_new)
        pop = self.update_fitness_population(mode, pop)

        ## Host cells infection
        x_mean = self._calculate_xmean(pop)
        xichma = self.xichma * (1 - (epoch + 1) / self.epoch)
        for i in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + xichma * np.random.normal(0, 1, self.problem.n_dims)
            pos_new = self.amend_position_random(pos_new)
            pop[i][self.ID_POS] = pos_new
        pop = self.update_fitness_population(mode, pop)

        ## Immune response
        for i in range(0, self.pop_size):
            pr = (self.problem.n_dims - i + 1) / self.problem.n_dims
            pos_new = pop[i][self.ID_POS]
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() > pr:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    pos_new[j] = pop[id1][self.ID_POS][j] - (pop[id2][self.ID_POS][j] - pop[i][self.ID_POS][j]) * np.random.uniform()
            fit = self.get_fitness_position(pos_new)
            if fit < pop[i][self.ID_FIT]:
                pop[i] = [pos_new, fit]
        return pop

