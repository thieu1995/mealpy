#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalLCO(Optimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)
        (A novel life choice-based optimizer)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.r1 = r1
        self.n_agents = int(np.ceil(np.sqrt(self.pop_size)))

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for i in range(0, self.pop_size):
            rand_number = np.random.random()

            if rand_number > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.random() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif rand_number < 0.7:  # Update using Eq. 2-6
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                if i == 0:
                    pop_new.append(deepcopy(self.g_best))
                    continue
                else:
                    best_diff = f1 * self.r1 * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                    better_diff = f2 * self.r1 * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                temp = self.pop[i][self.ID_POS] + np.random.random() * better_diff + np.random.random() * best_diff
            else:
                temp = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.random()
            pos_new = self.amend_position_faster(temp)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class BaseLCO(OriginalLCO):
    """
    My base version of: Life Choice-based Optimization (LCO)
        (A novel life choice-based optimizer)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, r1, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for i in range(0, self.pop_size):
            rand = np.random.random()

            if rand > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.random() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if i != 0:
                    better_diff = f * self.r1 * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                else:
                    better_diff = f * self.r1 * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                best_diff = (1 - f) * self.r1 * (self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS])
                temp = self.pop[i][self.ID_POS] + np.random.uniform() * better_diff + np.random.uniform() * best_diff
            else:
                temp = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position_faster(temp)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class ImprovedLCO(Optimizer):
    """
    The improved version of: Life Choice-Based Optimization (LCO) based on
        + Gaussian distribution
        + Mutation Mechanism
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
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

        self.pop_len = int(self.pop_size / 2)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for i in range(0, self.pop_size):
            rand = np.random.random()
            if rand > 0.875:  # Update using Eq. 1, update from n best position
                n = int(np.ceil(np.sqrt(self.pop_size)))
                pos_new = np.array([np.random.uniform() * self.pop[j][self.ID_POS] for j in range(0, n)])
                pos_new = np.mean(pos_new, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if i != 0:
                    better_diff = f * np.random.uniform() * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                else:
                    better_diff = f * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                best_diff = (1 - f) * np.random.uniform() * (self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS])
                pos_new = self.pop[i][self.ID_POS] + better_diff + best_diff
            else:
                pos_new = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)

        ## Sort the updated population based on fitness
        pop, local_best = self.get_global_best_solution(pop_new)
        pop_s1, pop_s2 = pop[:self.pop_len], pop[self.pop_len:]

        ## Mutation scheme
        for i in range(0, self.pop_len):
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))
            pop_s1[i][self.ID_POS] = self.amend_position_faster(pos_new)
        pop_s1 = self.update_fitness_population(pop_s1)

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        for i in range(0, self.pop_len):
            pos_new = (local_best[self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            pop_s2[i][self.ID_POS] = pos_new
        pop_s2 = self.update_fitness_population(pop_s2)

        ## Construct a new population
        self.pop = pop_s1 + pop_s2
