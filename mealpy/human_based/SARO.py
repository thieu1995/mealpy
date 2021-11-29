#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSARO(Optimizer):
    """
    My version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    Notes:
        + Remove all third loop
    """
    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.se = se
        self.mu = mu

        ## Dynamic variable
        self.dyn_USN = np.zeros(self.pop_size)

    def initialization(self):
        pop = self.create_population(pop_size=(2 * self.pop_size))
        self.pop, self.g_best = self.get_global_best_solution(pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_x = deepcopy(self.pop[:self.pop_size])
        pop_m = deepcopy(self.pop[self.pop_size:])

        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - self.pop[k][self.ID_POS]

            #### Remove third loop here, also using random flight back when out of bound
            pos_new_1 = self.pop[k][self.ID_POS] + np.random.uniform() * sd
            pos_new_2 = pop_x[idx][self.ID_POS] + np.random.uniform() * sd
            pos_new = np.where(np.logical_and(np.random.uniform(0, 1, self.problem.n_dims) < self.se,
                                              self.pop[k][self.ID_FIT] < pop_x[idx][self.ID_FIT]), pos_new_1, pos_new_2)
            pos_new = self.amend_position_random(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        pop = deepcopy(pop_x) + deepcopy(pop_m)
        pop_new = []
        for idx in range(self.pop_size):
            ## Individual phase
            k1, k2 = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            #### Remove third loop here, and flight back strategy now be a random
            pos_new = self.g_best[self.ID_POS] + np.random.uniform() * (pop[k1][self.ID_POS] - pop[k2][self.ID_POS])
            pos_new = self.amend_position_random(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution()
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m


class OriginalSARO(BaseSARO):
    """
    The original version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    """

    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, epoch, pop_size, se, mu, **kwargs)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_x = deepcopy(self.pop[:self.pop_size])
        pop_m = deepcopy(self.pop[self.pop_size:])

        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - self.pop[k][self.ID_POS]
            j_rand = np.random.randint(0, self.problem.n_dims)
            r1 = np.random.uniform(-1, 1)

            pos_new = deepcopy(pop_x[idx][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() < self.se or j == j_rand:
                    if self.compare_agent(self.pop[k], pop_x[idx]):
                        pos_new[j] = self.pop[k][self.ID_POS][j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[idx][self.ID_POS][j] + r1 * sd[j]

                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        ## Individual phase
        pop = deepcopy(pop_x) + deepcopy(pop_m)
        pop_new = []
        for idx in range(0, self.pop_size):
            k, m = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            pos_new = pop_x[idx][self.ID_POS] + np.random.uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx]
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution()
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m
