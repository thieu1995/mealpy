#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseNMR(Optimizer):
    """
    The original version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Link:
        https://www.doi.org10.1007/s00521-019-04464-7
    """

    def __init__(self, problem, epoch=10000, pop_size=100, bp=0.75, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            bp (float): breeding probability (0.75)
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = bp

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            if idx < self.size_b:  # breeding operators
                if np.random.uniform() < self.bp:
                    alpha = np.random.uniform()
                    pos_new = (1 - alpha) * self.pop[idx][self.ID_POS] + alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            else:  # working operators
                t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.pop[t1][self.ID_POS] - self.pop[t2][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class ImprovedNMR(BaseNMR):
    """
    My improved version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Notes:
        + Using mutation probability
        + Using levy-flight
        + Using crossover operator
    """

    def __init__(self, problem, epoch=10000, pop_size=100, bp=0.75, pm=0.01, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            bp (float): breeding probability (0.75)
            pm ():
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, bp, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        self.pm = pm

    def _crossover_random(self, pop, g_best):
        start_point = np.random.randint(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[np.random.randint(0, self.pop_size)][self.ID_POS]
        new_temp = deepcopy(g_best[self.ID_POS])
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            if idx < self.size_b:  # breeding operators
                if np.random.uniform() < self.bp:
                    pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                              (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    levy_step = self.get_levy_flight_step(beta=1, multiplier=0.001, case=-1)
                    pos_new = self.pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                              levy_step * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            # Exploitation
            else:  # working operators
                if np.random.uniform() < 0.5:
                    t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                    pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                              (self.pop[t1][self.ID_POS] - self.pop[t2][self.ID_POS])
                else:
                    pos_new = self._crossover_random(self.pop, self.g_best)
            # Mutation
            temp = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, temp, pos_new)
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
