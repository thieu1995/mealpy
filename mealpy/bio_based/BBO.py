#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:24, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalBBO(Optimizer):
    """
    The original version of: Biogeography-based optimization (BBO)
        Biogeography-Based Optimization
    Link:
        https://ieeexplore.ieee.org/abstract/document/4475427
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_m=0.01, elites=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_m (float): mutation probability, default=0.01
            elites (int): Number of elites will be keep for next generation, default=2
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.p_m = p_m
        self.elites = elites

        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        self.mr = 1 - self.mu

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        _, pop_elites, _ = self.get_special_solutions(self.pop, best=self.elites)
        pop = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(self.problem.n_dims):
                if np.random.uniform() < self.mr[idx]:  # Should we immigrate?
                    # Pick a position from which to emigrate (roulette wheel selection)
                    random_number = np.random.uniform() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (select_index < self.pop_size - 1):
                        select_index += 1
                        select += self.mu[select_index]
                    # this is the migration step
                    pos_new[j] = self.pop[select_index][self.ID_POS][j]

            noise = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, noise, pos_new)
            pos_new = self.amend_position_faster(pos_new)
            pop.append([pos_new, None])

        pop = self.update_fitness_population(pop)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.pop = self.get_sorted_strim_population(pop + pop_elites, self.pop_size)


class BaseBBO(OriginalBBO):
    """
    My version of: Biogeography-based optimization (BBO)
        Biogeography-Based Optimization
    Link:
        https://ieeexplore.ieee.org/abstract/document/4475427
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_m=0.01, elites=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_m (float): mutation probability, default=0.01
            elites (int): Number of elites will be keep for next generation, default=2
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, p_m, elites, **kwargs)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        _, pop_elites, _ = self.get_special_solutions(self.pop, best=self.elites)
        list_fitness = [agent[self.ID_FIT][self.ID_TAR] for agent in self.pop]
        pop = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            # Pick a position from which to emigrate (roulette wheel selection)
            idx_selected = self.get_index_roulette_wheel_selection(list_fitness)
            # this is the migration step
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.mr[idx], self.pop[idx_selected][self.ID_POS], self.pop[idx][self.ID_POS])
            # Mutation
            temp = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, temp, pos_new)
            pos_new = self.amend_position_faster(pos_new)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)
        # Replace the solutions with their new migrated and mutated versions then merge populations
        self.pop = self.get_sorted_strim_population(pop + pop_elites, self.pop_size)

