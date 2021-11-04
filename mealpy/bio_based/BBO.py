#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:24, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseBBO(Optimizer):
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
            p_m (): mutation probability
            elites (): Number of elites will be keep for next generation
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

    def create_child(self, idx, pop, list_fitness):
        # Probabilistic migration to the i-th position
        # Pick a position from which to emigrate (roulette wheel selection)
        idx_selected = self.get_index_roulette_wheel_selection(list_fitness)
        # this is the migration step
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.mr[idx], pop[idx_selected][self.ID_POS], pop[idx][self.ID_POS])
        # Mutation
        temp = np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, temp, pos_new)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

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
        pop_elites, local_best = self.get_global_best_solution(pop)
        list_fitness = [agent[self.ID_FIT][self.ID_TAR] for agent in pop]
        pop_idx = np.array(range(0, self.pop_size))
        # Use migration rates to decide how much information to share between solutions
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, list_fitness=list_fitness), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, list_fitness=list_fitness), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, list_fitness) for idx in pop_idx]

        # replace the solutions with their new migrated and mutated versions then Merge Populations
        pop = self.get_sorted_strim_population(child + pop_elites, self.pop_size)
        return pop


class OriginalBBO(BaseBBO):
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
            p_m (): mutation probability
            elites (): Number of elites will be keep for next generation
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)

    def create_child(self, idx, pop, list_fitness):
        # Probabilistic migration to the i-th position
        pos_new = pop[idx][self.ID_POS].copy()
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
                pos_new[j] = pop[select_index][self.ID_POS][j]

        noise = np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, noise, pos_new)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]
