#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:57, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseABC(Optimizer):
    """
        My version of: Artificial Bee Colony (ABC)
            + Taken from book: Clever Algorithms
            + Improved: _create_neigh_bee__ function
        Link:
            https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony
    """

    def __init__(self, problem, epoch=10000, pop_size=100, couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            couple_bees (list): number of bees which provided for good location and other location
            patch_variables (list): patch_variables = patch_variables * patch_factor (0.985)
            sites (list): 3 bees (employed bees, onlookers and scouts), 1 good partition
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.e_bees = couple_bees[0]
        self.o_bees = couple_bees[1]
        self.patch_size = patch_variables[0]
        self.patch_factor = patch_variables[1]
        self.num_sites = sites[0]
        self.elite_sites = sites[1]

        self.nfe_per_epoch = self.e_bees * self.elite_sites + self.o_bees * self.num_sites + (self.pop_size - self.num_sites)
        self.sort_flag = True

    def _create_neigh_bee__(self, individual=None, patch_size=None):
        t1 = np.random.randint(0, len(individual) - 1)
        new_bee = individual.copy()
        new_bee[t1] = (individual[t1] + np.random.uniform() * patch_size) if np.random.uniform() < 0.5 else (individual[t1] - np.random.uniform() * patch_size)
        new_bee[t1] = np.maximum(self.problem.lb[t1], np.minimum(self.problem.ub[t1], new_bee[t1]))
        return [new_bee, self.get_fitness_position(new_bee)]

    def _search_neigh__(self, parent=None, neigh_size=None):
        """
        Search 1 best position in neigh_size position
        """
        neigh = [self._create_neigh_bee__(parent[self.ID_POS], self.patch_size) for _ in range(0, neigh_size)]
        _, current_best = self.get_global_best_solution(neigh)
        return current_best

    def create_child(self, idx, pop):
        if idx < self.num_sites:
            if idx < self.elite_sites:
                neigh_size = self.e_bees
            else:
                neigh_size = self.o_bees
            return self._search_neigh__(pop[idx], neigh_size)
        else:
            return self.create_solution()

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
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop) for idx in pop_idx]
        return child
