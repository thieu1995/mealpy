#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:09, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCA(Optimizer):
    """
    The original version of: Culture Algorithm (CA)
        Based on Ruby version in the book: Clever Algorithm (Jason Brown)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, accepted_rate=0.15, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            accepted_rate (float): probability of accepted rate, Default: 0.15,
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.accepted_rate = accepted_rate

        ## Dynamic variables
        self.dyn_belief_space = {
            "lb": self.problem.lb,
            "ub": self.problem.ub,
        }
        self.dyn_accepted_num = int(self.accepted_rate * self.pop_size)
        # update situational knowledge (g_best here is a element inside belief space)

    def binary_tournament(self, population):
        id1, id2 = np.random.choice(list(range(0, len(population))), 2, replace=False)
        return population[id1] if (population[id1][self.ID_FIT] < population[id2][self.ID_FIT]) else population[id2]

    def create_faithful(self, lb, ub):
        position = np.random.uniform(lb, ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def update_belief_space(self, belief_space, pop_accepted):
        pos_list = np.array([solution[self.ID_POS] for solution in pop_accepted])
        belief_space["lb"] = np.min(pos_list, axis=0)
        belief_space["ub"] = np.max(pos_list, axis=0)
        return belief_space

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
        if mode != "sequential":
            print("CA optimizer is support sequential mode only!")
            exit(0)

        # create next generation
        pop_child = [self.create_faithful(self.dyn_belief_space["lb"], self.dyn_belief_space["ub"]) for _ in range(0, self.pop_size)]

        # select next generation
        pop = [self.binary_tournament(pop + pop_child) for _ in range(0, self.pop_size)]

        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=True)

        # Get accepted faithful
        accepted = pop[:self.dyn_accepted_num]

        # Update belief_space
        self.dyn_belief_space = self.update_belief_space(self.dyn_belief_space, accepted)

        return pop