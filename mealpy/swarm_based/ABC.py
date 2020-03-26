#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:57, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from random import random, randint
from numpy import maximum, minimum
from copy import deepcopy
from mealpy.root import Root


class BaseABC(Root):
    """
    Taken from book: Clever Algorithms
    - Improved: _create_neigh_bee__
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.e_bees = couple_bees[0]                # number of bees which provided for good location and other location
        self.o_bees = couple_bees[1]
        self.patch_size = patch_variables[0]        # patch_variables = patch_variables * patch_factor (0.985)
        self.patch_factor = patch_variables[1]
        self.num_sites = sites[0]                   # 3 bees (employed bees, onlookers and scouts), 1 good partition
        self.elite_sites = sites[1]

    def _create_neigh_bee__(self, individual=None, patch_size=None):
        t1 = randint(0, len(individual) - 1)
        new_bee = deepcopy(individual)
        new_bee[t1] = (individual[t1] + random() * patch_size) if random() < 0.5 else (individual[t1] - random() * patch_size)
        new_bee[t1] = maximum(self.domain_range[0], minimum(self.domain_range[1], new_bee[t1]))
        return [new_bee, self._fitness_model__(new_bee)]


    def _search_neigh__(self, parent=None, neigh_size=None):  # parent:  [ vector_individual, fitness ]
        """
        Search 1 best solution in neigh_size solution
        """
        neigh = [self._create_neigh_bee__(parent[self.ID_POS], self.patch_size) for _ in range(0, neigh_size)]
        return self._get_global_best__(neigh, self.ID_FIT, self.ID_MIN_PROB)

    def _create_scout_bees__(self, num_scouts=None):
        return [self._create_solution__() for _ in range(0, num_scouts)]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            next_gen = []
            for i in range(0, self.num_sites):
                if i < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                next_gen.append(self._search_neigh__(pop[i], neigh_size))

            scouts = self._create_scout_bees__(self.pop_size - self.num_sites)
            pop = next_gen + scouts

            ## sort pop and update global best
            g_best, pop = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, patch_size: {}, Best fit: {}".format(epoch + 1, self.patch_size, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train