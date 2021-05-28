#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:57, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import maximum, minimum
from copy import deepcopy
from mealpy.root import Root


class BaseABC(Root):
    """
        My version of: Artificial Bee Colony (ABC)
            + Taken from book: Clever Algorithms
            + Improved: _create_neigh_bee__ function
        Link:
            https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
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
        new_bee[t1] = (individual[t1] + uniform() * patch_size) if uniform() < 0.5 else (individual[t1] - uniform() * patch_size)
        new_bee[t1] = maximum(self.lb[t1], minimum(self.ub[t1], new_bee[t1]))
        return [new_bee, self.get_fitness_position(new_bee)]

    def _search_neigh__(self, parent=None, neigh_size=None):  # parent:  [ vector_individual, fitness ]
        """
        Search 1 best position in neigh_size position
        """
        neigh = [self._create_neigh_bee__(parent[self.ID_POS], self.patch_size) for _ in range(0, neigh_size)]
        return self.get_global_best_solution(neigh, self.ID_FIT, self.ID_MIN_PROB)

    def _create_scout_bees__(self, num_scouts=None):
        return [self.create_solution() for _ in range(0, num_scouts)]

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, patch_size: {}, Best fit: {}".format(epoch + 1, round(self.patch_size, 4), g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train