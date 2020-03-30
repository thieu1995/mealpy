#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, randint
from copy import deepcopy
from mealpy.root import Root


class BaseNMR(Root):
    """
    The original version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Link:
        https://www.doi.org10.1007/s00521-019-04464-7
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, bp=0.75):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = bp                                # breeding probability (0.75)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                if i < self.size_b:                     # breeding operators
                    if uniform() < self.bp:
                        alpha = uniform()
                        temp = (1 - alpha) * pop[i][self.ID_POS] + alpha * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                else:                                   # working operators
                    t1, t2 = choice(range(self.size_b, self.pop_size), 2, replace=False)
                    temp = pop[i][self.ID_POS] + uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])

                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyNMR(BaseNMR):
    """
    My speedup version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, bp=0.75):
        BaseNMR.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, bp)
        self.pm = 0.025

    def _crossover__(self, solution, g_best):
        start_point = randint(0, self.problem_size / 2)
        id1 = start_point
        id2 = int(start_point + self.problem_size / 3)
        id3 = int(self.problem_size)

        new_temp = deepcopy(solution[self.ID_POS])
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = solution[self.ID_POS][id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp

    def _crossover_random__(self, pop, g_best):
        start_point = randint(0, self.problem_size / 2)
        id1 = start_point
        id2 = int(start_point + self.problem_size / 3)
        id3 = int(self.problem_size)

        partner = pop[randint(0, self.pop_size)][self.ID_POS]
        new_temp = deepcopy(g_best[self.ID_POS])
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp

    ### Mutation
    def _mutation_flip_point__(self, parent, index):
        w = deepcopy(parent)
        w[index] = uniform(self.domain_range[0], self.domain_range[1])
        return w

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                # Exploration
                if i < self.size_b:  # breeding operators
                    if uniform() < self.bp:
                        alpha = uniform()
                        temp = pop[i][self.ID_POS] + alpha * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        #temp = self._crossover__(pop[i], g_best)
                        temp = self._crossover_random__(pop, g_best)

                # Exploitation
                else:  # working operators
                    if uniform() < 0.5:
                        t1, t2 = choice(range(self.size_b, self.pop_size), 2, replace=False)
                        temp = pop[i][self.ID_POS] + uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
                    else:
                        temp = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                    # Mutation
                    for id in range(0, self.problem_size):
                        if uniform() < self.pm:
                            temp = self._mutation_flip_point__(temp, id)

                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
