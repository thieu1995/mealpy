#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import where
from numpy.random import uniform, choice, randint, normal
from copy import deepcopy
from mealpy.root import Root


class BaseNMR(Root):
    """
    The original version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Link:
        https://www.doi.org10.1007/s00521-019-04464-7
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, bp=0.75, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = bp                                # breeding probability (0.75)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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

                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyNMR(BaseNMR):
    """
    My levy-flight version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Link:
        https://www.doi.org10.1007/s00521-019-04464-7

    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, bp=0.75, **kwargs):
        BaseNMR.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, bp, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                # Exploration
                if i < self.size_b:  # breeding operators
                    if uniform() < self.bp:
                        temp = pop[i][self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                # Exploitation
                else:  # working operators
                    if uniform() < 0.5:
                        t1, t2 = choice(range(0, self.size_b), 2, replace=False)
                        temp = pop[i][self.ID_POS] + uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
                    else:
                        temp = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])

                temp = self.amend_position_faster(temp)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedNMR(BaseNMR):
    """
    My improved version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Notes:
        + Using mutation probability
        + Using levy-flight
        + Using crossover operator
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, bp=0.75, pm=0.01, **kwargs):
        BaseNMR.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, bp, kwargs=kwargs)
        self.pm = pm

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

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                # Exploration
                if i < self.size_b:  # breeding operators
                    if uniform() < self.bp:
                        pos_new = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])


                # Exploitation
                else:  # working operators
                    if uniform() < 0.5:
                        t1, t2 = choice(range(self.size_b, self.pop_size), 2, replace=False)
                        pos_new = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
                    else:
                        pos_new = self._crossover_random__(pop, g_best)

                # Mutation
                temp = uniform(self.lb, self.ub)
                pos_new = where(uniform(0, 1, self.problem_size) < self.pm, temp, pos_new)
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
