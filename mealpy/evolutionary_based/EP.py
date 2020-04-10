#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:27, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, choice, randint
from numpy import abs
from copy import deepcopy
from mealpy.root import Root


class BaseEP(Root):
    """
        The original version of: Evolutionary Programming (EP)
            (Clever Algorithms: Nature-Inspired Programming Recipes - Evolutionary Programming)
    Link:
        http://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, bout_size=0.05):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        if bout_size < 1:                               # Number of tried with tournament selection (5% of pop_size)
            self.bout_size = int(bout_size * self.pop_size)
        else:
            self.bout_size = int(bout_size)
        self.distance = 0.05 * (self.domain_range[1] - self.domain_range[0])

    def _create_solution__(self, minmax=0):
        pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(pos)
        strategy = uniform(0, self.distance, self.problem_size)
        times_win = 0
        return [pos, fit, strategy, times_win]

    def _mutate_solution__(self, solution=None):
        child = solution[self.ID_POS] + solution[self.ID_STR] * normal(0, 1.0, self.problem_size)
        child = self._amend_solution_faster__(child)
        fit = self._fitness_model__(child)
        s_old = solution[self.ID_STR] + normal(0, 1.0, self.problem_size) * abs(solution[self.ID_STR]) ** 0.5
        return [child, fit, s_old, 0]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            children = [self._mutate_solution__(pop[i]) for i in range(0, self.pop_size)]
            # Update the global best
            children, g_best = self._sort_pop_and_update_global_best__(children, self.ID_MIN_PROB, g_best)

            pop = children + pop
            for i in range(0, len(pop)):
                ## Tournament winner (Tried with bout_size times)
                for idx in range(0, self.bout_size):
                    rand_idx = randint(0, len(pop))
                    if pop[i][self.ID_FIT] < pop[rand_idx][self.ID_FIT]:
                        pop[i][self.ID_WIN] += 1

            pop = sorted(pop, key=lambda item: item[self.ID_WIN], reverse=True)
            pop = pop[:self.pop_size]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyEP(BaseEP):
    """
        The levy version of: Evolutionary Programming (EP)
    Noted:
        + Applied levy-flight
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, bout_size=0.05):
        BaseEP.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, bout_size)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            children = [self._mutate_solution__(pop[i]) for i in range(0, self.pop_size)]
            # Update the global best
            children, g_best = self._sort_pop_and_update_global_best__(children, self.ID_MIN_PROB, g_best)

            pop = children + pop
            for i in range(0, len(pop)):
                ## Tournament winner (Tried with bout_size times)
                for idx in range(0, self.bout_size):
                    rand_idx = randint(0, len(pop))
                    if pop[i][self.ID_FIT] < pop[rand_idx][self.ID_FIT]:
                        pop[i][self.ID_WIN] += 1

            ## Keep the top population, but 50% of left population will make a comeback an take the good position
            pop = sorted(pop, key=lambda item: item[self.ID_WIN], reverse=True)
            pop_left = deepcopy(pop[self.pop_size:])
            pop = pop[:self.pop_size]

            ## Choice random 50% of population left
            pop_comeback = []
            idx_list = choice(range(0, len(pop_left)), int(0.5*len(pop_left)), replace=False)
            for idx in idx_list:
                pos_new = self._levy_flight__(epoch, pop_left[idx][self.ID_POS], g_best[self.ID_POS])
                fit = self._fitness_model__(pos_new)
                strategy = self.distance = 0.05 * (self.domain_range[1] - self.domain_range[0])
                pop_comeback.append([pos_new, fit, strategy, 0])

            pop = pop + pop_comeback
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
