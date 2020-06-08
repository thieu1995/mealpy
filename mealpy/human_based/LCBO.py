#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import array, mean, sqrt, ceil
from mealpy.root import Root


class BaseLCBO(Root):
    """
    The original version of: Life Choice-Based Optimization (LCBO)
        (A novel life choice-based optimizer)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100, r1=2.35):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r1 = r1

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                rand = uniform()

                if rand > 0.875:    # Update using Eq. 1, update from n best position
                    n = int(ceil(sqrt(self.pop_size)))
                    temp = array([uniform() * pop[j][self.ID_POS] for j in range(0, n)])
                    temp = mean(temp, axis=0)
                elif rand < 0.7:    # Update using Eq. 2-6
                    f = (epoch + 1) / self.epoch
                    if i != 0:
                        better_diff = f * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        better_diff = f * self.r1 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = (1-f) * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                else:
                    temp = self.ub - (pop[i][self.ID_POS] - self.lb) * uniform(self.lb, self.ub)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

                ## Batch size idea
                if i % self.batch_size:
                    pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyLCBO(BaseLCBO):
    """
    The modified version of: Life Choice-Based Optimization (LCBO) based on Levy_flight
        (A novel life choice-based optimizer)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100, r1=2.35):
        BaseLCBO.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size, r1)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                rand = uniform()

                if rand > 0.875:  # Update using Eq. 1, update from n best position
                    n = int(ceil(sqrt(self.pop_size)))
                    pos_new = array([uniform() * pop[j][self.ID_POS] for j in range(0, n)])
                    pos_new = mean(pos_new, axis=0)
                elif rand < 0.7:  # Update using Eq. 2-6
                    if uniform() < 0.5:
                        f = (epoch + 1) / self.epoch
                        if i != 0:
                            better_diff = f * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                        else:
                            better_diff = f * self.r1 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        best_diff = (1 - f) * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                    else:
                        pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS], case=0)  # My new updated here
                else:
                    pos_new = self.ub - (pop[i][self.ID_POS] - self.lb) * uniform(self.lb, self.ub)

                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                ## Batch size idea
                if i % self.batch_size:
                    pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedLCBO(BaseLCBO):
    """
    The improved version of: Life Choice-Based Optimization (LCBO) based on new ideas
        (A novel life choice-based optimizer)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100, r1=2.35):
        BaseLCBO.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size, r1)
        self.n1 = int(ceil(sqrt(self.pop_size)))                        # n best position
        self.n2 = self.n1 + int((self.pop_size - self.n1) / 2)          # 50% for both 2 group left

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                ## Since we already sorted population, we know which ones are 1st group
                if i < self.n1:
                    pos_new = array([uniform() * pop[j][self.ID_POS] for j in range(0, self.n1)])
                    pos_new = mean(pos_new, axis=0)
                elif self.n1 <= i < self.n2:  # People in group 2 learning from the best person in the history,
                    # because they want to be better than the current best person
                    pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS], case=0)
                else:  # People in group 3 learning from the current best person and the person slightly better than them,
                    # because they don't have vision
                    f = 1 - (epoch + 1) / self.epoch
                    better_diff = f * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = (1 - f) * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    pos_new = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
                else:
                    pos_new = self.ub - (pop[i][self.ID_POS] - self.lb) * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(pos_new)
                    pop[i] = [pos_new, fit]

                ## Batch size idea
                if i % self.batch_size:
                    pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
