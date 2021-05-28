#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import randint, uniform, choice
from numpy import zeros, where, logical_and
from copy import deepcopy
from mealpy.root import Root


class BaseSARO(Root):
    """
    My version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    Notes:
        + Remove all third loop
        + Using batch-size idea
        + Update whole position at the same time, but this seem make this algorithm less efficient than the original
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, se=0.5, mu=50, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.se = se    # social effect
        self.mu = mu    # maximum unsuccessful search number

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size * 2)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_x = deepcopy(pop[:self.pop_size])
        pop_m = deepcopy(pop[self.pop_size:])
        USN = zeros(self.pop_size)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                ## Social Phase
                k = choice(list(set(range(0, 2 * self.pop_size)) - {i}))
                sd = pop_x[i][self.ID_POS] - pop[k][self.ID_POS]

                #### Remove third loop here, also using random flight back when out of bound
                pos_new_1 = pop[k][self.ID_POS] + uniform() * sd
                pos_new_2 = pop_x[i][self.ID_POS] + uniform() * sd
                pos_new = where(logical_and(uniform(0, 1, self.problem_size) < self.se, pop[k][self.ID_FIT] < pop_x[i][self.ID_FIT]), pos_new_1, pos_new_2)
                pos_new = self.amend_position_random_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)

                if fit_new < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [pos_new, fit_new]
                    USN[i] = 0
                else:
                    USN[i] += 1

                ## Individual phase
                pop = deepcopy(pop_x + pop_m)
                k1, k2 = choice(list(set(range(0, 2 * self.pop_size)) - {i}), 2, replace=False)

                #### Remove third loop here, and flight back strategy now be a random
                pos_new = g_best[self.ID_POS] + uniform() * (pop[k1][self.ID_POS] - pop[k2][self.ID_POS])
                pos_new = self.amend_position_random_faster(pos_new)

                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [pos_new, fit_new]
                    USN[i] = 0
                else:
                    USN[i] += 1

                if USN[i] > self.mu:
                    pos_new = uniform(self.lb, self.ub)
                    fit_new = self.get_fitness_position(pos_new)
                    pop_x[i] = [pos_new, fit_new]
                    USN[i] = 0

                pop = deepcopy(pop_x + pop_m)

                # batch-size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSARO(BaseSARO):
    """
    The original version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, se=0.5, mu=50, **kwargs):
        BaseSARO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, se, mu, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size * 2)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_x = deepcopy(pop[:self.pop_size])
        pop_m = deepcopy(pop[self.pop_size:])
        USN = zeros(self.pop_size)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Social Phase
                k = choice(list(set(range(0, 2 * self.pop_size)) - {i}))
                sd = pop_x[i][self.ID_POS] - pop[k][self.ID_POS]
                j_rand = randint(0, self.problem_size)
                r1 = uniform(-1, 1)

                temp_ij = deepcopy(pop_x[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    if uniform() < self.se or j == j_rand:
                        if pop[k][self.ID_FIT] < pop_x[i][self.ID_FIT]:
                            temp_ij[j] = pop[k][self.ID_POS][j] + r1 * sd[j]
                        else:
                            temp_ij[j] = pop_x[i][self.ID_POS][j] + r1 * sd[j]

                    if temp_ij[j] < self.lb[j]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.lb[j]) / 2
                    if temp_ij[j] > self.ub[j]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.ub[j]) / 2
                fit_ij = self.get_fitness_position(temp_ij)

                if fit_ij < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0
                else:
                    USN[i] += 1

                ## Individual phase
                pop = deepcopy(pop_x + pop_m)

                k, m = choice(list(set(range(0, 2 * self.pop_size)) - {i}), 2, replace=False)
                temp_ij = pop_x[i][self.ID_POS] + uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
                for j in range(0, self.problem_size):
                    if temp_ij[j] < self.lb[j]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.lb[j]) / 2
                    if temp_ij[j] > self.ub[j]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.ub[j]) / 2

                fit_ij = self.get_fitness_position(temp_ij)
                if fit_ij < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0
                else:
                    USN[i] += 1

                if USN[i] > self.mu:
                    temp_ij = uniform(self.lb, self.ub)
                    fit_ij = self.get_fitness_position(temp_ij)
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0
                pop = deepcopy(pop_x + pop_m)

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
