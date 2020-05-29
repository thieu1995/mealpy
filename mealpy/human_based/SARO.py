#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import randint, uniform, choice
from numpy import mean, array, zeros
from copy import deepcopy
from mealpy.root import Root


class BaseSARO(Root):
    """
    The original version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, se=0.5, mu=50):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.se = se
        self.mu = mu

    def _calculate_mean__(self, pop=None):
        temp = mean(array([item[self.ID_POS] for item in pop]), axis=0)
        return temp

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size * 2)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_x = deepcopy(pop[:self.pop_size])
        pop_m = deepcopy(pop[self.pop_size:])
        USN = zeros((self.pop_size))

        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Social Phase
                while True:
                    k = randint(0, 2*self.pop_size)
                    if k != i:
                        break
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

                    if temp_ij[j] < self.domain_range[0]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.domain_range[0]) / 2
                    if temp_ij[j] > self.domain_range[1]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.domain_range[1]) / 2
                fit_ij = self._fitness_model__(temp_ij)

                if fit_ij < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0
                else:
                    USN[i] += 1

                ## Individual phase
                pop = deepcopy(pop_x + pop_m)

                while True:
                    k, m = choice(range(0, 2*self.pop_size), 2, replace=False)
                    if k != m and k != i:
                        break
                temp_ij = pop_x[i][self.ID_POS] + uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
                for j in range(0, self.problem_size):
                    if temp_ij[j] < self.domain_range[0]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.domain_range[0]) / 2
                    if temp_ij[j] > self.domain_range[1]:
                        temp_ij[j] = (pop_x[i][self.ID_POS][j] + self.domain_range[1]) / 2

                fit_ij = self._fitness_model__(temp_ij)
                if fit_ij < pop_x[i][self.ID_FIT]:
                    pop_m[randint(0, self.pop_size)] = deepcopy(pop_x[i])
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0
                else:
                    USN[i] += 1

                if USN[i] > self.mu:
                    temp_ij = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit_ij = self._fitness_model__(temp_ij)
                    pop_x[i] = [temp_ij, fit_ij]
                    USN[i] = 0

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
