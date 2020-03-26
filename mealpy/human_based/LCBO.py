#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import array, ones, mean, sqrt, ceil
from mealpy.root import Root


class BaseLCBO(Root):
    """
    The original version of: Life Choice-Based Optimization (LCBO)
        (A novel life choice-based optimizer)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r1 = 2.35

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                rand = uniform()

                if rand > 0.875:    # Update using Eq. 1, update from n best solution
                    n = int(ceil(sqrt(self.pop_size)))
                    temp = array([uniform() * pop[j][self.ID_POS] for j in range(0, n)])
                    temp = mean(temp, axis=0)
                    fit = self._fitness_model__(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = [temp, fit]
                elif rand < 0.7:    # Update using Eq. 2-6
                    f1 = 1 - (epoch + 1) / self.epoch
                    f2 = 1 - f1
                    if i != 0:
                        better_diff = f2 * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        better_diff = f2 * self.r1 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = f1 * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                    fit = self._fitness_model__(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = [temp, fit]
                else:
                    x_min = self.domain_range[0] * ones(self.problem_size)
                    x_max = self.domain_range[1] * ones(self.problem_size)
                    temp = x_max - (pop[i][self.ID_POS] - x_min) * uniform()
                    fit = self._fitness_model__(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = [temp, fit]

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
