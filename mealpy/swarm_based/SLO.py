#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:47, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, normal, random
from numpy import abs, sign, cos, pi, sin, sqrt, power
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseSLO(Root):
    """
        The original version of: Sea Lion Optimization Algorithm
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                c = 2 - 2 * epoch / self.epoch

                SP_leader = uniform(0, 1)
                if SP_leader >= 0.6:
                    new_pos = cos(2 * pi * uniform(-1, 1)) * abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) + g_best[self.ID_POS]
                else:
                    b = uniform(0, 1, self.problem_size)
                    if c <= 1:
                        new_pos = g_best[self.ID_POS] - c * b * abs(2 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        rand_index = randint(0, self.pop_size)
                        rand_SL = pop[rand_index]
                        new_pos = rand_SL[self.ID_POS] - c * abs(b * rand_SL[self.ID_POS] - pop[i][self.ID_POS])

                new_pos = self._amend_solution_random_faster__(new_pos)
                fit = self._fitness_model__(new_pos)
                pop[i] = [new_pos, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedSLO(BaseSLO):
    """
        The improved version of: Sea Lion Optimization (ISLO)
            (Sea Lion Optimization Algorithm)
        Noted:
            + Using the idea of shrink encircling combine with levy flight techniques
            + Also using the idea of local best in PSO
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseSLO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * sin(pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2))
        xich_ma_1 = power(up / down, 1 / beta)
        xich_ma_2 = 1
        a = normal(0, xich_ma_1, 1)
        b = normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (power(abs(b), 1 / beta)) * dist * c
        D = uniform(self.domain_range[0], self.domain_range[1], 1)
        levy = LB * D
        return (current_pos - sqrt(epoch + 1) * sign(random(1) - 0.5)) * levy

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                c = 2 - 2 * epoch / self.epoch
                if c > 1:
                    pa = 0.3        # At the beginning of the process, the probability for shrinking encircling is small
                else:
                    pa = 0.7        # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey

                SP_leader = uniform(0, 1)
                if SP_leader >= 0.6:
                    new_pos = cos(2 * pi * normal(0, 1)) * abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) + g_best[self.ID_POS]
                else:
                    if uniform() < pa:
                        dist1 = uniform() * abs(2 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        new_pos = self._shrink_encircling_levy__(pop[i][self.ID_POS], epoch, dist1, c)
                    else:
                        rand_index = randint(0, self.pop_size)
                        rand_SL = pop_local[rand_index][self.ID_POS]
                        rand_SL = 2 * g_best[self.ID_POS] - rand_SL
                        new_pos = rand_SL - c * abs(uniform() * rand_SL - pop[i][self.ID_POS])

                new_pos = self._amend_solution_random_faster__(new_pos)
                new_fit = self._fitness_model__(new_pos)
                pop[i] = [new_pos, new_fit]                     # Move to new position anyway
                # Update current_pos, current_fit, compare and update past_pos and past_fit
                if new_fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = deepcopy([new_pos, new_fit])
                    if pop_local[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(pop_local[i])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
