#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:47, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, normal, random
from numpy import abs, sign, cos, pi, sin, sqrt, power
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseSLO(Root):
    """
        This version developed by one on my student: Sea Lion Optimization Algorithm
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
        Notes:
            + The original paper is dummy, tons of unclear equations and parameters
            + You can check my question on the ResearchGate link, the authors seem to be scare so they didn't reply.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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

                new_pos = self.amend_position_random_faster(new_pos)
                fit = self.get_fitness_position(new_pos)
                pop[i] = [new_pos, fit]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedSLO(BaseSLO):
    """
        The improved version of: Sea Lion Optimization (ISLO)
            (Sea Lion Optimization Algorithm)
        Noted:
            + Using the idea of shrink encircling combine with levy flight techniques
            + Also using the idea of local best in PSO
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseSLO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def _shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * sin(pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2))
        xich_ma_1 = power(up / down, 1 / beta)
        xich_ma_2 = 1
        a = normal(0, xich_ma_1, 1)
        b = normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (power(abs(b), 1 / beta)) * dist * c
        D = uniform(self.lb, self.ub)
        levy = LB * D
        return (current_pos - sqrt(epoch + 1) * sign(random(1) - 0.5)) * levy

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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

                new_pos = self.amend_position_random_faster(new_pos)
                new_fit = self.get_fitness_position(new_pos)
                pop[i] = [new_pos, new_fit]                     # Move to new position anyway
                # Update current_pos, current_fit, compare and update past_pos and past_fit
                if new_fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = deepcopy([new_pos, new_fit])

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
