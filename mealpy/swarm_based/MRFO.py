#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from numpy.random import uniform
from mealpy.root import Root


class BaseMRFO(Root):
    """
    The is my version of: Manta Ray Foraging Optimization (MRFO)
        (Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications)
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, S=2):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.S = S                   # somersault factor that decides the somersault range of manta rays

    def _next_move__(self, pop=None, g_best=None, epoch=None, i=None):
        if uniform() < 0.5:  # Cyclone foraging (Eq. 5, 6, 7)
            r1 = uniform()
            beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

            if (epoch + 1) / self.epoch < uniform():
                x_rand = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)  # vector
                if i == 0:
                    x_t1 = x_rand + uniform() * (x_rand - pop[i][self.ID_POS]) + beta * (x_rand - pop[i][self.ID_POS])
                else:
                    x_t1 = x_rand + uniform() * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS]) + beta * (
                                x_rand - pop[i][self.ID_POS])
            else:
                if i == 0:
                    x_t1 = g_best[self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + beta * (
                                g_best[self.ID_POS] - pop[i][self.ID_POS])
                else:
                    x_t1 = g_best[self.ID_POS] + uniform() * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS]) + beta * (
                                g_best[self.ID_POS] - pop[i][self.ID_POS])

        else:  # Chain foraging (Eq. 1,2)
            r = uniform()
            alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
            if i == 0:
                x_t1 = pop[i][self.ID_POS] + r * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + alpha * (
                            g_best[self.ID_POS] - pop[i][self.ID_POS])
            else:
                x_t1 = pop[i][self.ID_POS] + r * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS]) + alpha * (
                            g_best[self.ID_POS] - pop[i][self.ID_POS])
        return x_t1


    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                x_t1 = self._next_move__(pop, g_best, epoch, i)
                x_t1 = self._amend_solution_faster__(x_t1)
                fit = self._fitness_model__(x_t1)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit]
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Somersault foraging   (Eq. 8)
            for i in range(0, self.pop_size):
                x_t1 = pop[i][self.ID_POS] + self.S * (uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                x_t1 = self._amend_solution_faster__(x_t1)
                fit = self._fitness_model__(x_t1)
                pop[i] = [x_t1, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalMRFO(BaseMRFO):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)
        (Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications)
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, S=2):
        BaseMRFO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, S)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                x_t1 = self._next_move__(pop, g_best, epoch, i)
                fit = self._fitness_model__(x_t1)
                pop[i] = [x_t1, fit]
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Somersault foraging   (Eq. 8)
            for i in range(0, self.pop_size):
                x_t1 = pop[i][self.ID_POS] + self.S * (uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                fit = self._fitness_model__(x_t1)
                pop[i] = [x_t1, fit]
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyMRFO(BaseMRFO):
    """
        The is modified version of: Manta Ray Foraging Optimization (MRFO) based on Levy_flight
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, S=2):
        BaseMRFO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, S)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                x_t1 = self._next_move__(pop, g_best, epoch, i)
                x_t1 = self._amend_solution_faster__(x_t1)
                fit = self._fitness_model__(x_t1)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit]
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Somersault foraging   (Eq. 8)
            for i in range(0, self.pop_size):
                if uniform() < 0.5:
                    x_t1 = pop[i][self.ID_POS] + self.S * (uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                else:
                    x_t1 = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                x_t1 = self._amend_solution_faster__(x_t1)
                fit = self._fitness_model__(x_t1)
                pop[i] = [x_t1, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
