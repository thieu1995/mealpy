#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, sin, pi, abs, sqrt, log
from numpy.random import uniform
from mealpy.root import Root


class BaseMRFO(Root):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)
        (Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103300
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, S=2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.S = S                   # somersault factor that decides the somersault range of manta rays

    def _next_move__(self, pop=None, g_best=None, epoch=None, i=None):
        if uniform() < 0.5:  # Cyclone foraging (Eq. 5, 6, 7)
            r1 = uniform()
            beta = 2 * exp(r1 * (self.epoch - epoch) / self.epoch) * sin(2 * pi * r1)

            if (epoch + 1) / self.epoch < uniform():
                x_rand = uniform(self.lb, self.ub)
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
            alpha = 2 * r * sqrt(abs(log(r)))
            if i == 0:
                x_t1 = pop[i][self.ID_POS] + r * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + alpha * (
                            g_best[self.ID_POS] - pop[i][self.ID_POS])
            else:
                x_t1 = pop[i][self.ID_POS] + r * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS]) + alpha * (
                            g_best[self.ID_POS] - pop[i][self.ID_POS])
        return x_t1

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                x_t1 = self._next_move__(pop, g_best, epoch, i)
                fit = self.get_fitness_position(x_t1)
                pop[i] = [x_t1, fit]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Somersault foraging   (Eq. 8)
            for i in range(0, self.pop_size):
                x_t1 = pop[i][self.ID_POS] + self.S * (uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                fit = self.get_fitness_position(x_t1)
                pop[i] = [x_t1, fit]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyMRFO(BaseMRFO):
    """
        My modified version of: Manta Ray Foraging Optimization (MRFO) based on Levy_flight
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, S=2, **kwargs):
        BaseMRFO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, S, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                x_t1 = self._next_move__(pop, g_best, epoch, i)
                x_t1 = self.amend_position_faster(x_t1)
                fit = self.get_fitness_position(x_t1)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Somersault foraging   (Eq. 8)
            for i in range(0, self.pop_size):
                if uniform() < 0.5:
                    x_t1 = pop[i][self.ID_POS] + self.S * (uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                else:
                    x_t1 = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                x_t1 = self.amend_position_faster(x_t1)
                fit = self.get_fitness_position(x_t1)
                pop[i] = [x_t1, fit]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
