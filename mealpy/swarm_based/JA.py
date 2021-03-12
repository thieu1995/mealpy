#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:30, 16/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, rand
from numpy import abs
from mealpy.root import Root


class BaseJA(Root):
    """
        My original version of: Jaya Algorithm (JA)
            (A simple and new optimization algorithm for solving constrained and unconstrained optimization problems)
        Link:
            + Remove all third loop in algorithm
            + Change the second random variable r2 to Gaussian instead of uniform
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                po = pop[i][self.ID_POS]
                pos_new = po + uniform() * (g_best[self.ID_POS] - abs(po)) + normal() * (g_worst[self.ID_POS] - abs(po))
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalJA(BaseJA):
    """
        The original version of: Jaya Algorithm (JA)
            (A simple and new optimization algorithm for solving constrained and unconstrained optimization problems)
        Link:
            http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseJA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                po = pop[i][self.ID_POS]
                for j in range(0, self.problem_size):
                    pos_new = po[j] + uniform()*(g_best[self.ID_POS][j] - abs(po[j])) - uniform()*(g_worst[self.ID_POS][j] - abs(po[j]))
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LJA(BaseJA):
    """
        The original version of: Levy-flight Jaya Algorithm (LJA)
            (An improved Jaya optimization algorithm with Levy flight)
        Link:
            + https://doi.org/10.1016/j.eswa.2020.113902
        Note:
            + This version I still remove all third loop in algorithm
            + The beta value of Levy-flight equal to 1.8 as the best value in the paper.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseJA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                L1 = self.step_size_by_levy_flight(multiplier=1.0, beta=1.0, case=-1)
                L2 = self.step_size_by_levy_flight(multiplier=1.0, beta=1.0, case=-1)
                po = pop[i][self.ID_POS]
                pos_new = po + abs(L1) * (g_best[self.ID_POS] - abs(po)) - abs(L2) * (g_worst[self.ID_POS] - abs(po))
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

