#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:47, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import pi, exp, cos, sin
from numpy.random import uniform
from mealpy.optimizer import Root


class BaseSOA(Root):
    """
        My modified version of: Sandpiper Optimization Algorithm (SOA)
        Notes:
            + I changed some equations and the flow of algorithm
            + Remember this paper and algorithm is dummy
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        C_f = 1.0

        # Epoch loop
        for epoch in range(self.epoch):

            ## Each individual loop
            for i in range(self.pop_size):

                ### Based on Eq.5, 6, 7, 8, 9
                C_sp = (C_f - epoch * (C_f/self.epoch)) * pop[i][self.ID_POS]
                M_sp = uniform() * ( g_best[self.ID_POS] - pop[i][self.ID_POS] )
                D_sp = C_sp + M_sp

                ### Based on Eq. 10, 11, 12, 13, 14
                r = exp(uniform(0, 2*pi))
                temp = r * (sin(uniform(0, 2*pi)) + cos(uniform(0, 2*pi)) + uniform(0, 2*pi))
                P_sp = (D_sp * temp) * g_best[self.ID_POS]

                P_sp = self.amend_position_faster(P_sp)
                fit = self.get_fitness_position(P_sp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [P_sp, fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [P_sp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSOA(BaseSOA):
    """
        The original version of: Sandpiper Optimization Algorithm (SOA)
            (Sandpiper optimization algorithm: a novel approach for solving real-life engineering problems or
            A bio-inspired based optimization algorithm for industrial engineering problems.)
        Notes:
            + This algorithm is trash, unethical when summit a paper to 2 journals.
            + Can't even update its position.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseSOA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        C_f = 2

        # Epoch loop
        for epoch in range(self.epoch):

            ## Each individual loop
            for i in range(self.pop_size):

                ### Based on Eq.5, 6, 7, 8, 9
                C_sp = (C_f - epoch * (C_f / self.epoch)) * pop[i][self.ID_POS]
                M_sp = 0.5 * uniform() * ( g_best[self.ID_POS] - pop[i][self.ID_POS] )
                D_sp = C_sp + M_sp

                ### Based on Eq. 10, 11, 12, 13, 14
                r = exp(uniform(0, 2*pi))
                temp = r * (sin(uniform(0, 2*pi)) + cos(uniform(0, 2*pi)) + uniform(0, 2*pi))
                P_sp = (D_sp * temp) * g_best[self.ID_POS]
                fit = self.get_fitness_position(P_sp)
                pop[i] = [P_sp, fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [P_sp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

