#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp
from numpy.random import uniform, normal
from copy import deepcopy
from mealpy.optimizer import Root


class BaseEPO(Root):
    """
    My modified version of: Emperor penguin optimizer (EPO)
    Notes:
        + First: I changed the Eq. T_s and no need T and random R.
        + Second: Updated the old position if fitness value better or kept the old position if otherwise
        + Third: Remove third loop for faster
        + Fourth: Batch size idea
        + Fifth: Add normal() component and change minus sign to plus
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        M = 2
        for epoch in range(self.epoch):
            ## First: Changed Eq. T_s
            T_s = 1.0 - 1.0 / (self.epoch - epoch)
            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])

                ## Remove third loop
                P_grid = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                A = M * (T_s + P_grid) * uniform(0, 1, self.problem_size) - T_s
                C = uniform(0, 1, self.problem_size)
                f = uniform(2, 3, self.problem_size)
                l = uniform(1.5, 2, self.problem_size)

                S_A = abs(f * exp(-epoch / l) - exp(-epoch))
                D_ep = abs(S_A * g_best[self.ID_POS] - C * pop[i][self.ID_POS])
                pos_new = g_best[self.ID_POS] + normal() * A * D_ep                 # Added normal() component and change minus sign to plus

                pos_new = self.amend_position_faster(pos_new)
                ## Second: Updated the old position if fitness value better or kept the old position if otherwise
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

                ## Batch size idea
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


class OriginalEPO(BaseEPO):
    """
    The original version of: Emperor penguin optimizer (EPO)
        (Emperor penguin optimizer: A bio-inspired algorithm for engineering problems)
    Fake:
        + This version is basically wrong at the Eq. T and Eq. T_s
        + (First: The value of T always = 1, because R = rand() always < 1.)
        + (Second: T_s = T - max_epoch / (x - max_epoch) = T + max_epoch / (max_epoch - x)   ===> T <= T_s <= max_epoch
        + ===> What?, T_s should be in the range (0, 1) )
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseEPO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        M = 2
        for epoch in range(self.epoch):
            R = uniform()
            T = 0 if R > 1 else 1
            T_s = T - self.epoch / (epoch - self.epoch)
            for i in range(self.pop_size):
                for j in range(self.problem_size):

                    P_grid = abs( g_best[self.ID_POS][j] - pop[i][self.ID_POS][j] )
                    A = M * (T_s + P_grid) * uniform() - T_s
                    C = uniform()

                    f = uniform(2, 3)
                    l = uniform(1.5, 2)
                    S_A = abs(f * exp(-epoch / l) - exp(-epoch))
                    D_ep = abs( S_A * g_best[self.ID_POS][j] - C * pop[i][self.ID_POS][j] )
                    pop[i][self.ID_POS][j] = g_best[self.ID_POS][j] - A * D_ep
                pop[i][self.ID_POS] = self.amend_position_faster(pop[i][self.ID_POS])

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
