#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseEPO(Root):
    """
    This is my variant version of Emperor penguin optimizer, and its work.
        + First: I changed the Eq. T_s and no need T and random R.
        + Second: Updated the old solution if fitness value better or kept the old solution if otherwise
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        M = 2
        for epoch in range(self.epoch):
            ## First: Changed Eq. T_s
            T_s = 1.0 - 1.0 / (self.epoch - epoch)
            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                for j in range(self.problem_size):

                    P_grid = abs( g_best[self.ID_POS][j] - pop[i][self.ID_POS][j] )
                    A = M * (T_s + P_grid) * uniform() - T_s
                    C = uniform()

                    f = uniform(2, 3)
                    l = uniform(1.5, 2)
                    S_A = abs(f * exp(-epoch / l) - exp(-epoch))
                    D_ep = abs( S_A * g_best[self.ID_POS][j] - C * pop[i][self.ID_POS][j] )
                    temp[j] = g_best[self.ID_POS][j] - A * D_ep
                temp = self._amend_solution_faster__(temp)
                ## Second: Updated the old solution if fitness value better or kept the old solution if otherwise
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalEPO(Root):
    """
    Paper: Emperor penguin optimizer: A bio-inspired algorithm for engineering problems
    This version is basically wrong at the Eq. T and Eq. T_s
    (First: The value of T always = 1, because R = rand() always < 1.)
    (Second: T_s = T - max_epoch / (x - max_epoch) = T + max_epoch / (max_epoch - x)   ===> T <= T_s <= max_epoch
        ===> What?, T_s should be in the range (0, 1) )
    """

    def __init__(self, root_paras=None, epoch = 750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

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
                pop[i][self.ID_POS] = self._amend_solution_faster__(pop[i][self.ID_POS])

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
