#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs
from numpy.random import uniform, standard_cauchy
from copy import deepcopy
from mealpy.root import Root


class BaseGWO(Root):
    """
        The original version of: Grey Wolf Optimizer (GWO)
        Notes:
            In this algorithms: Prey means the best position
            https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        best_1, best_2, best_3 = deepcopy(sorted_pop[:3])

        for epoch in range(self.epoch):
            a = 2 - 2 * epoch / (self.epoch - 1)            # linearly decreased from 2 to 0

            for i in range(self.pop_size):

                A1, A2, A3 = a * (2 * uniform() - 1), a * (2 * uniform() - 1), a * (2 * uniform() - 1)
                C1, C2, C3 = 2 * uniform(), 2*uniform(), 2*uniform()

                X1 = best_1[self.ID_POS] - A1 * abs(C1 * best_1[self.ID_POS] - pop[i][self.ID_POS])
                X2 = best_2[self.ID_POS] - A2 * abs(C2 * best_2[self.ID_POS] - pop[i][self.ID_POS])
                X3 = best_3[self.ID_POS] - A3 * abs(C3 * best_3[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = (X1 + X2 + X3) / 3.0
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

            sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            cur_best_1, cur_best_2, cur_best_3 = deepcopy(sorted_pop[:3])
            if cur_best_1[self.ID_FIT] < best_1[self.ID_FIT]:
                best_1 = deepcopy(cur_best_1)
            if cur_best_2[self.ID_FIT] < best_2[self.ID_FIT]:
                best_2 = deepcopy(cur_best_2)
            if cur_best_3[self.ID_FIT] < best_3[self.ID_FIT]:
                best_3 = deepcopy(cur_best_3)

            self.loss_train.append(best_1[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, best_1[self.ID_FIT]))
        self.solution = best_1
        return best_1[self.ID_POS], best_1[self.ID_FIT], self.loss_train


class RW_GWO(Root):
    """
        The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)
        - A novel Random Walk Grey Wolf Optimizer (What a trash paper, the random walk GWO always perform worst than original GWO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        leaders = deepcopy(sorted_pop[:3])
        g_best = deepcopy(sorted_pop[0])

        for epoch in range(self.epoch):
            b = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0, Eq. 5
            a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
            ## Random walk here
            for i in range(0, len(leaders)):
                pos_new = leaders[i][self.ID_POS] + a * standard_cauchy(self.problem_size)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < leaders[i][self.ID_FIT]:
                    leaders[i] = [pos_new, fit_new]
            ## Update other wolfs
            for i in range(self.pop_size):
                miu1, miu2, miu3 = b * (2 * uniform() - 1), b * (2 * uniform() - 1), b * (2 * uniform() - 1)    # Eq. 3
                c1, c2, c3 = 2 * uniform(), 2 * uniform(), 2 * uniform()                                        # Eq. 4
                X1 = leaders[0][self.ID_POS] - miu1 * abs(c1 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                X2 = leaders[1][self.ID_POS] - miu2 * abs(c2 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                X3 = leaders[2][self.ID_POS] - miu3 * abs(c3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = (X1 + X2 + X3) / 3.0
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
            sorted_pop = sorted(pop + leaders, key=lambda temp: temp[self.ID_FIT])
            pop = deepcopy(sorted_pop[:self.pop_size])
            leaders = deepcopy(sorted_pop[:3])
            if sorted_pop[self.ID_MIN_PROB][self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(sorted_pop[self.ID_MIN_PROB])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
