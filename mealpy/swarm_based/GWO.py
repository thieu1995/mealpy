#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root

class BaseGWO(Root):
    """
    Standard version of Grey Wolf Optimizer (GWO)
        - In this algorithms: Prey means the best solution
        https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
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
                temp = (X1 + X2 + X3) / 3.0
                fit  = self._fitness_model__(temp)

                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            cur_best_1, cur_best_2, cur_best_3 = deepcopy(sorted_pop[:3])
            if cur_best_1[self.ID_FIT] < best_1[self.ID_FIT]:
                best_1 = deepcopy(cur_best_1)
            if cur_best_2[self.ID_FIT] < best_2[self.ID_FIT]:
                best_2 = deepcopy(cur_best_2)
            if cur_best_3[self.ID_FIT] < best_3[self.ID_FIT]:
                best_3 = deepcopy(cur_best_3)

            self.loss_train.append(best_1[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, best_1[self.ID_FIT]))

        return best_1[self.ID_POS], best_1[self.ID_FIT], self.loss_train
