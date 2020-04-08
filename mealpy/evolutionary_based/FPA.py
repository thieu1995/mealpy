#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:34, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from mealpy.root import Root


class BaseFPA(Root):
    """
        The original version of: Flower Pollination Algorithm (FPA)
            (Flower Pollination Algorithm for Global Optimization)
    Link:
        https://doi.org/10.1007/978-3-642-32894-7_27
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, p=0.8):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p = p

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                if uniform() < self.p:
                    levy = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                    pos_new = pop[i][self.ID_POS] + levy * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    id1, id2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    pos_new = pop[i][self.ID_POS] + uniform() * (pop[id1][self.ID_POS] - pop[id2][self.ID_POS])
                pos_new = self._amend_solution_random_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
