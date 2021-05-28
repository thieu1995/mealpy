#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:34, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
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

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, p=0.8, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p = p      # Switch probability

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                if uniform() < self.p:
                    levy = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                    pos_new = pop[i][self.ID_POS] + levy * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    id1, id2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    pos_new = pop[i][self.ID_POS] + uniform() * (pop[id1][self.ID_POS] - pop[id2][self.ID_POS])
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                # batch size idea to update the global best
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
