#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:58, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from copy import deepcopy
from mealpy.root import Root


class BaseGSKA(Root):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)
        (Gaining‑sharing Knowledge-Based Algorithm For Solving Optimization Problems: A Novel Nature‑inspired Algorithm)
    Link:
        DOI: https://doi.org/10.1007/s13042-019-01053-x
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100,
                 p=0.1, kf=0.5, kr=0.9, k=10):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size  # n: pop_size, m: clusters
        self.p = p                  # percent of the best   0.1%, 0.8%, 0.1%
        self.kf = kf                # knowledge factor that controls the total amount of gained and shared knowledge added from others to the current
                                    # individuals during generations
        self.kr = kr                # knowledge ratio
        self.k = k                  # KNOWLEDGE rate

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            D = int(self.problem_size * (1 - (epoch+1)/self.epoch) ** self.k)
            for i in range(1, self.pop_size-1):

                pos_new = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    if j < D:                       # junior gaining and sharing
                        if uniform() <= self.kr:
                            rand_idx = choice(list(set(range(0, self.pop_size)) - {i - 1, i, i + 1}))
                            if pop[i][self.ID_FIT] > pop[rand_idx][self.ID_FIT]:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                          (pop[i-1][self.ID_POS][j] - pop[i+1][self.ID_POS][j] + pop[rand_idx][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                          (pop[i - 1][self.ID_POS][j] - pop[i + 1][self.ID_POS][j] + pop[i][self.ID_POS][j] - pop[rand_idx][self.ID_POS][j])
                    else:                           # senior gaining and sharing
                        if uniform() <= self.kr:
                            id1 = int(self.p * self.pop_size)
                            id2 = id1 + int(self.pop_size - 2 * 100 * self.p)
                            rand_best = choice(list(set(range(0, id1)) - {i}))
                            rand_worst = choice(list(set(range(id2, self.pop_size)) - {i}))
                            rand_mid = choice(list(set(range(id1, id2)) - {i}))
                            if pop[i][self.ID_FIT] > pop[rand_mid][self.ID_FIT]:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                    (pop[rand_best][self.ID_POS][j] - pop[rand_worst][self.ID_POS][j] + pop[rand_mid][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                    (pop[rand_best][self.ID_POS][j] - pop[rand_worst][self.ID_POS][j] + pop[i][self.ID_POS][j] - pop[rand_mid][self.ID_POS][j])
                pos_new = self._amend_solution_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Sort the population and update the global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

