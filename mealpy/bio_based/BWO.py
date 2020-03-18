#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:42, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, randint
from numpy import argmax, argmin, array
from copy import deepcopy
from mealpy.root import Root


class BaseBWO(Root):
    """
    The original version of: Black Widow Optimization (BWO)
        (Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103249
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100, pp=0.6, cr=0.44, pm=0.4):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_p = pp                   # procreating probability (crossover probability)   # default: 0.6
        self.c_r = cr                   # cannibalism rate (evolution theory)               # default: 0.44
        self.p_m = pm                   # mutation probability                              # default: 0.4

    ## Selection parents
    def _get_parents_kway_tournament_selection__(self, pop=None, k_way=10):
        list_id = choice(range(self.pop_size), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[0:2]

    def _get_parents_random_selection__(self, range=None, n=2):
        return randint(0, range, n)

    def _train__(self):
        # initialization
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        n_r = int(self.pop_size * self.p_p)     # Number of reproduction
        n_m = int(self.pop_size * self.p_m)     # Number of mutation children

        for epoch in range(self.epoch):
            ## Select the best nr solutions in pop and save them in pop1
            pop1 = deepcopy(pop[:n_r])
            pop2 = []
            ## Procreating and cannibalism
            for i in range(0, n_r):
                ### Selection based on k-way tournament
                #c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=10)

                ## Select parents
                c1, c2 = self._get_parents_random_selection__(n_r, 2)

                dad_id = argmax(array([pop1[c1][self.ID_FIT], pop1[c2][self.ID_FIT]]))
                mom_id = argmin(array([pop1[c1][self.ID_FIT], pop1[c2][self.ID_FIT]]))

                pop_new = []
                ## Mating. Eq. 1
                for j in range(0, int(self.problem_size/2)):
                    #alpha = uniform(0, 1, self.problem_size)
                    alpha = uniform()
                    y1 = alpha * pop1[dad_id][self.ID_POS] + (1.0 - alpha) * pop1[mom_id][self.ID_POS]
                    y2 = alpha * pop1[mom_id][self.ID_POS] + (1.0 - alpha) * pop1[dad_id][self.ID_POS]
                    fit1 = self._fitness_model__(y1)
                    fit2 = self._fitness_model__(y2)
                    pop_new.extend([deepcopy(pop1[mom_id]), [deepcopy(y1), fit1], [deepcopy(y2), fit2]])
                ## Based on cannibalism rate, destroy dad, destroy some children
                pop_new = sorted(pop_new, key=lambda item: item[self.ID_FIT])
                pop_new = pop_new[:int(self.c_r * len(pop_new))]
                pop2.extend(pop_new)

            ## Mutation
            for i in range(0, n_m):
                id_pos = randint(0, n_r)
                temp = pop1[id_pos][self.ID_POS]

                ## Mutation with 1 or 2 points seem not working well here.
                id_var1, id_var2 = randint(0, self.problem_size, 2)
                temp[id_var1], temp[id_var2] = temp[id_var2], temp[id_var1]
                ##temp[id_var1], temp[id_var2] = g_best[self.ID_POS][id_var1], g_best[self.ID_POS][id_var2]

                ## Apply mutation with multiple points with a probability 0.1 or 0.5
                # for j in range(0, self.problem_size):
                #     if uniform() < 0.5:
                #         temp[j] = g_best[self.ID_POS][j]

                fit = self._fitness_model__(temp)
                pop2.extend([ [deepcopy(temp), fit] ])

            pop = sorted(pop2, key=lambda item: item[self.ID_FIT])
            pop = pop[:self.pop_size]

            ## Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
