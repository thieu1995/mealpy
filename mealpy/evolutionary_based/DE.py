#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:48, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import where
from numpy.random import uniform, choice
from copy import deepcopy
from mealpy.root import Root


class BaseDE(Root):
    """
        The original version of: Differential Evolution (DE)
    """
    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100, wf=0.8, cr=0.9):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr

    def _mutation__(self, p0, p1, p2, p3):
        ### Remove third loop here
        pos_new = deepcopy(p0)
        temp = p1 + self.weighting_factor * (p2 - p3)
        pos_new = where(uniform(0, 1, self.problem_size) < self.crossover_rate, temp, pos_new)
        return self.amend_position_faster(pos_new)

    def _create_children__(self, pop):
        pop_child = deepcopy(pop)
        for i in range(0, self.pop_size):
            # Choose 3 random element and different to i
            temp = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
            #create new child and append in children array
            child = self._mutation__(pop[i][self.ID_POS], pop[temp[0]][self.ID_POS], pop[temp[1]][self.ID_POS], pop[temp[2]][self.ID_POS])
            fit = self.get_fitness_position(child)
            pop_child[i] = [child, fit]
        return pop_child

    ### Survivor Selection
    def _greedy_selection__(self, pop_old=None, pop_new=None):
        pop = [pop_new[i] if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT] else pop_old[i] for i in range(self.pop_size)]
        return pop

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # create children
            pop_child = self._create_children__(pop)
            # create new pop by comparing fitness of corresponding each member in pop and children
            pop = self._greedy_selection__(pop, pop_child)

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train