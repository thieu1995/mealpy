#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:48, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import randint, uniform, choice
from copy import deepcopy
from mealpy.root import RootAlgo


class BaseDE(RootAlgo):

    def __init__(self, root_algo_paras=None, de_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = de_paras["epoch"]
        self.pop_size = de_paras["pop_size"]
        self.weighting_factor = de_paras["wf"]
        self.crossover_rate = de_paras["cr"]

    def _mutation__(self, p0, p1, p2, p3):
        # Choose a cut point which differs 0 and chromosome-1 (first and last element)
        cut_point = randint(1, self.problem_size - 1)
        sample = deepcopy(p0)
        for k in range(self.problem_size):
            if k == cut_point or uniform() < self.crossover_rate:
                sample[k] = p1[k] + self.weighting_factor * ( p2[k] - p3[k])
        return self._amend_solution_faster__(sample)

    def _create_children__(self, pop):
        pop_child = deepcopy(pop)
        for k in range(0, self.pop_size):
            # Choose 3 random element and different to i
            temp = choice(list(set(range(0, self.pop_size)) - {k}), 3, replace=False)
            #create new child and append in children array
            child = self._mutation__(pop[k][self.ID_POS], pop[temp[0]][self.ID_POS], pop[temp[1]][self.ID_POS], pop[temp[2]][self.ID_POS])
            fit = self._fitness_model__(child)
            pop_child[k] = [child, fit]
        return pop_child

    ### Survivor Selection
    def _greedy_selection__(self, pop_old=None, pop_new=None):
        pop = [pop_new[i] if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT] else pop_old[i] for i in range(self.pop_size)]
        return pop

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # create children
            pop_child = self._create_children__(pop)
            # create new pop by comparing fitness of corresponding each member in pop and children
            pop = self._greedy_selection__(pop, pop_child)

            # update global best solution
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch : {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train