#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:42, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import argmax, argmin, array, where
from copy import deepcopy
from mealpy.optimizer import Root


class BaseBWO(Root):
    """
    My version of: Black Widow Optimization (BWO)
        (Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103249
    Notes:
        + Using k-way tournamemt selection to select parent instead of randomize
        + Repeat cross-over population_size / 2 instead of n_var/2
        + Mutation 50% of position instead of swap only 2 variable in single position
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 pp=0.6, cr=0.44, pm=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_p = pp       # procreating probability (crossover probability)   # default: 0.6
        self.c_r = cr       # cannibalism rate (evolution theory)               # default: 0.44
        self.p_m = pm       # mutation probability                              # default: 0.4

    def train(self):
        # initialization
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        n_r = int(self.pop_size * self.p_p)  # Number of reproduction
        n_m = int(self.pop_size * self.p_m)  # Number of mutation children

        for epoch in range(self.epoch):
            ## Select the best nr solutions in pop and save them in pop1
            pop1 = deepcopy(pop[:n_r])
            pop2 = []
            ## Procreating and cannibalism
            for i in range(0, n_r):
                ### Selection parents based on k-way tournament
                dad, mom = self.get_parent_kway_tournament_selection(pop1, 0.2)
                pop_new = []
                ## Mating. Eq. 1
                for j in range(0, int(self.pop_size / 2)):
                    alpha = uniform(0, 1, self.problem_size)
                    y1 = alpha * dad[self.ID_POS] + (1.0 - alpha) * mom[self.ID_POS]
                    y2 = alpha * mom[self.ID_POS] + (1.0 - alpha) * dad[self.ID_POS]
                    fit1 = self.get_fitness_position(y1)
                    fit2 = self.get_fitness_position(y2)
                    pop_new.extend([deepcopy(mom), [deepcopy(y1), fit1], [deepcopy(y2), fit2]])
                ## Based on cannibalism rate, destroy dad, destroy some children
                pop_new = sorted(pop_new, key=lambda item: item[self.ID_FIT])
                pop_new = pop_new[:int(self.c_r * len(pop_new))]
                pop2.extend(pop_new)

            ## Mutation
            for i in range(0, n_m):
                id_pos = randint(0, n_r)
                temp = pop1[id_pos][self.ID_POS]
                ## Mutation 50% of position
                pos_new = where(uniform(0, 1, self.problem_size) < self.p_m, g_best[self.ID_POS], temp)
                fit_new = self.get_fitness_position(pos_new)
                pop2.extend([[deepcopy(pos_new), fit_new]])

            ## Update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop2, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalBWO(Root):
    """
    The original version of: Black Widow Optimization (BWO)
        (Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103249
        + This algorithm is just a variant flow of genetic algorithm
        + The performance even worst than GA because the random choice in selecting parents and mutation by swap position
        + The worst part is repeat cross-over n_var/2 times -> For nothing, more offspring not guarantee broader search space
        ==> It make the algorithm is much more slower even running in parallel as author claim.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 pp=0.6, cr=0.44, pm=0.4, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_p = pp                   # procreating probability (crossover probability)   # default: 0.6
        self.c_r = cr                   # cannibalism rate (evolution theory)               # default: 0.44
        self.p_m = pm                   # mutation probability                              # default: 0.4

    def train(self):
        # initialization
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        n_r = int(self.pop_size * self.p_p)     # Number of reproduction
        n_m = int(self.pop_size * self.p_m)     # Number of mutation children

        for epoch in range(self.epoch):
            ## Select the best nr solutions in pop and save them in pop1
            pop1 = deepcopy(pop[:n_r])
            pop2 = []
            ## Procreating and cannibalism
            for i in range(0, n_r):
                ## Select parents
                c1, c2 = randint(0, n_r, 2)
                dad_id = argmax(array([pop1[c1][self.ID_FIT], pop1[c2][self.ID_FIT]]))
                mom_id = argmin(array([pop1[c1][self.ID_FIT], pop1[c2][self.ID_FIT]]))

                pop_new = []
                ## Mating. Eq. 1
                for j in range(0, int(self.problem_size/2)):
                    alpha = uniform()
                    y1 = alpha * pop1[dad_id][self.ID_POS] + (1.0 - alpha) * pop1[mom_id][self.ID_POS]
                    y2 = alpha * pop1[mom_id][self.ID_POS] + (1.0 - alpha) * pop1[dad_id][self.ID_POS]
                    fit1 = self.get_fitness_position(y1)
                    fit2 = self.get_fitness_position(y2)
                    pop_new.extend([deepcopy(pop1[mom_id]), [deepcopy(y1), fit1], [deepcopy(y2), fit2]])
                ## Based on cannibalism rate, destroy dad, destroy some children
                pop_new = sorted(pop_new, key=lambda item: item[self.ID_FIT])
                pop_new = pop_new[:int(self.c_r * len(pop_new))]
                pop2.extend(pop_new)

            ## Mutation
            for i in range(0, n_m):
                id_pos = randint(0, n_r)
                pos_new = pop1[id_pos][self.ID_POS]

                ## Mutation with 1 or 2 points seem not working well here.
                id_var1, id_var2 = randint(0, self.problem_size, 2)
                pos_new[id_var1], pos_new[id_var2] = pos_new[id_var2], pos_new[id_var1]
                fit = self.get_fitness_position(pos_new)
                pop2.extend([[deepcopy(pos_new), fit]])

            pop = sorted(pop2, key=lambda item: item[self.ID_FIT])
            pop = pop[:self.pop_size]

            ## Update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
