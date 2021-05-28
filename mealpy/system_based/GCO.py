#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import max, min, array, where, ones, exp
from mealpy.root import Root
from copy import deepcopy


class BaseGCO(Root):
    """
    My modified version of: Germinal Center Optimization (GCO)
        (Germinal Center Optimization Algorithm)
    Link:
        https://www.atlantis-press.com/journals/ijcis/25905179/view
    Noted:
        + Using batch-size updating
        + Instead randomize choosing 3 solution, I use 2 random solution and global best solution
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 cr=0.7, f=1.25, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.cr = cr                # Same as DE algorithm  # default: 0.7
        self.f = f                  # Same as DE algorithm  # default: 1.25

    def train(self):
        list_cell_counter = ones(self.pop_size)             # CEll Counter
        list_life_signal = 70 * ones(self.pop_size)         # 70% to duplicate, and 30% to die  # LIfe-Signal
        pop = [self.create_solution() for _ in range(self.pop_size)]  # B-cells population
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Dark-zone process
            for i in range(0, self.pop_size):
                if uniform(0, 100) < list_life_signal[i]:
                    list_cell_counter[i] += 1
                else:
                    list_cell_counter[i] = 1

                # Mutate process
                r1, r2 = choice(range(0, self.pop_size), 2, replace=False)
                pos_new = g_best[self.ID_POS] + self.f * (pop[r2][self.ID_POS] - pop[r1][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < self.cr, pos_new, pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                    list_cell_counter[i] += 10

                ## Update based on batch-size training
                if self.batch_idea:
                    if (i+1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            ## Light-zone process
            for i in range(0, self.pop_size):
                list_cell_counter[i] = 10
                fit_list = array([item[self.ID_FIT] for item in pop])
                fit_max = max(fit_list)
                fit_min = min(fit_list)
                list_cell_counter[i] += 10 * (pop[i][self.ID_FIT] - fit_max) / (fit_min - fit_max + self.EPSILON)

            ## Update the global best
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalGCO(Root):
    """
    Original version of: Germinal Center Optimization (GCO)
        (Germinal Center Optimization Algorithm)
    Link:
        DOI: https://doi.org/10.2991/ijcis.2018.25905179
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 cr=0.7, f=1.25, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.cr = cr                # Same as DE algorithm  # default: 0.7
        self.f = f                  # Same as DE algorithm  # default: 1.25

    def train(self):
        list_cell_counter = ones(self.pop_size)
        list_life_signal = 70 * ones(self.pop_size)         # 70% to duplicate, and 30% to die
        pop = [self.create_solution() for _ in range(self.pop_size)]  # B-cells population
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Dark-zone process
            for i in range(0, self.pop_size):
                if uniform(0, 100) < list_life_signal[i]:
                    list_cell_counter[i] += 1
                else:
                    list_cell_counter[i] = 1

                # Mutate process
                r1, r2, r3 = choice(range(0, self.pop_size), 3, replace=False)
                pos_new = pop[r1][self.ID_POS] + self.f * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < self.cr, pos_new, pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                    list_cell_counter[i] += 10
                    if pop[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(pop[i])

            ## Light-zone process
            for i in range(0, self.pop_size):
                list_cell_counter[i] = 10
                fit_list = array([item[self.ID_FIT] for item in pop])
                fit_max = max(fit_list)
                fit_min = min(fit_list)
                list_cell_counter[i] += 10 * (pop[i][self.ID_FIT] - fit_max) / (fit_min - fit_max)

            ## Update the global best
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
