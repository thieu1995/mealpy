#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import ceil
from copy import deepcopy
from mealpy.root import Root


class BaseIWO(Root):
    """
    My version of: weed colonization (IWO)
        A novel numerical optimization algorithm inspired from weed colonization
    Noted:
        https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 seeds=(2, 10), exponent=2, sigma=(0.5, 0.001), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.seeds = seeds          # (Min, Max) Number of Seeds
        self.exponent = exponent    # Variance Reduction Exponent
        self.sigma = sigma          # (Initial, Final) Value of Standard Deviation

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        fit_best = g_best[self.ID_FIT]
        fit_worst = pop[self.ID_MAX_PROB][self.ID_FIT]
        for epoch in range(self.epoch):
            # Update Standard Deviation
            sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma[0] - self.sigma[1]) + self.sigma[1]
            # Reproduction
            pop_new = []
            for item in pop:
                ratio = (item[self.ID_FIT] - fit_worst) / (fit_best - fit_worst + self.EPSILON)
                s = int(ceil(self.seeds[0] + (self.seeds[1] - self.seeds[0]) * ratio))
                for j in range(s):
                    # Initialize Offspring and Generate Random Location
                    pos_new = item[self.ID_POS] + sigma * uniform(self.lb, self.ub)
                    pos_new = self.amend_position_faster(pos_new)
                    fit = self.get_fitness_position(pos_new)
                    pop_new.append([pos_new, fit])

            # Re-calculate best train and worst train
            pop = pop + pop_new
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            fit_worst = pop[self.ID_MAX_PROB][self.ID_FIT]
            fit_best = pop[self.ID_MIN_PROB][self.ID_FIT]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalIWO(Root):
    """
    Original version of: weed colonization (IWO)
        A novel numerical optimization algorithm inspired from weed colonization
    Link:
        https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 seeds=(2, 10), exponent=2, sigma=(0.5, 0.001), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.seeds = seeds              # (Min, Max) Number of Seeds
        self.exponent = exponent        # Variance Reduction Exponent
        self.sigma = sigma              # (Initial, Final) Value of Standard Deviation

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_sorted, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        cost_best = g_best[self.ID_FIT]
        cost_worst = pop_sorted[self.ID_MAX_PROB][self.ID_FIT]
        for epoch in range(self.epoch):
            # Update Standard Deviation
            sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma[0] - self.sigma[1]) + self.sigma[1]
            # Reproduction
            pop_new = []
            for item in pop:
                ratio = (item[self.ID_FIT] - cost_worst) / (cost_best - cost_worst)
                S = int(ceil(self.seeds[0] + (self.seeds[1] - self.seeds[0]) * ratio))
                for j in range(S):
                    # Initialize Offspring and Generate Random Location
                    pos_new = item[self.ID_POS] + sigma * uniform(self.lb, self.ub)
                    pos_new = self.amend_position_faster(pos_new)
                    fit = self.get_fitness_position(pos_new)
                    pop_new.append([pos_new, fit])

            # Merge Populations
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Re-calculate best train and worst train
            cost_worst = pop[self.ID_MAX_PROB][self.ID_FIT]
            if cost_best > pop[self.ID_MIN_PROB][self.ID_FIT]:
                g_best = deepcopy(pop[self.ID_MIN_PROB])
                cost_best = g_best[self.ID_FIT]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
