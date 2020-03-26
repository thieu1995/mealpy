#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseIWO(Root):
    """
    A novel numerical optimization algorithm inspired from weed colonization (IWO)
    Link:
        https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, seeds=(0, 5), exponent=2, sigma=(0.5, 0.001)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.seeds = seeds              # (Min, Max) Number of Seeds
        self.exponent = exponent        # Variance Reduction Exponent
        self.sigma = sigma              # (Initial, Final) Value of Standard Deviation

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_sorted, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        cost_best = g_best[self.ID_FIT]
        cost_worst = pop_sorted[self.ID_MAX_PROB][self.ID_FIT]
        for epoch in range(self.epoch):
            # Update Standard Deviation
            sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma[0] - self.sigma[1]) + self.sigma[1]
            # Reproduction
            pop_new = []
            for item in pop:
                ratio = (item[self.ID_FIT] - cost_worst) / (cost_best - cost_worst)
                S = int(self.seeds[0] + (self.seeds[1] - self.seeds[0]) * ratio)
                for j in range(S):
                    # Initialize Offspring and Generate Random Location
                    temp = item[self.ID_POS] + sigma * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    temp = self._amend_solution_faster__(temp)
                    fit = self._fitness_model__(temp)
                    pop_new.append([temp, fit])

            # Merge Populations
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Re-calculate best fit and worst fit
            cost_worst = pop[self.ID_MAX_PROB][self.ID_FIT]
            if cost_best > pop[self.ID_MIN_PROB][self.ID_FIT]:
                g_best = deepcopy(pop[self.ID_MIN_PROB])
                cost_best = g_best[self.ID_FIT]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
