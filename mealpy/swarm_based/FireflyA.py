#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:13, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import sqrt, exp, matmul
from numpy.random import uniform
from scipy.linalg import norm
from copy import deepcopy
from mealpy.root import Root


class BaseFireflyA(Root):
    """
        This version is converted from Matlab code of Hoang Nguyen (nguyenhoang.mdc@gmail.com):
        Link:
            DOI:
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 gamma=1, beta_base=1, alpha=0.2, alpha_damp=0.99, delta=0.05, m=2, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.gamma = gamma              # Light Absorption Coefficient
        self.beta_base = beta_base      # Attraction Coefficient Base Value
        self.alpha = alpha              # Mutation Coefficient
        self.alpha_damp = alpha_damp    # Mutation Coefficient Damp Rate
        self.delta = delta              # Mutation Step Size
        self.m = m                      # Exponent

    def train(self):

        # Initialization population
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        # Initial Value of Mutation Coefficient
        alpha = self.alpha

        for epoch in range(self.epoch):
            # Maximum Distance
            dmax = sqrt(self.problem_size)
            # Decision Vector Size

            pop_new = deepcopy(pop)
            for i in range(0, self.pop_size-1):

                for j in range(i+1, self.pop_size):
                    # Move Towards Better Solutions

                    if pop[j][self.ID_FIT] < pop[i][self.ID_FIT]:
                        # Calculate Radius and Attraction Level
                        rij = norm(pop[i][self.ID_POS] - pop[j][self.ID_POS]) / dmax
                        beta = self.beta_base * exp(-self.gamma * rij ** self.m)

                        # Mutation Vector
                        e = self.delta * uniform(-1, 1, self.problem_size)

                        pos_new = pop[i][self.ID_POS] + alpha * e +\
                                  beta * matmul((pop[j][self.ID_POS] - pop[i][self.ID_POS]), uniform(0, 1, (self.problem_size, self.problem_size)))
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)

                        # Compare to Previous Solution
                        if fit_new < pop[i][self.ID_FIT]:
                            pop_new[i] = [pos_new, fit_new]

            # Merge, Sort and Selection, update global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop + pop_new, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]

            alpha = self.alpha_damp * alpha

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
