#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:02, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import sin, abs, sqrt, pi, subtract, array, exp
from numpy import log as loge
from numpy.random import uniform, normal, choice, rand
from numpy.linalg import norm
from copy import deepcopy
from math import gamma
from scipy.stats import rankdata
from mealpy.root import Root


class BaseNRO(Root):
    """
    The original version of: Nuclear Reaction Optimization (NRO)
        An Approach Inspired from Nuclear Reaction Processes for Numerical Optimization
        Nuclear Reaction Optimization: A novel and powerful physics-based algorithm for global optimization
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            xichma_v = 1
            xichma_u = ((gamma(1 + 1.5) * sin(pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
            levy_b = (normal(0, xichma_u ** 2)) / (sqrt(abs(normal(0, xichma_v ** 2))) ** (1.0 / 1.5))

            # NFi phase
            Pb = uniform()
            Pfi = uniform()
            freq = 0.05
            alpha = 0.01
            for i in range(self.pop_size):

                ## Calculate neutron vector Nei by Eq. (2)
                ## Random 1 more index to select neutron
                temp1 = list(set(range(0, self.pop_size)) - {i})
                i1 = choice(temp1, replace=False)
                Nei = (pop[i][self.ID_POS] + pop[i1][self.ID_POS]) / 2
                ## Update population of fission products according to Eq.(3), (6) or (9);
                if uniform() <= Pfi:
                    ### Update based on Eq. 3
                    if uniform() <= Pb:
                        xichma1 = (loge(epoch + 1) * 1.0 / (epoch+1)) * abs( subtract(pop[i][self.ID_POS], g_best[self.ID_POS]))
                        gauss = array([normal(g_best[self.ID_POS][j], xichma1[j]) for j in range(self.problem_size)])
                        Xi = gauss + uniform() * g_best[self.ID_POS] - round(rand() + 1)*Nei
                    ### Update based on Eq. 6
                    else:
                        i2 = choice(temp1, replace=False)
                        xichma2 = (loge(epoch + 1) * 1.0 / (epoch+1)) * abs( subtract(pop[i2][self.ID_POS], g_best[self.ID_POS]))
                        gauss = array([normal(pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem_size)])
                        Xi = gauss + uniform() * g_best[self.ID_POS] - round(rand() + 2) * Nei
                ## Update based on Eq. 9
                else:
                    i3 = choice(temp1, replace=False)
                    xichma2 = (loge(epoch + 1) * 1.0 / (epoch+1)) * abs( subtract(pop[i3][self.ID_POS], g_best[self.ID_POS]))
                    Xi = array([normal(pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem_size)])

                ## Check the boundary and evaluate the fitness function
                Xi = self.amend_position_random_faster(Xi)
                fit = self.get_fitness_position(Xi, self.ID_MIN_PROB)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [Xi, fit]
                    if fit < g_best[self.ID_FIT]:
                        g_best = [Xi, fit]

            # NFu phase

            ## Ionization stage
            ## Calculate the Pa through Eq. (10);
            ranked_pop = rankdata([pop[i][self.ID_FIT] for i in range(self.pop_size)])
            for i in range(self.pop_size):
                X_ion = deepcopy(pop[i][self.ID_POS])
                if (ranked_pop[i] * 1.0 / self.pop_size) < uniform():
                    i1, i2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    for j in range(self.problem_size):
                        #### Levy flight strategy is described as Eq. 18
                        if pop[i2][self.ID_POS][j] == pop[i][self.ID_POS][j]:
                            X_ion[j] = pop[i][self.ID_POS][j] + alpha * levy_b * (pop[i][self.ID_POS][j] - g_best[self.ID_POS][j])
                        #### If not, based on Eq. 11, 12
                        else:
                            if uniform() <= 0.5:
                                X_ion[j] = pop[i1][self.ID_POS][j] + uniform() * (pop[i2][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                X_ion[j] = pop[i1][self.ID_POS][j] - uniform() * (pop[i2][self.ID_POS][j] - pop[i][self.ID_POS][j])

                else:   #### Levy flight strategy is described as Eq. 21
                    X_worst = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MAX_PROB)
                    for j in range(self.problem_size):
                        ##### Based on Eq. 21
                        if X_worst[self.ID_POS][j] == g_best[self.ID_POS][j]:
                            X_ion[j] = pop[i][self.ID_POS][j] + alpha * levy_b * (self.ub[j] - self.lb[j])
                        ##### Based on Eq. 13
                        else:
                            X_ion[j] = pop[i][self.ID_POS][j] + round(uniform()) * uniform()*(X_worst[self.ID_POS][j] - g_best[self.ID_POS][j])

                ## Check the boundary and evaluate the fitness function for X_ion
                X_ion = self.amend_position_random_faster(X_ion)
                fit = self.get_fitness_position(X_ion, self.ID_MIN_PROB)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_ion, fit]
                    if fit < g_best[self.ID_FIT]:
                        g_best = [X_ion, fit]
            ## Fusion Stage

            ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
            ranked_pop = rankdata([pop[i][self.ID_FIT] for i in range(self.pop_size)])
            for i in range(self.pop_size):
                i1, i2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

                #### Generate fusion nucleus
                if (ranked_pop[i] * 1.0 / self.pop_size) < uniform():
                    t1 = uniform() * (pop[i1][self.ID_POS] - g_best[self.ID_POS])
                    t2 = uniform() * (pop[i2][self.ID_POS] - g_best[self.ID_POS])
                    temp2 = pop[i1][self.ID_POS] - pop[i2][self.ID_POS]
                    X_fu = pop[i][self.ID_POS] + t1 + t2 - exp(-norm(temp2)) * temp2
                #### Else
                else:
                    ##### Based on Eq. 22
                    check_equal = (pop[i1][self.ID_POS] == pop[i2][self.ID_POS])
                    if check_equal.all():
                        X_fu = pop[i][self.ID_POS] + alpha * levy_b * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                    ##### Based on Eq. 16, 17
                    else:
                        if uniform() > 0.5:
                            X_fu = pop[i][self.ID_POS] - 0.5*(sin(2*pi*freq*epoch + pi)*(self.epoch - epoch)/self.epoch + 1)*(pop[i1][self.ID_POS] - pop[i2][self.ID_POS])
                        else:
                            X_fu = pop[i][self.ID_POS] - 0.5 * (sin(2 * pi * freq * epoch + pi) * epoch / self.epoch + 1) * (pop[i1][self.ID_POS] - pop[i2][self.ID_POS])

                X_fu = self.amend_position_random_faster(X_fu)
                fit = self.get_fitness_position(X_fu, self.ID_MIN_PROB)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_fu, fit]
                    if fit < g_best[self.ID_FIT]:
                        g_best = [X_fu, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train