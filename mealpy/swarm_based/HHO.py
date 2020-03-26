#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import mean, abs, power, pi, sin
from numpy.random import uniform, randint
from copy import deepcopy
from math import gamma
from mealpy.root import Root


class BaseHHO(Root):
    """
    Harris Hawks Optimization: Algorithm and Applications
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Update the location of Harris' hawks
            for i in range(0, self.pop_size):
                E0 = 2 * uniform() - 1                        # -1 < E0 < 1
                E = 2 * E0 * (1 - (epoch + 1) * 1.0 / self.epoch)       # factor to show the decreasing energy of rabbit
                J = 2 * (1 - uniform())

                # -------- Exploration phase Eq. (1) in paper -------------------
                if (abs(E) >= 1):
                    # Harris' hawks perch randomly based on 2 strategy:
                    if (uniform() >= 0.5):        # perch based on other family members
                        X_rand = deepcopy(pop[randint(0, self.pop_size)][self.ID_POS])
                        pop[i][self.ID_POS] = X_rand - uniform() * abs(X_rand - 2 * uniform() * pop[i][self.ID_POS])

                    else:           # perch on a random tall tree (random site inside group's home range)
                        X_m = mean([x[self.ID_POS] for x in pop])
                        pop[i][self.ID_POS] = (g_best[self.ID_POS] - X_m) - uniform()*(
                            self.domain_range[0] + uniform() * (self.domain_range[1] - self.domain_range[0]))

                # -------- Exploitation phase -------------------
                else:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                    if (uniform() >= 0.5):
                        delta_X = g_best[self.ID_POS] - pop[i][self.ID_POS]
                        if (abs(E) >= 0.5):          # Hard besiege Eq. (6) in paper
                            pop[i][self.ID_POS] = delta_X - E * abs( J * g_best[self.ID_POS] - pop[i][self.ID_POS] )
                        else:                           # Soft besiege Eq. (4) in paper
                            pop[i][self.ID_POS] = g_best[self.ID_POS] - E * abs(delta_X)
                    else:
                        xichma = power((gamma(1 + 1.5) * sin(pi * 1.5 / 2.0)) / (gamma((1 + 1.5) * 1.5 * power(2, (1.5 - 1) / 2)) / 2.0), 1.0 / 1.5)
                        LF_D = 0.01 * uniform() * xichma / power(abs(uniform()), 1.0 / 1.5)
                        fit_Y, Y = None, None
                        if (abs(E) >= 0.5):      # Soft besiege Eq. (10) in paper
                            Y = g_best[self.ID_POS] - E * abs( J * g_best[self.ID_POS] - pop[i][self.ID_POS] )
                            fit_Y = self._fitness_model__(Y)
                        else:                       # Hard besiege Eq. (11) in paper
                            X_m = mean([x[self.ID_POS] for x in pop])
                            Y = g_best[self.ID_POS] - E * abs( J * g_best[self.ID_POS] - X_m )
                            fit_Y = self._fitness_model__(Y)

                        Z = Y + uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * LF_D
                        fit_Z = self._fitness_model__(Z)

                        if fit_Y < pop[i][self.ID_FIT]:
                            pop[i] = [Y, fit_Y]
                        if fit_Z < pop[i][self.ID_FIT]:
                            pop[i] = [Z, fit_Z]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

