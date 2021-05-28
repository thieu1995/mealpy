#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import mean, abs, power, pi, sin
from numpy.random import uniform, randint
from copy import deepcopy
from math import gamma
from mealpy.root import Root


class BaseHHO(Root):
    """
        The original version of: Harris Hawks Optimization (HHO)
            (Harris Hawks Optimization: Algorithm and Applications)
        Link:
            https://doi.org/10.1016/j.future.2019.02.028
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
                        pop[i][self.ID_POS] = (g_best[self.ID_POS] - X_m) - uniform()*(self.lb + uniform() * (self.ub - self.lb))

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
                        if (abs(E) >= 0.5):      # Soft besiege Eq. (10) in paper
                            Y = g_best[self.ID_POS] - E * abs( J * g_best[self.ID_POS] - pop[i][self.ID_POS] )
                            fit_Y = self.get_fitness_position(Y)
                        else:                       # Hard besiege Eq. (11) in paper
                            X_m = mean([x[self.ID_POS] for x in pop])
                            Y = g_best[self.ID_POS] - E * abs( J * g_best[self.ID_POS] - X_m )
                            fit_Y = self.get_fitness_position(Y)

                        Z = Y + uniform(self.lb, self.ub) * LF_D
                        fit_Z = self.get_fitness_position(Z)

                        if fit_Y < pop[i][self.ID_FIT]:
                            pop[i] = [Y, fit_Y]
                        if fit_Z < pop[i][self.ID_FIT]:
                            pop[i] = [Z, fit_Z]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

