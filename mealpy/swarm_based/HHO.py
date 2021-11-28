#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseHHO(Optimizer):
    """
        The original version of: Harris Hawks Optimization (HHO)
            (Harris Hawks Optimization: Algorithm and Applications)
        Link:
            https://doi.org/10.1016/j.future.2019.02.028
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 1.5*pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # -1 < E0 < 1
            E0 = 2 * np.random.uniform() - 1
            # factor to show the decreasing energy of rabbit
            E = 2 * E0 * (1 - (epoch + 1) * 1.0 / self.epoch)
            J = 2 * (1 - np.random.uniform())

            # -------- Exploration phase Eq. (1) in paper -------------------
            if (np.abs(E) >= 1):
                # Harris' hawks perch randomly based on 2 strategy:
                if np.random.rand() >= 0.5:  # perch based on other family members
                    X_rand = deepcopy(self.pop[np.random.randint(0, self.pop_size)][self.ID_POS])
                    pos_new = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * self.pop[idx][self.ID_POS])

                else:  # perch on a random tall tree (random site inside group's home range)
                    X_m = np.mean([x[self.ID_POS] for x in self.pop])
                    pos_new = (self.g_best[self.ID_POS] - X_m) - np.random.uniform() * \
                              (self.problem.lb + np.random.uniform() * (self.problem.ub - self.problem.lb))
                pos_new = self.amend_position_faster(pos_new)
                pop_new.append([pos_new, None])
            # -------- Exploitation phase -------------------
            else:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                if (np.random.rand() >= 0.5):
                    delta_X = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
                    if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                        pos_new = delta_X - E * np.abs(J * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:  # Soft besiege Eq. (4) in paper
                        pos_new = self.g_best[self.ID_POS] - E * np.abs(delta_X)
                    pos_new = self.amend_position_faster(pos_new)
                    pop_new.append([pos_new, None])
                else:
                    xichma = np.power((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2.0)) /
                                      (gamma((1 + 1.5) * 1.5 * np.power(2, (1.5 - 1) / 2)) / 2.0), 1.0 / 1.5)
                    LF_D = 0.01 * np.random.uniform() * xichma / np.power(np.abs(np.random.uniform()), 1.0 / 1.5)
                    if np.abs(E) >= 0.5:  # Soft besiege Eq. (10) in paper
                        Y = self.g_best[self.ID_POS] - E * np.abs(J * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:  # Hard besiege Eq. (11) in paper
                        X_m = np.mean([x[self.ID_POS] for x in self.pop])
                        Y = self.g_best[self.ID_POS] - E * np.abs(J * self.g_best[self.ID_POS] - X_m)
                    pos_Y = self.amend_position_faster(Y)
                    fit_Y = self.get_fitness_position(pos_Y)
                    Z = Y + np.random.uniform(self.problem.lb, self.problem.ub) * LF_D
                    pos_Z = self.amend_position_faster(Z)
                    fit_Z = self.get_fitness_position(pos_Z)

                    if self.compare_agent([pos_Y, fit_Y], self.pop[idx]):
                        pop_new.append([pos_Y, fit_Y])
                        continue
                    if self.compare_agent([pos_Z, fit_Z], self.pop[idx]):
                        pop_new.append([pos_Z, fit_Z])
                        continue
                    pop_new.append(deepcopy(self.pop[idx]))
        self.pop = self.update_fitness_population(pop_new)
