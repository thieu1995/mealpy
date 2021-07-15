#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:53, 07/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import normal, rand, choice
from numpy import mean, pi, sin, cos, array
from math import gamma
from mealpy.optimizer import Root


class OriginalAO(Root):
    """
    The original version of: Aquila Optimization (AO)
    Link:
        Aquila Optimizer: A novel meta-heuristic optimization Algorithm
        https://doi.org/10.1016/j.cie.2021.107250
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def get_simple_levy_step(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = normal(0, 1, self.problem_size) * sigma
        v = normal(1, self.problem_size)
        step = u / abs(v) ** (1 / beta)
        return step

    def train(self):
        alpha = 0.1
        delta = 0.1

        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            g1 = 2 * rand() - 1                 # Eq. 16
            g2 = 2 * (1 - epoch / self.epoch)   # Eq. 17

            dim_list = array(list(range(1, self.problem_size + 1)))
            miu = 0.00565
            r0 = 10
            r = r0 + miu * dim_list
            w = 0.005
            phi0 = 3 * pi / 2
            phi = -w * dim_list + phi0
            x = r * sin(phi)           # Eq.(9)
            y = r * cos(phi)           # Eq.(10)
            QF = (epoch+1) ** ((2 * rand() - 1) / (1 - self.epoch) ** 2)   # Eq.(15)        Quality function

            for i in range(self.pop_size):
                x_mean = mean(array([item[self.ID_FIT] for item in pop]), axis=0)
                if (epoch+1) <= (2/3) * self.epoch:        # Eq. 3, 4
                    if rand() < 0.5:
                        pos_new = g_best[self.ID_POS] * (1 - (epoch+1)/self.epoch) + rand() * (x_mean - g_best[self.ID_POS])
                    else:
                        idx = choice(list(set(range(0, self.pop_size)) - {i}))
                        pos_new = g_best[self.ID_POS] * self.get_simple_levy_step() + pop[idx][self.ID_POS] + rand() * (y - x)          # Eq. 5
                else:
                    if rand() < 0.5:
                        pos_new = alpha * (g_best[self.ID_POS] - x_mean) - rand() * (rand() * (self.ub - self.lb) + self.lb) * delta    # Eq. 13
                    else:
                        pos_new = QF * g_best[self.ID_POS] - (g2 * pop[i][self.ID_POS] * rand()) - g2 * self.get_simple_levy_step() + rand() * g1   # Eq. 14
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

