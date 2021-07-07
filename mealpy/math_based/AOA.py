#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:56, 07/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import rand
from mealpy.root import Root


class OriginalAOA(Root):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)
    Link:
        https://doi.org/10.1016/j.cma.2020.113609
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha          # Default: 5, fixed parameter, sensitive exploitation parameter
        self.miu = miu              # Default: 0.5, fixed parameter , control parameter to adjust the search process
        self.moa_min = moa_min      # Default: 0.2, range min of Math Optimizer Accelerated
        self.moa_max = moa_max      # Default: 0.9, range max of Math Optimizer Accelerated

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            moa = self.moa_min + epoch * ((self.moa_max - self.moa_min) / self.epoch)           # Eq. 2
            mop = 1 - (epoch ** (1.0 / self.alpha)) / (self.epoch ** (1.0 / self.alpha))        # Eq. 4

            for i in range(0, self.pop_size):
                pos_new = pop[i][self.ID_POS]
                for j in range(0, self.problem_size):
                    r1, r2, r3 = rand(3)
                    if r1 > moa:        # Exploration phase
                        if r2 < 0.5:
                            pos_new[j] = g_best[self.ID_POS][j] / (mop + self.EPSILON) * ((self.ub[j] - self.lb[j]) * self.miu + self.lb[j])
                        else:
                            pos_new[j] = g_best[self.ID_POS][j] * mop  * ((self.ub[j] - self.lb[j]) * self.miu + self.lb[j])
                    else:               # Exploitation phase
                        if r3 < 0.5:
                            pos_new[j] = g_best[self.ID_POS][j] - mop * ((self.ub[j] - self.lb[j]) * self.miu + self.lb[j])
                        else:
                            pos_new[j] = g_best[self.ID_POS][j] + mop * ((self.ub[j] - self.lb[j]) * self.miu + self.lb[j])
                fit_new = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit_new]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
