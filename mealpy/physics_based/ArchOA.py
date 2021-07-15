#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:10, 08/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import exp
from numpy.random import uniform, choice, rand
from numpy import min as np_min
from numpy import max as np_max
from mealpy.optimizer import Root


class OriginalArchOA(Root):
    """
    The original version of: Archimedes Optimization Algorithm (ArchOA)
        (Archimedes optimization algorithm: a new metaheuristic algorithm for solving optimization problems)
    Link:
        https://doi.org/10.1007/s10489-020-01893-z
    """
    ID_POS = 0
    ID_FIT = 1
    ID_DEN = 2  # Density
    ID_VOL = 3  # Volume
    ID_ACC = 4  # Acceleration

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c1=2, c2=6, c3=2, c4=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1    # Default belongs [1, 2]
        self.c2 = c2    # Default belongs [2, 4, 6]
        self.c3 = c3    # Default belongs [1, 2]
        self.c4 = c4    # Default belongs [0.5, 1]

    def create_solution(self, minmax=0):
        pos = uniform(self.lb, self.ub, self.problem_size)
        fit = self.get_fitness_position(pos)
        den = uniform(self.lb, self.ub, self.problem_size)
        vol = uniform(self.lb, self.ub, self.problem_size)
        acc = self.lb + uniform(self.lb, self.ub, self.problem_size) * (self.ub - self.lb)
        return [pos, fit, den, vol, acc]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        acc_upper = 0.9
        acc_lower = 0.1

        for epoch in range(0, self.epoch):
            tf = exp((epoch + 1) / self.epoch - 1)                          # Transfer operator Eq. 8
            ddf = exp(1 - (epoch + 1)/self.epoch) - (epoch+1)/self.epoch    # Density decreasing factor Eq. 9

            list_acc = []
            ## Calculate new density, volume and acceleration
            for i in range(0, self.pop_size):
                # Update density and volume of each object using Eq. 7
                new_den = pop[i][self.ID_DEN] + uniform() * (g_best[self.ID_DEN] - pop[i][self.ID_DEN])
                new_vol = pop[i][self.ID_VOL] + uniform() * (g_best[self.ID_VOL] - pop[i][self.ID_VOL])

                if tf <= 0.5:       # Exploration phase
                    # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                    id_rand = choice(list(set(range(0, self.pop_size)) - {i}))
                    new_acc = (pop[id_rand][self.ID_DEN] + pop[id_rand][self.ID_VOL] * pop[id_rand][self.ID_ACC]) / (new_den * new_vol)
                else:
                    new_acc = (g_best[self.ID_DEN] + g_best[self.ID_VOL] * g_best[self.ID_ACC]) / (new_den * new_vol)
                list_acc.append(new_acc)
                pop[i][self.ID_DEN] = new_den
                pop[i][self.ID_VOL] = new_vol
            min_acc = np_min(list_acc)
            max_acc = np_max(list_acc)
            ## Normalize acceleration using Eq. 12
            for i in range(0, self.pop_size):
                pop[i][self.ID_ACC] = acc_upper * (pop[i][self.ID_ACC] - min_acc) / (max_acc - min_acc) + acc_lower

            for i in range(0, self.pop_size):
                if tf <= 0.5:   # update position using Eq. 13
                    id_rand = choice(list(set(range(0, self.pop_size)) - {i}))
                    pos_new = pop[i][self.ID_POS] + self.c1 * uniform() * pop[i][self.ID_ACC] * ddf * (pop[id_rand][self.ID_POS] - pop[i][self.ID_POS])
                else:
                    p = 2 * rand()  - self.c4
                    f = 1 if p <= 0.5 else -1
                    t = self.c3 * tf
                    pos_new = g_best[self.ID_POS] + f * self.c2 * rand() * pop[i][self.ID_ACC] * ddf * (t * g_best[self.ID_POS] - pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit_new

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

