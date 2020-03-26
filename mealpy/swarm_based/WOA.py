#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, rand
from numpy import abs, exp, cos, pi
from mealpy.root import Root


class BaseWOA(Root):
    """
    Standard version of Whale Optimization Algorithm (belongs to Swarm-based Algorithms)
    - In this algorithms: Prey means the best solution
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_POS, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            a = 2 - 2 * epoch / (self.epoch - 1)            # linearly decreased from 2 to 0

            for i in range(self.pop_size):

                r = rand()
                A = 2 * a * r - a
                C = 2 * r
                l = uniform(-1, 1)
                p = 0.5
                b = 1
                if uniform() < p:
                    if abs(A) < 1:
                        D = abs(C * g_best[self.ID_POS] - pop[i][self.ID_POS] )
                        new_position = g_best[self.ID_POS] - A * D
                    else :
                        #x_rand = pop[np.random.randint(self.pop_size)]         # select random 1 solution in pop
                        x_rand = self._create_solution__()
                        D = abs(C * x_rand[self.ID_POS] - pop[i][self.ID_POS])
                        new_position = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    new_position = D1 * exp(b * l) * cos(2 * pi * l) + g_best[self.ID_POS]

                new_position = self._amend_solution_faster__(new_position)
                fit = self._fitness_model__(new_position)
                pop[i] = [new_position, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT],self.loss_train
