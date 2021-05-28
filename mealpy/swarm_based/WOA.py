#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, rand
from numpy import abs, exp, cos, pi
from mealpy.root import Root


class BaseWOA(Root):
    """
        The original version of: Whale Optimization Algorithm (WOA)
            - In this algorithms: Prey means the best position
        Link:
            https://doi.org/10.1016/j.advengsoft.2016.01.008
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

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
                        #x_rand = pop[np.random.randint(self.pop_size)]         # select random 1 position in pop
                        x_rand = self.create_solution()
                        D = abs(C * x_rand[self.ID_POS] - pop[i][self.ID_POS])
                        new_position = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    new_position = D1 * exp(b * l) * cos(2 * pi * l) + g_best[self.ID_POS]

                new_position = self.amend_position_faster(new_position)
                fit = self.get_fitness_position(new_position)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [new_position, fit]

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
        return g_best[self.ID_POS], g_best[self.ID_FIT],self.loss_train
