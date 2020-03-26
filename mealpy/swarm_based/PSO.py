#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:49, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, random_sample
from copy import deepcopy
from mealpy.root import Root


class BasePSO(Root):
    """
    Particle Swarm Optimization
    """
    ID_POS = 0              # current position
    ID_FIT = 1              # current fitness
    ID_POS_PAST = 2         # personal best position
    ID_FIT_PAST = 3         # fitness best in the past
    ID_VEL = 4              # velocity of bird

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, c1=1.2, c2=1.2, w_min=0.4, w_max=0.9):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1            # [0-2]  -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min      # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max

    def _create_solution__(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
                x: current position
                x_past_best: the best personal position so far (in history)
                v: velocity of this bird (same number of dimension of x)
        """
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(solution=x, minmax=minmax)
        x_past_best = deepcopy(x)
        fit_past = fit
        v = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        return [x, fit, x_past_best, fit_past, v]

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                r1 = random_sample()
                r2 = random_sample()
                v_new = w * pop[i][self.ID_VEL] + self.c1 * r1 * (pop[i][self.ID_POS_PAST] - pop[i][self.ID_POS]) +\
                            self.c2 * r2 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new             # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                fit_new = self._fitness_model__(solution=x_new, minmax=0)
                fit_old = pop[i][self.ID_FIT_PAST]

                # Update current position, current velocity and compare with past position, past fitness (local best)
                pop[i][self.ID_POS] = deepcopy(x_new)
                pop[i][self.ID_VEL] = deepcopy(v_new)
                pop[i][self.ID_FIT] = fit_new

                if fit_new < fit_old:
                    pop[i][self.ID_POS_PAST] = deepcopy(x_new)
                    pop[i][self.ID_FIT_PAST] = fit_new

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

