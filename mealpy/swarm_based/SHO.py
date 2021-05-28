#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 10:55, 02/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                        #
#-------------------------------------------------------------------------------------------------------#

from numpy import Inf, dot, abs, mean, array
from numpy.random import uniform, choice
from mealpy.root import Root


class BaseSHO(Root):
    """
    My modified version: Spotted Hyena Optimizer (SHO)
        (Spotted hyena optimizer: A novel bio-inspired based metaheuristic technique for engineering applications)
    Link:
        https://doi.org/10.1016/j.advengsoft.2017.05.014
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 h=5, M=(0.5, 1), N_tried=10, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.h = h                  # default = 5, coefficient linearly decreased from 5 to 0
        self.M = M                  # default = [0.5, 1], random vector in [0.5, 1]
        self.N_tried = N_tried      # default = 10,

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Epoch loop
        for epoch in range(self.epoch):

            ## Each individual loop
            for i in range(self.pop_size):

                h = 5 - (epoch + 1.0) * (5 / self.epoch)
                rd1 = uniform(0, 1, self.problem_size)
                rd2 = uniform(0, 1, self.problem_size)
                B = 2 * rd1
                E = 2 * h * rd2 - h

                if uniform() < 0.5:
                    D_h = abs( dot(B, g_best[self.ID_POS]) - pop[i][self.ID_POS] )
                    x_new = g_best[self.ID_POS] - dot(E, D_h)
                else:
                    N = 0
                    fit = Inf
                    while fit > g_best[self.ID_FIT] and N < self.N_tried:
                        temp = g_best[self.ID_POS] + uniform(self.M[0], self.M[1], self.problem_size)
                        fit = self.get_fitness_position(temp)
                        N += 1
                    circle_list = []
                    idx_list = choice(range(0, self.pop_size), N, replace=False)
                    for j in range(0, N):
                        D_h = abs(dot(B, g_best[self.ID_POS]) - pop[idx_list[j]][self.ID_POS])
                        p_k = g_best[self.ID_POS] - dot(E, D_h)
                        circle_list.append(p_k)
                    x_new = mean(array(circle_list))
                x_new = self.amend_position_faster(x_new)
                fit = self.get_fitness_position(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_new, fit]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
