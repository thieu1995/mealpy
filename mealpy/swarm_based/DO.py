#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 04:43, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, sum, all, abs, zeros, any, pi, sin
from numpy.random import rand, randn
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseDO(Root):
    """
    The original version of: Dragonfly Optimization (DO)
    Link:
        https://link.springer.com/article/10.1007/s00521-015-1920-1
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def dragonfly_levy(self):
        beta = 3 / 2
        # Eq.(3.10)
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = randn(self.problem_size) * sigma
        v = randn(self.problem_size)
        step = u / abs(v) ** (1 / beta)
        # Eq.(3.9)
        return 0.01 * step

    def train(self):
        # Initial radius of dragonflies' neighbouhoods
        r = (self.ub - self.lb) / 10
        delta_max = (self.ub - self.lb) / 10

        # Initial population
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        g_best, g_worst = self.get_global_best_global_worst_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        pop_delta = [self.create_solution(minmax=0) for _ in range(self.pop_size)]

        # Main loop
        for epoch in range(0, self.epoch):

            r = (self.ub - self.lb) / 4 + ((self.ub - self.lb) * (2 * (epoch+1) / self.epoch))
            w = 0.9 - (epoch+1) * ((0.9 - 0.4) / self.epoch)
            my_c = 0.1 - (epoch+1) * ((0.1 - 0) / (self.epoch / 2))
            my_c = 0 if my_c < 0 else my_c

            s = 2 * rand() * my_c     # Seperation weight
            a = 2 * rand() * my_c     # Alignment weight
            c = 2 * rand() * my_c     # Cohesion weight
            f = 2 * rand()            # Food attraction weight
            e = my_c                  # Enemy distraction weight

            for i in range(0, self.pop_size):
                pos_neighbours = []
                pos_neighbours_delta = []
                neighbours_num = 0
                # Find the neighbouring solutions
                for j in range(0, self.pop_size):
                    dist = abs(pop[i][self.ID_POS] - pop[j][self.ID_POS])
                    if all(dist <= r) and all(dist != 0):
                        neighbours_num += 1
                        pos_neighbours.append(deepcopy(pop[j][self.ID_POS]))
                        pos_neighbours_delta.append(deepcopy(pop_delta[j][self.ID_POS]))

                pos_neighbours = array(pos_neighbours)
                pos_neighbours_delta = array(pos_neighbours_delta)

                # Separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
                if neighbours_num > 1:
                    S = sum(pos_neighbours, axis=0) - neighbours_num * pop[i][self.ID_POS]
                    A = sum(pos_neighbours_delta, axis=0) / neighbours_num
                    C_temp = sum(pos_neighbours, axis=0) / neighbours_num
                else:
                    S = zeros(self.problem_size)
                    A = deepcopy(pop_delta[i][self.ID_POS])
                    C_temp = deepcopy(pop[i][self.ID_POS])
                C = C_temp - pop[i][self.ID_POS]

                # Attraction to food: Eq 3.4
                dist_to_food = abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                if all(dist_to_food <= r):
                    F = g_best[self.ID_POS] - pop[i][self.ID_POS]
                else:
                    F = zeros(self.problem_size)

                # Distraction from enemy: Eq 3.5
                dist_to_enemy = abs(pop[i][self.ID_POS] - g_worst[self.ID_POS])
                if all(dist_to_enemy <= r):
                    enemy = g_worst[self.ID_POS] + pop[i][self.ID_POS]
                else:
                    enemy = zeros(self.problem_size)

                if any(dist_to_food > r):
                    if neighbours_num > 1:
                        for j in range(0, self.problem_size):
                            temp = w * pop_delta[i][self.ID_POS][j] + rand()*A[j] + rand() * C[j] + rand() * S[j]
                            if temp > delta_max[j]:
                                temp = delta_max[j]
                            if temp < -delta_max[j]:
                                temp = -delta_max[j]
                            pop_delta[i][self.ID_POS][j] = temp
                            pop[i][self.ID_POS][j] += temp
                    else:   # Eq. 3.8
                        pop[i][self.ID_POS] += self.dragonfly_levy() * pop[i][self.ID_POS]
                        pop_delta[i][self.ID_POS] = zeros(self.problem_size)
                else:
                    for j in range(0, self.problem_size):
                        # Eq. 3.6
                        temp = (a * A[j] + c * C[j] + s * S[j] + f * F[j] + e * enemy[j]) + w * pop_delta[i][self.ID_POS][j]
                        if temp > delta_max[j]:
                            temp = delta_max[j]
                        if temp < -delta_max[j]:
                            temp = -delta_max[j]
                        pop_delta[i][self.ID_POS][j] = temp
                        pop[i][self.ID_POS][j] += temp

                # Amend solution
                pop[i][self.ID_POS] = self.amend_position_faster(pop[i][self.ID_POS])
                pop_delta[i][self.ID_POS] = self.amend_position_faster(pop_delta[i][self.ID_POS])

            # Update fitness of all solution
            for i in range(0, self.pop_size):
                pop[i][self.ID_FIT] = self.get_fitness_position(pop[i][self.ID_POS])
                pop_delta[i][self.ID_FIT] = self.get_fitness_position(pop_delta[i][self.ID_POS])

            # Update global best and global worst solution
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fitness: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
