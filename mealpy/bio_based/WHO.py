#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:51, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy.linalg import norm
from copy import deepcopy
from mealpy.root import Root


class BaseWHO(Root):
    """
    My version of: Wildebeest Herd Optimization (WHO)
        (Wildebeest herd optimization: A new global optimization algorithm inspired by wildebeest herding behaviour)
    Noted:
        + Before updated old position, i check whether new position is better or not.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_s=3, n_e=3, eta=0.15, local_move=(0.9, 0.3), global_move=(0.2, 0.8), p_hi=0.9, delta=(2.0, 2.0), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_s = n_s                  # default = 3, number of exploration step
        self.n_e = n_e                  # default = 3, number of exploitation step
        self.eta = eta                  # default = 0.15, learning rate
        self.local_move = local_move    # default = (0.9, 0.3), (alpha 1, beta 1) - control local movement
        self.global_move = global_move  # default = (0.2, 0.8), (alpha 2, beta 2) - control global movement
        self.p_hi = p_hi                # default = 0.9, the probability of wildebeest move to another position based on herd instinct
        self.delta = delta              # default = (2.0, 2.0) , (delta_w, delta_c) - (dist to worst, dist to best)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Begin the Wildebeest Herd Optimization process
            for i in range(0, self.pop_size):
                ### 1. Local movement (Milling behaviour)
                local_list = []
                for j in range(0, self.n_s):
                    temp = pop[i][self.ID_POS] + self.eta * uniform() * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    local_list.append([temp, fit])
                best_local = self.get_global_best_solution(local_list, self.ID_FIT, self.ID_MIN_PROB)
                temp = self.local_move[0] * best_local[self.ID_POS] + self.local_move[1] * (pop[i][self.ID_POS] - best_local[self.ID_POS])
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = deepcopy([temp, fit])

            for i in range(0, self.pop_size):
                ### 2. Herd instinct
                idr = choice(range(0, self.pop_size))
                if uniform() < self.p_hi and pop[idr][self.ID_FIT] < pop[i][self.ID_FIT]:
                    temp = self.global_move[0] * pop[i][self.ID_POS] + self.global_move[1] * pop[idr][self.ID_POS]
                    fit = self.get_fitness_position(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = deepcopy([temp, fit])
                        if fit < g_best[self.ID_FIT]:
                            g_best = deepcopy([temp, fit])

            g_worst = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MAX_PROB)
            for i in range(0, self.pop_size):
                dist_to_worst = norm(pop[i][self.ID_POS] - g_worst[self.ID_POS])
                dist_to_best = norm(pop[i][self.ID_POS] - g_best[self.ID_POS])

                ### 3. Starvation avoidance
                if dist_to_worst < self.delta[0]:
                    temp = pop[i][self.ID_POS] + uniform() * (self.ub - self.lb) * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = deepcopy([temp, fit])
                        if fit < g_best[self.ID_FIT]:
                            g_best = deepcopy([temp, fit])

                ### 4. Population pressure
                if 1.0 < dist_to_best and dist_to_best < self.delta[1]:
                      temp = g_best[self.ID_POS] + self.eta * uniform(self.lb, self.ub)
                      fit = self.get_fitness_position(temp)
                      if fit < pop[i][self.ID_FIT]:
                          pop[i] = deepcopy([temp, fit])
                          if fit < g_best[self.ID_FIT]:
                              g_best = deepcopy([temp, fit])

                ### 5. Herd social memory
                for j in range(0, self.n_e):
                    temp = g_best[self.ID_POS] + 0.1 * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalWHO(BaseWHO):
    """
    The original version of: Wildebeest Herd Optimization (WHO)
        (Wildebeest herd optimization: A new global optimization algorithm inspired by wildebeest herding behaviour)
    Link:
        http://doi.org/10.3233/JIFS-190495
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_s=3, n_e=3, eta=0.15, local_move=(0.9, 0.3), global_move=(0.2, 0.8), p_hi=0.9, delta=(2.0, 2.0), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_s = n_s                  # default = 3, number of exploration step
        self.n_e = n_e                  # default = 3, number of exploitation step
        self.eta = eta                  # default = 0.15, learning rate
        self.local_move = local_move    # default = (0.9, 0.3), (alpha 1, beta 1) - control local movement
        self.global_move = global_move  # default = (0.2, 0.8), (alpha 2, beta 2) - control global movement
        self.p_hi = p_hi                # default = 0.9, the probability of wildebeest move to another position based on herd instinct
        self.delta = delta              # default = (2.0, 2.0) , (delta_w, delta_c) - (dist to worst, dist to best)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Begin the Wildebeest Herd Optimization process
            for i in range(0, self.pop_size):
                ### 1. Local movement (Milling behaviour)
                local_list = []
                for j in range(0, self.n_s):
                    temp = pop[i][self.ID_POS] + self.eta * uniform() * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    local_list.append([temp, fit])
                best_local = self.get_global_best_solution(local_list, self.ID_FIT, self.ID_MIN_PROB)
                temp = self.local_move[0] * best_local[self.ID_POS] + self.local_move[1] * (pop[i][self.ID_POS] - best_local[self.ID_POS])
                fit = self.get_fitness_position(temp)
                pop[i] = deepcopy([temp, fit])
                if fit < g_best[self.ID_FIT]:
                    g_best = deepcopy([temp, fit])

            for i in range(0, self.pop_size):
                ### 2. Herd instinct
                idr = choice(range(0, self.pop_size))
                if uniform() < self.p_hi and pop[idr][self.ID_FIT] < pop[i][self.ID_FIT]:
                    temp = self.global_move[0] * pop[i][self.ID_POS] + self.global_move[1] * pop[idr][self.ID_POS]
                    fit = self.get_fitness_position(temp)
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = deepcopy([temp, fit])

            g_worst = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MAX_PROB)
            for i in range(0, self.pop_size):
                dist_to_worst = norm(pop[i][self.ID_POS] - g_worst[self.ID_POS])
                dist_to_best = norm(pop[i][self.ID_POS] - g_best[self.ID_POS])

                ### 3. Starvation avoidance
                if dist_to_worst < self.delta[0]:
                    temp = pop[i][self.ID_POS] + uniform() * (self.ub - self.lb) * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

                ### 4. Population pressure
                if 1.0 < dist_to_best and dist_to_best < self.delta[1]:
                      temp = g_best[self.ID_POS] + self.eta * uniform(self.lb, self.ub)
                      fit = self.get_fitness_position(temp)
                      pop[i] = deepcopy([temp, fit])
                      if fit < g_best[self.ID_FIT]:
                          g_best = [temp, fit]

                ### 5. Herd social memory
                for j in range(0, self.n_e):
                    temp = g_best[self.ID_POS] + 0.1 * uniform(self.lb, self.ub)
                    fit = self.get_fitness_position(temp)
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
