#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:51, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
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
        Before updated old solution, i check whether new solution is better or not.
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, n_s=3, n_e=3, eta=0.15, local_move=(0.9, 0.3), global_move=(0.2, 0.8), p_hi=0.9, delta=(2.0, 2.0)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_s = n_s                  # default = 3, number of exploration step
        self.n_e = n_e                  # default = 3, number of exploitation step
        self.eta = eta                  # default = 0.15, learning rate
        self.local_move = local_move    # default = (0.9, 0.3), (alpha 1, beta 1) - control local movement
        self.global_move = global_move  # default = (0.2, 0.8), (alpha 2, beta 2) - control global movement
        self.p_hi = p_hi                # default = 0.9, the probability of wildebeest move to another position based on herd instinct
        self.delta = delta              # default = (2.0, 2.0) , (delta_w, delta_c) - (dist to worst, dist to best)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Begin the Wildebeest Herd Optimization process
            for i in range(0, self.pop_size):
                ### 1. Local movement (Milling behaviour)
                local_list = []
                for j in range(0, self.n_s):
                    temp = pop[i][self.ID_POS] + self.eta * uniform() * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    local_list.append([temp, fit])
                best_local = self._get_global_best__(local_list, self.ID_FIT, self.ID_MIN_PROB)
                temp = self.local_move[0] * best_local[self.ID_POS] + self.local_move[1] * (pop[i][self.ID_POS] - best_local[self.ID_POS])
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = deepcopy([temp, fit])

            for i in range(0, self.pop_size):
                ### 2. Herd instinct
                idr = choice(range(0, self.pop_size))
                if uniform() < self.p_hi and pop[idr][self.ID_FIT] < pop[i][self.ID_FIT]:
                    temp = self.global_move[0] * pop[i][self.ID_POS] + self.global_move[1] * pop[idr][self.ID_POS]
                    fit = self._fitness_model__(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = deepcopy([temp, fit])
                        if fit < g_best[self.ID_FIT]:
                            g_best = deepcopy([temp, fit])

            g_worst = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROB)
            for i in range(0, self.pop_size):
                dist_to_worst = norm(pop[i][self.ID_POS] - g_worst[self.ID_POS])
                dist_to_best = norm(pop[i][self.ID_POS] - g_best[self.ID_POS])

                ### 3. Starvation avoidance
                if dist_to_worst < self.delta[0]:
                    temp = pop[i][self.ID_POS] + uniform() * (self.domain_range[1] - self.domain_range[0]) * \
                           uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = deepcopy([temp, fit])
                        if fit < g_best[self.ID_FIT]:
                            g_best = deepcopy([temp, fit])

                ### 4. Population pressure
                if 1.0 < dist_to_best and dist_to_best < self.delta[1]:
                      temp = g_best[self.ID_POS] + self.eta * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                      fit = self._fitness_model__(temp)
                      if fit < pop[i][self.ID_FIT]:
                          pop[i] = deepcopy([temp, fit])
                          if fit < g_best[self.ID_FIT]:
                              g_best = deepcopy([temp, fit])

                ### 5. Herd social memory
                for j in range(0, self.n_e):
                    temp = g_best[self.ID_POS] + 0.1 * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalWHO(BaseWHO):
    """
    The original version of: Wildebeest Herd Optimization (WHO)
        (Wildebeest herd optimization: A new global optimization algorithm inspired by wildebeest herding behaviour)
    Link:
        http://doi.org/10.3233/JIFS-190495
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, n_s=3, n_e=3, eta=0.15, local_move=(0.9, 0.3), global_move=(0.2, 0.8), p_hi=0.9, delta=(2.0, 2.0)):
        BaseWHO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, n_s, n_e, eta, local_move, global_move, p_hi, delta)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Begin the Wildebeest Herd Optimization process
            for i in range(0, self.pop_size):
                ### 1. Local movement (Milling behaviour)
                local_list = []
                for j in range(0, self.n_s):
                    temp = pop[i][self.ID_POS] + self.eta * uniform() * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    local_list.append([temp, fit])
                best_local = self._get_global_best__(local_list, self.ID_FIT, self.ID_MIN_PROB)
                temp = self.local_move[0] * best_local[self.ID_POS] + self.local_move[1] * (pop[i][self.ID_POS] - best_local[self.ID_POS])
                fit = self._fitness_model__(temp)
                pop[i] = deepcopy([temp, fit])
                if fit < g_best[self.ID_FIT]:
                    g_best = deepcopy([temp, fit])

            for i in range(0, self.pop_size):
                ### 2. Herd instinct
                idr = choice(range(0, self.pop_size))
                if uniform() < self.p_hi and pop[idr][self.ID_FIT] < pop[i][self.ID_FIT]:
                    temp = self.global_move[0] * pop[i][self.ID_POS] + self.global_move[1] * pop[idr][self.ID_POS]
                    fit = self._fitness_model__(temp)
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = deepcopy([temp, fit])

            g_worst = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROB)
            for i in range(0, self.pop_size):
                dist_to_worst = norm(pop[i][self.ID_POS] - g_worst[self.ID_POS])
                dist_to_best = norm(pop[i][self.ID_POS] - g_best[self.ID_POS])

                ### 3. Starvation avoidance
                if dist_to_worst < self.delta[0]:
                    temp = pop[i][self.ID_POS] + uniform() * (self.domain_range[1] - self.domain_range[0]) * \
                           uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    pop[i] = deepcopy([temp, fit])
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

                ### 4. Population pressure
                if 1.0 < dist_to_best and dist_to_best < self.delta[1]:
                      temp = g_best[self.ID_POS] + self.eta * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                      fit = self._fitness_model__(temp)
                      pop[i] = deepcopy([temp, fit])
                      if fit < g_best[self.ID_FIT]:
                          g_best = [temp, fit]

                ### 5. Herd social memory
                for j in range(0, self.n_e):
                    temp = g_best[self.ID_POS] + 0.1 * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    fit = self._fitness_model__(temp)
                    if fit < g_best[self.ID_FIT]:
                        g_best = [temp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
