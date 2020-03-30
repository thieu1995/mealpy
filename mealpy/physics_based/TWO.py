#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randn, random
from numpy import ones, power
from copy import deepcopy
from mealpy.root import Root


class BaseTWO(Root):
    """
    The original version of: Tug of War Optimization
        A novel meta-heuristic algorithm: tug of war optimization
    """
    ID_POS = 0
    ID_FIT = 1
    ID_WEIGHT = 2

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def _create_solution__(self, minmax=0):
        solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        weight = 0.0
        return [solution, fitness, weight]

    def _update_weight__(self, teams):
        best_fitness = max(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        worst_fitness = min(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        for i in range(self.pop_size):
            teams[i][self.ID_WEIGHT] = (teams[i][self.ID_FIT] - worst_fitness)/(best_fitness - worst_fitness) + 1
        return teams

    def _update_fit__(self, teams):
        for i in range(self.pop_size):
            teams[i][self.ID_FIT] = self._fitness_model__(teams[i][self.ID_POS], minmax=self.ID_MAX_PROB)
        return teams

    def _amend_and_return_pop__(self, pop_old, pop_new, g_best, epoch):
        for i in range(self.pop_size):
            for j in range(self.problem_size):
                if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                    if random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = g_best[self.ID_POS][j] + randn()/(epoch+1)*(g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                            pop_new[i][self.ID_POS][j] = pop_old[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[0]
                        if pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[1]
        return pop_new

    def _train__(self):
        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range( self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                       force = max(pop_old[i][self.ID_WEIGHT]*self.muy_s, pop_old[j][self.ID_WEIGHT]*self.muy_s)
                       resultant_force = force - pop_old[i][self.ID_WEIGHT]*self.muy_k
                       g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                       acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*self.muy_k)
                       delta_x = 1/2*acceleration + power(self.alpha,epoch+1)\
                                *self.beta*(self.domain_range[1] -  self.domain_range[0])*randn(self.problem_size)
                       pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        return g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON, self.loss_train


class OppoTWO(BaseTWO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseTWO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * self.muy_s, pop_old[j][self.ID_WEIGHT] * self.muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * self.muy_k
                        g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * self.muy_k)
                        delta_x = 1 / 2 * acceleration + power(self.alpha, epoch + 1) \
                                  * self.beta * (self.domain_range[1] - self.domain_range[0]) * \
                                  randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self._create_opposition_solution__(pop_old[i][self.ID_POS], g_best[self.ID_POS])
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROB)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        return g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON, self.loss_train


class OTWO(BaseTWO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseTWO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop_temp = [self._create_solution__(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            item_oppo = self.domain_range[1] * ones(self.problem_size) + self.domain_range[0] * ones(self.problem_size) - pop_temp[i][self.ID_POS]
            fit = self._fitness_model__(item_oppo, self.ID_MAX_PROB)
            pop_oppo[i] = [item_oppo, fit, 0.0]
        pop_oppo = pop_temp + pop_oppo
        pop_old = sorted(pop_oppo, key=lambda item: item[self.ID_FIT])[self.pop_size:]
        pop_old = self._update_weight__(pop_old)

        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * self.muy_s, pop_old[j][self.ID_WEIGHT] * self.muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * self.muy_k
                        g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * self.muy_k)
                        delta_x = 1 / 2 * acceleration + power(self.alpha, epoch + 1) \
                                  * self.beta * (self.domain_range[1] - self.domain_range[0]) * \
                                  randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self._create_opposition_solution__(pop_old[i][self.ID_POS], g_best[self.ID_POS])
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROB)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        return g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON, self.loss_train


class LevyTWO(BaseTWO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseTWO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                if uniform() < 0.5:
                    pop_new[i][self.ID_POS] = self._levy_flight__(epoch=epoch, solution=pop_new[i][self.ID_POS], g_best=g_best[self.ID_POS], step=0.001, case=1)
                else:
                    for j in range( self.pop_size):
                        if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                           force = max(pop_old[i][self.ID_WEIGHT]*self.muy_s, pop_old[j][self.ID_WEIGHT]*self.muy_s)
                           resultant_force = force - pop_old[i][self.ID_WEIGHT]*self.muy_k
                           g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                           acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*self.muy_k)
                           delta_x = 1/2*acceleration + power(self.alpha,epoch+1)*self.beta*\
                                     (self.domain_range[1] -  self.domain_range[0])*randn(self.problem_size)
                           pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        return g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON, self.loss_train


class ITWO(OppoTWO, LevyTWO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        OppoTWO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                if uniform() < 0.5:
                    pop_new[i][self.ID_POS] = self._levy_flight__(epoch=epoch, solution=pop_new[i][self.ID_POS], g_best=g_best[self.ID_POS], step=0.001, case=1)
                else:
                    for j in range(self.pop_size):
                        if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                            force = max(pop_old[i][self.ID_WEIGHT] * self.muy_s, pop_old[j][self.ID_WEIGHT] * self.muy_s)
                            resultant_force = force - pop_old[i][self.ID_WEIGHT] * self.muy_k
                            g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                            acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * self.muy_k)
                            delta_x = 1 / 2 * acceleration + power(self.alpha, epoch + 1) * self.beta * \
                                      (self.domain_range[1] - self.domain_range[0]) * randn(self.problem_size)
                            pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self._create_opposition_solution__(pop_old[i][self.ID_POS], g_best[self.ID_POS])
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROB)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        return g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON, self.loss_train
