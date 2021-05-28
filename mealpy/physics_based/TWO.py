#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randn, random, normal
from numpy import power, where, logical_and, logical_or
from copy import deepcopy
from mealpy.root import Root


class BaseTWO(Root):
    """
    The original version of: Tug of War Optimization (TWO)
        A novel meta-heuristic algorithm: tug of war optimization
    Link:
        https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_WEIGHT = 2

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def create_solution(self, minmax=0):
        solution = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=solution, minmax=minmax)
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
            teams[i][self.ID_FIT] = self.get_fitness_position(teams[i][self.ID_POS], minmax=self.ID_MAX_PROB)
        return teams

    def _amend_and_return_pop__(self, pop_old, pop_new, g_best, epoch):
        for i in range(self.pop_size):
            for j in range(self.problem_size):
                if pop_new[i][self.ID_POS][j] < self.lb[j] or pop_new[i][self.ID_POS][j] > self.ub[j]:
                    if random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = g_best[self.ID_POS][j] + randn()/(epoch+1)*(g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.lb[j] or pop_new[i][self.ID_POS][j] > self.ub[j]:
                            pop_new[i][self.ID_POS][j] = pop_old[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.lb[j]:
                           pop_new[i][self.ID_POS][j] = self.lb[j]
                        if pop_new[i][self.ID_POS][j] > self.ub[j]:
                           pop_new[i][self.ID_POS][j] = self.ub[j]
        return pop_new

    def train(self):
        pop_old = [self.create_solution(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
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
                       delta_x = 1/2*acceleration + power(self.alpha,epoch+1)*self.beta*(self.ub - self.lb)*randn(self.problem_size)
                       pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self.get_global_best_solution(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        g_best = [g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON]
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OppoTWO(BaseTWO):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseTWO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop_temp = [self.create_solution(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            item_oppo = self.ub + self.lb - pop_temp[i][self.ID_POS]
            fit = self.get_fitness_position(item_oppo, self.ID_MAX_PROB)
            pop_oppo[i] = [item_oppo, fit, 0.0]
        pop_oppo = pop_temp + pop_oppo
        pop_old = sorted(pop_oppo, key=lambda item: item[self.ID_FIT])[self.pop_size:]
        pop_old = self._update_weight__(pop_old)

        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):

            ## Apply force of others solution on each individual solution
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * self.muy_s, pop_old[j][self.ID_WEIGHT] * self.muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * self.muy_k
                        g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * self.muy_k)
                        delta_x = 1 / 2 * acceleration + power(self.alpha, epoch + 1) * self.beta * (self.ub - self.lb) * randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            ## Amend solution and update fitness value
            for i in range(self.pop_size):
                pos_new = g_best[self.ID_POS] + normal(0, 1, self.problem_size) / (epoch + 1) * (g_best[self.ID_POS] - pop_new[i][self.ID_POS])
                conditions = logical_or(pop_new[i][self.ID_POS] < self.lb, pop_new[i][self.ID_POS] > self.ub)
                conditions = logical_and(conditions, uniform(0, 1, self.problem_size) < 0.5)
                pos_new = where(conditions, pos_new, pop_old[i][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)

                pop_new[i][self.ID_FIT] = self.get_fitness_position(pos_new, minmax=self.ID_MAX_PROB)
                pop_new[i][self.ID_POS] = pos_new

            ## Opposition-based here
            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self.create_opposition_position(pop_old[i][self.ID_POS], g_best[self.ID_POS])
                    fit_op = self.get_fitness_position(C_op, self.ID_MAX_PROB)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self.get_global_best_solution(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        g_best = [g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON]
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyTWO(BaseTWO):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseTWO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop_old = [self.create_solution(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for k in range( self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[k][self.ID_WEIGHT]:
                       force = max(pop_old[i][self.ID_WEIGHT]*self.muy_s, pop_old[k][self.ID_WEIGHT]*self.muy_s)
                       resultant_force = force - pop_old[i][self.ID_WEIGHT]*self.muy_k
                       g = pop_old[k][self.ID_POS] - pop_old[i][self.ID_POS]
                       acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*self.muy_k)
                       delta_x = 1/2*acceleration + power(self.alpha,epoch+1)*self.beta* (self.ub - self.lb) * randn(self.problem_size)
                       pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            ### Apply levy-flight here
            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    pos_new = self.levy_flight(epoch, pop_new[i][self.ID_POS], g_best[self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new, minmax=self.ID_MAX_PROB)
                    pop_old[i] = [pos_new, fit_new, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self.get_global_best_solution(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        g_best = [g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON]
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedTWO(OppoTWO, LevyTWO):
    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseTWO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)
        LevyTWO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):

        pop_temp = [self.create_solution(minmax=self.ID_MAX_PROB) for _ in range(self.pop_size)]
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            item_oppo = self.ub + self.lb - pop_temp[i][self.ID_POS]
            fit = self.get_fitness_position(item_oppo, self.ID_MAX_PROB)
            pop_oppo[i] = [item_oppo, fit, 0.0]
        pop_oppo = pop_temp + pop_oppo
        pop_old = sorted(pop_oppo, key=lambda item: item[self.ID_FIT])[self.pop_size:]
        pop_old = self._update_weight__(pop_old)

        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for k in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[k][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * self.muy_s, pop_old[k][self.ID_WEIGHT] * self.muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * self.muy_k
                        g = pop_old[k][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * self.muy_k)
                        delta_x = 1 / 2 * acceleration + power(self.alpha, epoch + 1) * self.beta * (self.ub - self.lb) * randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self.create_opposition_position(pop_old[i][self.ID_POS], g_best[self.ID_POS])
                    fit_op = self.get_fitness_position(C_op, self.ID_MAX_PROB)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
                    else:
                        pos_new = self.levy_flight(epoch, pop_new[i][self.ID_POS], g_best[self.ID_POS])
                        fit_new = self.get_fitness_position(pos_new, minmax=self.ID_MAX_PROB)
                        pop_old[i] = [pos_new, fit_new, 0.0]

            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self.get_global_best_solution(pop_old, self.ID_FIT, self.ID_MAX_PROB)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT] - self.EPSILON)
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1.0 / g_best[self.ID_FIT] - self.EPSILON))
        g_best = [g_best[self.ID_POS], 1.0 / g_best[self.ID_FIT] - self.EPSILON]
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
