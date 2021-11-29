#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseTWO(Optimizer):
    """
    The original version of: Tug of War Optimization (TWO)
        A novel meta-heuristic algorithm: tug of war optimization
    Link:
        https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_WEIGHT = 2

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def create_solution(self, minmax=0):
        solution = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=solution)
        weight = 0.0
        return [solution, fitness, weight]

    def _update_weight(self, teams):
        _, best, worst = self.get_special_solutions(teams, best=1, worst=1)
        best_fit = best[0][self.ID_FIT][self.ID_TAR]
        worst_fit = worst[0][self.ID_FIT][self.ID_TAR]
        if best_fit == worst_fit:
            for i in range(self.pop_size):
                teams[i][self.ID_WEIGHT] = np.random.uniform(0.5, 1.5)
        else:
            for i in range(self.pop_size):
                teams[i][self.ID_WEIGHT] = (teams[i][self.ID_FIT][self.ID_TAR] - worst_fit)/(best_fit - worst_fit + self.EPSILON) + 1
        return teams

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        self.pop = self._update_weight(self.pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pop_new[i][self.ID_POS] += delta_x
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):
                if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.lb[j]
                        if pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.ub[j]
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)


class OppoTWO(BaseTWO):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def initialization(self):
        pop_temp = self.create_population(int(self.pop_size/2))
        pop_oppo = []
        for i in range(len(pop_temp)):
            item_oppo = self.problem.ub + self.problem.lb - pop_temp[i][self.ID_POS]
            pop_oppo.append([item_oppo, None, 0.0])
        pop_oppo = self.update_fitness_population(pop_oppo)
        self.pop = pop_temp + pop_oppo
        self.pop = self._update_weight(self.pop)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Apply force of others solution on each individual solution
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    self.pop[i][self.ID_POS] += delta_x

        ## Amend solution and update fitness value
        for i in range(self.pop_size):
            pos_new = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) / (epoch + 1) * \
                      (self.g_best[self.ID_POS] - pop_new[i][self.ID_POS])
            conditions = np.logical_or(pop_new[i][self.ID_POS] < self.problem.lb, pop_new[i][self.ID_POS] > self.problem.ub)
            conditions = np.logical_and(conditions, np.random.uniform(0, 1, self.problem.n_dims) < 0.5)
            pos_new = np.where(conditions, pos_new, self.pop[i][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new[i][self.ID_POS] = pos_new
        pop_new = self.update_fitness_population(pop_new)

        ## Opposition-based here
        for i in range(self.pop_size):
            if self.compare_agent(self.pop[i], pop_new[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                C_op = self.create_opposition_position(self.pop[i][self.ID_POS], self.g_best[self.ID_POS])
                fit_op = self.get_fitness_position(C_op)
                if self.compare_agent(self.pop[i], [C_op, fit_op]):
                    self.pop[i] = [C_op, fit_op, 0.0]
        self.pop = self._update_weight(self.pop)


class LevyTWO(BaseTWO):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pop_new[i][self.ID_POS] += delta_x
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):
                if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.lb[j]
                        if pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.ub[j]

        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)

        ### Apply levy-flight here
        for i in range(self.pop_size):
            if self.compare_agent(self.pop[i], pop_new[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                pos_new = pop_new[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * levy_step
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                self.pop[i] = [pos_new, fit_new, 0.0]
        self.pop = self._update_weight(pop_new)


class ImprovedTWO(OppoTWO, LevyTWO):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def initialization(self):
        pop_temp = self.create_population(self.pop_size)
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            item_oppo = self.problem.ub + self.problem.lb - pop_temp[i][self.ID_POS]
            pop_oppo[i][self.ID_POS] = item_oppo
        pop_oppo = self.update_fitness_population(pop_oppo)
        self.pop = self.get_sorted_strim_population(pop_temp + pop_oppo, self.pop_size)
        self.pop = self._update_weight(self.pop)
        self.g_best = deepcopy(self.pop[0])

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pop_new[i][self.ID_POS] += delta_x
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):
                if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j] or pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.problem.lb[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.lb[j]
                        if pop_new[i][self.ID_POS][j] > self.problem.ub[j]:
                            pop_new[i][self.ID_POS][j] = self.problem.ub[j]
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)

        for i in range(self.pop_size):
            if self.compare_agent(self.pop[i], pop_new[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                C_op = self.create_opposition_position(self.pop[i][self.ID_POS], self.g_best[self.ID_POS])
                fit_op = self.get_fitness_position(C_op)
                if self.compare_agent([C_op, fit_op], self.pop[i]):
                    self.pop[i] = [C_op, fit_op, 0.0]
                else:
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                    pos_new = pop_new[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * levy_step
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    self.pop[i] = [pos_new, fit_new, 0.0]
        self.pop = self._update_weight(pop_new)
