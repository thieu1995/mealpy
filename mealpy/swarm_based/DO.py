# !/usr/bin/env python
# Created by "Thieu" at 04:43, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import math
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseDO(Optimizer):
    """
    The original version of: Dragonfly Optimization (DO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-015-1920-1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.DO import BaseDO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = BaseDO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for
    solving single-objective, discrete, and multi-objective problems.
    Neural computing and applications, 27(4), pp.1053-1073.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """

        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = False

    def dragonfly_levy(self):
        beta = 3 / 2
        # Eq.(3.10)
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.problem.n_dims) * sigma
        v = np.random.randn(self.problem.n_dims)
        step = u / np.abs(v) ** (1 / beta)
        # Eq.(3.9)
        return 0.01 * step

    def initialization(self):
        # Initial radius of dragonflies' neighbouhoods
        self.radius = (self.problem.ub - self.problem.lb) / 10
        self.delta_max = (self.problem.ub - self.problem.lb) / 10

        # Initial population
        self.pop = self.create_population(self.pop_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        self.pop_delta = self.create_population(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        self.g_best, self.g_worst = best[0], worst[0]

        r = (self.problem.ub - self.problem.lb) / 4 + ((self.problem.ub - self.problem.lb) * (2 * (epoch + 1) / self.epoch))
        w = 0.9 - (epoch + 1) * ((0.9 - 0.4) / self.epoch)
        my_c = 0.1 - (epoch + 1) * ((0.1 - 0) / (self.epoch / 2))
        my_c = 0 if my_c < 0 else my_c

        s = 2 * np.random.rand() * my_c  # Seperation weight
        a = 2 * np.random.rand() * my_c  # Alignment weight
        c = 2 * np.random.rand() * my_c  # Cohesion weight
        f = 2 * np.random.rand()  # Food attraction weight
        e = my_c  # Enemy distraction weight

        for i in range(0, self.pop_size):
            pos_neighbours = []
            pos_neighbours_delta = []
            neighbours_num = 0
            # Find the neighbouring solutions
            for j in range(0, self.pop_size):
                dist = np.abs(self.pop[i][self.ID_POS] - self.pop[j][self.ID_POS])
                if np.all(dist <= r) and np.all(dist != 0):
                    neighbours_num += 1
                    pos_neighbours.append(self.pop[j][self.ID_POS])
                    pos_neighbours_delta.append(self.pop_delta[j][self.ID_POS])

            pos_neighbours = np.array(pos_neighbours)
            pos_neighbours_delta = np.array(pos_neighbours_delta)

            # Separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
            if neighbours_num > 1:
                S = np.sum(pos_neighbours, axis=0) - neighbours_num * self.pop[i][self.ID_POS]
                A = np.sum(pos_neighbours_delta, axis=0) / neighbours_num
                C_temp = np.sum(pos_neighbours, axis=0) / neighbours_num
            else:
                S = np.zeros(self.problem.n_dims)
                A = deepcopy(self.pop_delta[i][self.ID_POS])
                C_temp = deepcopy(self.pop[i][self.ID_POS])
            C = C_temp - self.pop[i][self.ID_POS]

            # Attraction to food: Eq 3.4
            dist_to_food = np.abs(self.pop[i][self.ID_POS] - self.g_best[self.ID_POS])
            if np.all(dist_to_food <= r):
                F = self.g_best[self.ID_POS] - self.pop[i][self.ID_POS]
            else:
                F = np.zeros(self.problem.n_dims)

            # Distraction from enemy: Eq 3.5
            dist_to_enemy = np.abs(self.pop[i][self.ID_POS] - self.g_worst[self.ID_POS])
            if np.all(dist_to_enemy <= r):
                enemy = self.g_worst[self.ID_POS] + self.pop[i][self.ID_POS]
            else:
                enemy = np.zeros(self.problem.n_dims)

            pos_new = deepcopy(self.pop[i][self.ID_POS]).astype(float)
            pos_delta_new = deepcopy(self.pop_delta[i][self.ID_POS]).astype(float)
            if np.any(dist_to_food > r):
                if neighbours_num > 1:
                    temp = w * self.pop_delta[i][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * A + \
                           np.random.uniform(0, 1, self.problem.n_dims) * C + np.random.uniform(0, 1, self.problem.n_dims) * S
                    temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                    pos_delta_new = deepcopy(temp)
                    pos_new += temp
                else:  # Eq. 3.8
                    pos_new += self.dragonfly_levy() * self.pop[i][self.ID_POS]
                    pos_delta_new = np.zeros(self.problem.n_dims)
            else:
                # Eq. 3.6
                temp = (a * A + c * C + s * S + f * F + e * enemy) + w * self.pop_delta[i][self.ID_POS]
                temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                pos_delta_new = temp
                pos_new += temp

            # Amend solution
            self.pop[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            self.pop_delta[i][self.ID_POS] = self.amend_position(pos_delta_new, self.problem.lb, self.problem.ub)

        self.pop = self.update_target_wrapper_population(self.pop)
        self.pop_delta = self.update_target_wrapper_population(self.pop_delta)
