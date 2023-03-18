#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from copy import deepcopy


class OriginalESOA(Optimizer):
    """
    The original version of: Egret Swarm Optimization Algorithm (ESOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/115595-egret-swarm-optimization-algorithm-esoa
        2. https://www.mdpi.com/2313-7673/7/4/144

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ESOA import OriginalESOA
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
    >>> model = OriginalESOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, Z., Francis, A., Li, S., Liao, B., Xiao, D., Ha, T. T., ... & Cao, X. (2022). Egret Swarm Optimization Algorithm:
    An Evolutionary Computation Approach for Model Free Optimization. Biomimetics, 7(4), 144.
    """

    ID_WEI = 2
    ID_LOC_X = 3
    ID_LOC_Y = 4
    ID_G = 5
    ID_M = 6
    ID_V = 7

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.support_parallel_modes = False
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        weights = np.random.uniform(-1, 1, len(lb))
        g = (np.sum(weights * position) - target[self.ID_FIT]) * position
        m = np.zeros(self.problem.n_dims)
        v = np.zeros(self.problem.n_dims)
        return [position, target, weights, position.copy(), deepcopy(target), g, m, v]

    def initialize_variables(self):
        self.beta1 = 0.9
        self.beta2 = 0.99

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        hop = self.problem.ub - self.problem.lb
        for idx in range(0, self.pop_size):
            # Individual Direction
            p_d = self.pop[idx][self.ID_LOC_X] - self.pop[idx][self.ID_POS]
            p_d = p_d * (self.pop[idx][self.ID_LOC_Y][self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT])
            p_d = p_d / ((np.sum(p_d) + self.EPSILON)**2)
            d_p = p_d + self.pop[idx][self.ID_G]

            # Group Direction
            c_d = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
            c_d = c_d * (self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT])
            c_d = c_d / ((np.sum(c_d) + self.EPSILON)**2)
            d_g = c_d + self.g_best[self.ID_G]

            # Gradient Estimation
            r1 = np.random.random(self.problem.n_dims)
            r2 = np.random.random(self.problem.n_dims)
            g = (1 - r1 - r2) * self.pop[idx][self.ID_G] + r1 * d_p + r2 * d_g
            g = g / (np.sum(g) + self.EPSILON)

            self.pop[idx][self.ID_M] = self.beta1 * self.pop[idx][self.ID_M] + (1 - self.beta1) * g
            self.pop[idx][self.ID_V] = self.beta2 * self.pop[idx][self.ID_V] + (1 - self.beta2) * g**2
            self.pop[idx][self.ID_WEI] -= self.pop[idx][self.ID_M] / (np.sqrt(self.pop[idx][self.ID_V]) + self.EPSILON)

            # Advice Forward
            x_0 = self.pop[idx][self.ID_POS] + np.exp(-1.0 / (0.1 * self.epoch)) * 0.1 * hop * g
            x_0 = self.amend_position(x_0, self.problem.lb, self.problem.ub)
            y_0 = self.get_target_wrapper(x_0)

            # Random Search
            r3 = np.random.uniform(-np.pi/2, np.pi/2, self.problem.n_dims)
            x_n = self.pop[idx][self.ID_POS] + np.tan(r3) * hop / (1 + epoch) * 0.5
            x_n = self.amend_position(x_n, self.problem.lb, self.problem.ub)
            y_n = self.get_target_wrapper(x_n)

            # Encircling Mechanism
            d = self.pop[idx][self.ID_LOC_X] - self.pop[idx][self.ID_POS]
            d_g = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
            r1 = np.random.random(self.problem.n_dims)
            r2 = np.random.random(self.problem.n_dims)
            x_m = (1 - r1 - r2) * self.pop[idx][self.ID_POS] + r1 * d + r2 * d_g
            x_m = self.amend_position(x_m, self.problem.lb, self.problem.ub)
            y_m = self.get_target_wrapper(x_m)

            # Discriminant Condition
            y_list_compare = [y_0[self.ID_FIT], y_n[self.ID_FIT], y_m[self.ID_FIT]]
            y_list = [y_0, y_n, y_m]
            x_list = [x_0, x_n, x_m]
            if self.problem.minmax == "min":
                id_best = np.argmin(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]
            else:
                id_best = np.argmax(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]

            if self.compare_agent([x_best, y_best], self.pop[idx]):
                self.pop[idx][self.ID_POS] = x_best
                self.pop[idx][self.ID_TAR] = y_best
                if self.compare_agent([x_best, y_best], [self.pop[idx][self.ID_LOC_X], self.pop[idx][self.ID_LOC_Y]]):
                    self.pop[idx][self.ID_LOC_X] = x_best
                    self.pop[idx][self.ID_LOC_Y] = y_best
                    self.pop[idx][self.ID_G] = (np.sum(self.pop[idx][self.ID_WEI] * self.pop[idx][self.ID_POS]) - self.pop[idx][self.ID_TAR][self.ID_FIT]) * self.pop[idx][self.ID_POS]
            else:
                if np.random.rand() < 0.3:
                    self.pop[idx][self.ID_POS] = x_best
                    self.pop[idx][self.ID_TAR] = y_best
