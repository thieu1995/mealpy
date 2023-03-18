#!/usr/bin/env python
# Created by "Thieu" at 13:42, 06/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from math import gamma


class OriginalSeaHO(Optimizer):
    """
    The original version of: Sea-Horse Optimization (SeaHO)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03994-3
        2. https://www.mathworks.com/matlabcentral/fileexchange/115945-sea-horse-optimizer

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SeaHO import OriginalSeaHO
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
    >>> model = OriginalSeaHO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, S., Zhang, T., Ma, S., & Wang, M. (2022). Sea-horse optimizer: a novel nature-inspired
    meta-heuristic for global optimization problems. Applied Intelligence, 1-28.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.uu = 0.05
        self.vv = 0.05
        self.ll = 0.05

    def levy__(self, omega, size):
        num = gamma(1 + omega) * np.sin(np.pi * omega/2)
        den = gamma((1 + omega)/2) * omega* 2**((omega - 1) / 2)
        sigma_u = (num / den) ** (1 / omega)
        uu = np.random.normal(0, sigma_u, size)
        vv = np.random.normal(0, 1, size)
        zz = uu / (np.abs(vv) ** (1 / omega))
        return zz

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # The motor behavior of sea horses
        step_length = self.levy__(1.5, (self.pop_size, self.problem.n_dims))
        pop_new = []
        for idx in range(0, self.pop_size):
            beta = np.random.normal(0, 1, self.problem.n_dims)
            theta = 2 * np.pi * np.random.rand(self.problem.n_dims)
            row = self.uu * np.exp(theta * self.vv)
            xx, yy, zz = row * np.cos(theta), row * np.sin(theta), row * theta
            if np.random.normal(0, 1) > 0:      # Eq. 4
                pos_new = self.pop[idx][self.ID_POS] + step_length[idx] * ((self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * xx * yy * zz + self.g_best[self.ID_POS])
            else:                               # Eq. 7
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * self.ll * beta * (self.g_best[self.ID_POS] - beta * self.g_best[self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])

        # The predation behavior of sea horses
        pop_child = []
        alpha = (1 - (epoch+1)/self.epoch) ** (2 * (epoch+1) / self.epoch)
        for idx in range(0, self.pop_size):
            r1 = np.random.rand(self.problem.n_dims)
            if np.random.rand() >= 0.1:
                pos_new = alpha * (self.g_best[self.ID_POS] - np.random.rand(self.problem.n_dims) * pop_new[idx][self.ID_POS]) + \
                          (1 - alpha) * self.g_best[self.ID_POS]        # Eq. 10
            else:
                pos_new = (1 - alpha) * (pop_new[idx][self.ID_POS] - np.random.rand(self.problem.n_dims) * self.g_best[self.ID_POS])  + \
                        alpha * pop_new[idx][self.ID_POS]               # Eq. 11
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
        pop_child, _ = self.get_global_best_solution(pop_child)         # Sorted population

        # The reproductive behavior of sea horses
        dads = pop_child[:int(self.pop_size/2)]
        moms = pop_child[int(self.pop_size/2):]
        pop_offspring = []
        for kdx in range(0, int(self.pop_size/2)):
            r3 = np.random.rand()
            pos_new = r3 * dads[kdx][self.ID_POS] + (1 - r3) * moms[kdx][self.ID_POS]       # Eq. 13
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_offspring.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_offspring[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_offspring = self.update_target_wrapper_population(pop_offspring)

        # Sea horses selection
        self.pop = self.get_sorted_strim_population(pop_child + pop_offspring, self.pop_size)
