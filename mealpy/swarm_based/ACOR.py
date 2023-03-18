#!/usr/bin/env python
# Created by "Thieu" at 14:14, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalACOR(Optimizer):
    """
    The original version of: Ant Colony Optimization Continuous (ACOR)

    Notes
    ~~~~~
    + Use Gaussian Distribution instead of random number (np.random.normal() function)
    + Amend solution when they went out of space

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + sample_count (int): [2, 10000], Number of Newly Generated Samples, default = 25
        + intent_factor (float): [0.2, 1.0], Intensification Factor (Selection Pressure), (q in the paper), default = 0.5
        + zeta (float): [1, 2, 3], Deviation-Distance Ratio, default = 1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ACOR import OriginalACOR
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
    >>> sample_count = 25
    >>> intent_factor = 0.5
    >>> zeta = 1.0
    >>> model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
    European journal of operational research, 185(3), pp.1155-1173.
    """

    def __init__(self, epoch=10000, pop_size=100, sample_count=25, intent_factor=0.5, zeta=1.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sample_count (int): Number of Newly Generated Samples, default = 25
            intent_factor (float): Intensification Factor (Selection Pressure) (q in the paper), default = 0.5
            zeta (float): Deviation-Distance Ratio, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sample_count = self.validator.check_int("sample_count", sample_count, [2, 10000])
        self.intent_factor = self.validator.check_float("intent_factor", intent_factor, (0, 1.0))
        self.zeta = self.validator.check_float("zeta", zeta, (0, 5))
        self.set_parameters(["epoch", "pop_size", "sample_count", "intent_factor", "zeta"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Selection Probabilities
        pop_rank = np.array([i for i in range(1, self.pop_size + 1)])
        qn = self.intent_factor * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)  # Normalize to find the probability.

        # Means and Standard Deviations
        matrix_pos = np.array([solution[self.ID_POS] for solution in self.pop])
        matrix_sigma = []
        for i in range(0, self.pop_size):
            matrix_i = np.repeat(self.pop[i][self.ID_POS].reshape((1, -1)), self.pop_size, axis=0)
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (self.pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)

        # Generate Samples
        pop_new = []
        for i in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)
            for j in range(0, self.problem.n_dims):
                idx = self.get_index_roulette_wheel_selection(matrix_p)
                child[j] = self.pop[idx][self.ID_POS][j] + np.random.normal() * matrix_sigma[idx, j]  # (1)
            pos_new = self.amend_position(child, self.problem.lb, self.problem.ub)  # (2)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
