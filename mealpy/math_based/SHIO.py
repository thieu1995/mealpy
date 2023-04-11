#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSHIO(Optimizer):
    """
    The original version of: Success History Intelligent Optimizer (SHIO)

    Links:
        1. https://link.springer.com/article/10.1007/s11227-021-04093-9
        2. https://www.mathworks.com/matlabcentral/fileexchange/122157-success-history-intelligent-optimizer-shio

    Notes:
        1. The algorithm is designed with simplicity and ease of implementation in mind, utilizing basic operators.
        2. This algorithm has several limitations and weak when dealing with several problems
        3. The algorithm's convergence is slow. The Matlab code has many errors and unnecessary things.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SHIO import OriginalSHIO
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
    >>> model = OriginalSHIO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of Supercomputing, 1-42.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, (b1, b2, b3), _ = self.get_special_solutions(self.pop, best=3, worst=1)
        a = 1.5
        pop_new = []
        for idx in range(0, self.pop_size):
            a = a - 0.04
            x1 = b1[self.ID_POS] + (a*2*np.random.rand(self.problem.n_dims) - a)*np.abs(np.random.rand(self.problem.n_dims) * b1[self.ID_POS] - self.pop[idx][self.ID_POS])
            x2 = b2[self.ID_POS] + (a*2*np.random.rand(self.problem.n_dims) - a)*np.abs(np.random.rand(self.problem.n_dims) * b1[self.ID_POS] - self.pop[idx][self.ID_POS])
            x3 = b3[self.ID_POS] + (a*2*np.random.rand(self.problem.n_dims) - a)*np.abs(np.random.rand(self.problem.n_dims) * b1[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (x1 + x2 + x3) / 3
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop_new
