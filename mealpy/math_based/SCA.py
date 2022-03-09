# !/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSCA(Optimizer):
    """
    My changed version of: Sine Cosine Algorithm (SCA)

    Notes
    ~~~~~
    + The flow and few equations is changed
    + Removed third loop for faster computational time

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SCA import BaseSCA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = BaseSCA(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a - (epoch + 1) * (a / self.epoch)
            # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
            r2 = 2 * np.pi * np.random.uniform(0, 1, self.problem.n_dims)
            r3 = 2 * np.random.uniform(0, 1, self.problem.n_dims)
            # Eq. 3.3, 3.1 and 3.2
            pos_new1 = self.pop[idx][self.ID_POS] + r1 * np.sin(r2) * abs(r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new2 = self.pop[idx][self.ID_POS] + r1 * np.cos(r2) * abs(r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, pos_new1, pos_new2)
            # Check the bound
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalSCA(BaseSCA):
    """
    The original version of: Sine Cosine Algorithm (SCA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2015.12.022
        2. https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SCA import OriginalSCA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalSCA(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. SCA: a sine cosine algorithm for solving optimization problems. Knowledge-based systems, 96, pp.120-133.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def amend_position(self, position=None):
        """
        If solution out of bound at dimension x, then it will re-arrange to random location in the range of domain

        Args:
            position: vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.where(np.logical_and(self.problem.lb <= position, position <= self.problem.ub),
                        position, np.random.uniform(self.problem.lb, self.problem.ub))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a - (epoch + 1) * (a / self.epoch)
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(self.problem.n_dims):  # j-th dimension
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * np.pi * np.random.uniform()
                r3 = 2 * np.random.uniform()
                r4 = np.random.uniform()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    pos_new[j] = pos_new[j] + r1 * np.sin(r2) * abs(r3 * self.g_best[self.ID_POS][j] - pos_new[j])
                else:
                    pos_new[j] = pos_new[j] + r1 * np.cos(r2) * abs(r3 * self.g_best[self.ID_POS][j] - pos_new[j])
            # Check the bound
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
