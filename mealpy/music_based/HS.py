#!/usr/bin/env python
# Created by "Thieu" at 17:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseHS(Optimizer):
    """
    The developed version: Harmony Search (HS)

    Links:
        1. https://doi.org/10.1177/003754970107600201

    Notes
    ~~~~~
    - Used the global best in the harmony memories
    - Removed all third for loops

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_r (float): [0.1, 0.5], Harmony Memory Consideration Rate), default = 0.15
        + pa_r (float): [0.3, 0.8], Pitch Adjustment Rate, default=0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.music_based.HS import BaseHS
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
    >>> c_r = 0.95
    >>> pa_r = 0.05
    >>> model = BaseHS(epoch, pop_size, c_r, pa_r)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, c_r=0.95, pa_r=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_r (float): Harmony Memory Consideration Rate, default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c_r = self.validator.check_float("c_r", c_r, (0, 1.0))
        self.pa_r = self.validator.check_float("pa_r", pa_r, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c_r", "pa_r"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        self.fw = 0.0001 * (self.problem.ub - self.problem.lb)  # Fret Width (Bandwidth)
        self.fw_damp = 0.9995  # Fret Width Damp Ratio
        self.dyn_fw = self.fw

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Create New Harmony Position
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)

            # Use Harmony Memory
            pos_new = np.where(np.random.random(self.problem.n_dims) < self.c_r, self.g_best[self.ID_POS], pos_new)
            # Pitch Adjustment
            x_new = pos_new + delta
            pos_new = np.where(np.random.random(self.problem.n_dims) < self.pa_r, x_new, pos_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp

        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)


class OriginalHS(BaseHS):
    """
    The original version of: Harmony Search (HS)

    Links:
        1. https://doi.org/10.1177/003754970107600201

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_r (float): [0.1, 0.5], Harmony Memory Consideration Rate), default = 0.15
        + pa_r (float): [0.3, 0.8], Pitch Adjustment Rate, default=0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.music_based.HS import OriginalHS
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
    >>> c_r = 0.95
    >>> pa_r = 0.05
    >>> model = OriginalHS(epoch, pop_size, c_r, pa_r)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Geem, Z.W., Kim, J.H. and Loganathan, G.V., 2001. A new heuristic
    optimization algorithm: harmony search. simulation, 76(2), pp.60-68.
    """

    def __init__(self, epoch=10000, pop_size=100, c_r=0.95, pa_r=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_r (float): Harmony Memory Consideration Rate), default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(epoch, pop_size, c_r, pa_r, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            for j in range(self.problem.n_dims):
                # Use Harmony Memory
                if np.random.uniform() <= self.c_r:
                    random_index = np.random.randint(0, self.pop_size)
                    pos_new[j] = self.pop[random_index][self.ID_POS][j]
                # Pitch Adjustment
                if np.random.uniform() <= self.pa_r:
                    delta = self.dyn_fw * np.random.normal(self.problem.lb, self.problem.ub)  # Gaussian(Normal)
                    pos_new[j] = pos_new[j] + delta[j]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp

        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
