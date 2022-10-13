#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFOA(Optimizer):
    """
    The original version of: Fruit-fly Optimization Algorithm (FOA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2011.07.001

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FOA import OriginalFOA
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
    >>> model = OriginalFOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Pan, W.T., 2012. A new fruit fly optimization algorithm: taking the financial distress model
    as an example. Knowledge-Based Systems, 26, pp.69-74.
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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def norm_consecutive_adjacent__(self, position=None):
        return np.array([np.linalg.norm([position[x], position[x + 1]]) for x in range(0, self.problem.n_dims - 1)] + \
                        [np.linalg.norm([position[-1], position[0]])])

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: a solution with format [position, target]
        """
        if pos is None:
            pos = self.generate_position(self.problem.lb, self.problem.ub)
        s = self.norm_consecutive_adjacent__(pos)
        pos = self.amend_position(s, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos)
        return [pos, target]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * np.random.normal(self.problem.lb, self.problem.ub)
            pos_new = self.norm_consecutive_adjacent__(pos_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)


class BaseFOA(OriginalFOA):
    """
    The developed version: Fruit-fly Optimization Algorithm (FOA)

    Notes
    ~~~~~
    + The fitness function (small function) is changed by taking the distance each 2 adjacent dimensions
    + Update the position if only new generated solution is better
    + The updated position is created by norm distance * gaussian random number

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FOA import BaseFOA
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
    >>> model = BaseFOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.normal(self.problem.lb, self.problem.ub)
            pos_new = c * np.random.rand() * self.norm_consecutive_adjacent__(pos_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)


class WhaleFOA(OriginalFOA):
    """
    The original version of: Whale Fruit-fly Optimization Algorithm (WFOA)

    Links:
        1. https://doi.org/10.1016/j.eswa.2020.113502

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FOA import WhaleFOA
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
    >>> model = WhaleFOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Fan, Y., Wang, P., Heidari, A.A., Wang, M., Zhao, X., Chen, H. and Li, C., 2020. Boosted hunting-based
    fruit fly optimization and advances in real-world problems. Expert Systems with Applications, 159, p.113502.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0

        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.rand() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # select random 1 position in pop
                    x_rand = self.pop[np.random.randint(self.pop_size)]
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = (x_rand[self.ID_POS] - A * D)
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + self.g_best[self.ID_POS]
            smell = self.norm_consecutive_adjacent__(pos_new)
            pos_new = self.amend_position(smell, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)
