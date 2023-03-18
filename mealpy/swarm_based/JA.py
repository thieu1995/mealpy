#!/usr/bin/env python
# Created by "Thieu" at 16:30, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseJA(Optimizer):
    """
    The developed version: Jaya Algorithm (JA)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.JA import BaseJA
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
    >>> model = BaseJA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R., 2016. Jaya: A simple and new optimization algorithm for solving constrained and
    unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), pp.19-34.
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
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) + \
                      np.random.normal() * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalJA(BaseJA):
    """
    The original version of: Jaya Algorithm (JA)

    Links:
        1. https://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.JA import OriginalJA
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
    >>> model = OriginalJA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R., 2016. Jaya: A simple and new optimization algorithm for solving constrained and
    unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), pp.19-34.
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
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                      (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) - \
                      np.random.uniform(0, 1, self.problem.n_dims) * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop_new


class LevyJA(BaseJA):
    """
    The original version of: Levy-flight Jaya Algorithm (LJA)

    Links:
        1. https://doi.org/10.1016/j.eswa.2020.113902

    Notes
    ~~~~~
    + All third loops in this version also are removed
    + The beta value of Levy-flight equal to 1.8 as the best value in the paper.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.JA import LevyJA
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
    >>> model = LevyJA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Iacca, G., dos Santos Junior, V.C. and de Melo, V.V., 2021. An improved Jaya optimization
    algorithm with Lévy flight. Expert Systems with Applications, 165, p.113902.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            L1 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
            L2 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
            pos_new = self.pop[idx][self.ID_POS] + np.abs(L1) * (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) - \
                      np.abs(L2) * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
