#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHC(Optimizer):
    """
    The original version of: Hill Climbing (HC)

    Notes
    ~~~~~
    + The number of neighbour solutions are equal to user defined
    + The step size to calculate neighbour group is randomized (range from -1 to 1)
    + HC is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, 1000], fixed parameter, sensitive exploitation parameter, Default: 50

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.HC import OriginalHC
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
    >>> pop_size = 2
    >>> neighbour_size = 50
    >>> model = OriginalHC(epoch, pop_size, neighbour_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mitchell, M., Holland, J. and Forrest, S., 1993. When will a genetic algorithm
    outperform hill climbing. Advances in neural information processing systems, 6.
    """

    def __init__(self, epoch=10000, pop_size=2, neighbour_size=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 2
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 50
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.neighbour_size = self.validator.check_int("neighbour_size", neighbour_size, [2, 1000])
        self.set_parameters(["epoch", "pop_size", "neighbour_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        step_size = 1 - 2 * (1 - (epoch + 1) / self.epoch)
        pop_neighbours = []
        for idx in range(0, self.neighbour_size):
            pos_new = self.g_best[self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * step_size
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_neighbours.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_neighbours[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_neighbours = self.update_target_wrapper_population(pop_neighbours)
        self.pop = pop_neighbours


class SwarmHC(Optimizer):
    """
    The developed version: Swarm-based Hill Climbing (S-HC)

    Notes
    ~~~~~
    + Based on swarm-of people are trying to climb on the mountain idea
    + The number of neighbour solutions are equal to population size
    + The step size to calculate neighbour is randomized and based on rank of solution.
        + The guys near on top of mountain will move slower than the guys on bottom of mountain.
        + Imagination: exploration when far from global best, and exploitation when near global best
    + Who on top of mountain first will be the winner. (global optimal)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, pop_size/2], fixed parameter, sensitive exploitation parameter, Default: 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.HC import SwarmHC
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
    >>> neighbour_size = 10
    >>> model = SwarmHC(epoch, pop_size, neighbour_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, neighbour_size=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.neighbour_size = self.validator.check_int("neighbour_size", neighbour_size, [2, int(self.pop_size/2)])
        self.set_parameters(["epoch", "pop_size", "neighbour_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ranks = np.array(list(range(1, self.pop_size + 1)))
        ranks = ranks / np.sum(ranks)
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * (epoch + 1) / self.epoch)
        ss = step_size * ranks
        pop = []
        for idx in range(0, self.pop_size):
            pop_neighbours = []
            for j in range(0, self.neighbour_size):
                pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * ss[idx]
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_neighbours.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_neighbours[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_neighbours = self.update_target_wrapper_population(pop_neighbours)
            _, agent = self.get_global_best_solution(pop_neighbours)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self.get_better_solution(agent, self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.greedy_selection_population(self.pop, pop)
