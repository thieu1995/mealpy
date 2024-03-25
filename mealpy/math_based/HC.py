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

    Notes:
        + The number of neighbour solutions are equal to user defined
        + The step size to calculate neighbour group is randomized
        + HC is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, 1000], fixed parameter, sensitive exploitation parameter, Default: 50

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HC
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = HC.OriginalHC(epoch=1000, pop_size=50, neighbour_size = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mitchell, M., Holland, J. and Forrest, S., 1993. When will a genetic algorithm
    outperform hill climbing. Advances in neural information processing systems, 6.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 2, neighbour_size: int = 50, **kwargs: object) -> None:
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
        step_size = np.exp(-2 * epoch / self.epoch)
        pop_neighbours = []
        for idx in range(0, self.neighbour_size):
            pos_new = self.g_best.solution + self.generator.uniform(self.problem.lb, self.problem.ub) * step_size
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_neighbours.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_neighbours[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_neighbours)


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
    >>> from mealpy import FloatVar, HC
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = HC.SwarmHC(epoch=1000, pop_size=50, neighbour_size = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
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
            for jdx in range(0, self.neighbour_size):
                pos_new = self.pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * ss[idx]
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_neighbours.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_neighbours[-1].target = self.get_target(pos_new)
            pop_neighbours = self.update_target_for_population(pop_neighbours)
            best_local = self.get_best_agent(pop_neighbours, self.problem.minmax)
            pop.append(best_local)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self.get_better_agent(best_local, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
