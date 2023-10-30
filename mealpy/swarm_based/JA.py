#!/usr/bin/env python
# Created by "Thieu" at 16:30, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevJA(Optimizer):
    """
    The developed version: Jaya Algorithm (JA)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, JA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = JA.DevJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R., 2016. Jaya: A simple and new optimization algorithm for solving constrained and
    unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), pp.19-34.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, (g_best, ), (g_worst, ) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (g_best.solution - np.abs(self.pop[idx].solution)) + \
                      self.generator.normal() * (g_worst.solution - np.abs(self.pop[idx].solution))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalJA(DevJA):
    """
    The original version of: Jaya Algorithm (JA)

    Links:
        1. https://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, JA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = JA.OriginalJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R., 2016. Jaya: A simple and new optimization algorithm for solving constrained and
    unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), pp.19-34.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
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
        _, (g_best, ), (g_worst, ) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.generator.uniform(0, 1, self.problem.n_dims) * \
                      (g_best.solution - np.abs(self.pop[idx].solution)) - \
                      self.generator.uniform(0, 1, self.problem.n_dims) * (g_worst.solution - np.abs(self.pop[idx].solution))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new


class LevyJA(DevJA):
    """
    The original version of: Levy-flight Jaya Algorithm (LJA)

    Notes
        + All third loops in this version also are removed
        + The beta value of Levy-flight equal to 1.8 as the best value in the paper.
        + https://doi.org/10.1016/j.eswa.2020.113902

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, JA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = JA.LevyJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Iacca, G., dos Santos Junior, V.C. and de Melo, V.V., 2021. An improved Jaya optimization
    algorithm with Lévy flight. Expert Systems with Applications, 165, p.113902.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
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
        _, (g_best, ), (g_worst, ) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            L1 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
            L2 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
            pos_new = self.pop[idx].solution + np.abs(L1) * (g_best.solution - np.abs(self.pop[idx].solution)) - \
                      np.abs(L2) * (g_worst.solution - np.abs(self.pop[idx].solution))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
