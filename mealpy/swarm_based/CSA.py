#!/usr/bin/env python
# Created by "Thieu" at 18:37, 28/05/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCSA(Optimizer):
    """
    The original version of: Cuckoo Search Algorithm (CSA)

    Links:
        1. https://doi.org/10.1109/NABIC.2009.5393690

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_a (float): [0.1, 0.7], probability a, default=0.3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CSA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = CSA.OriginalCSA(epoch=1000, pop_size=50, p_a = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S. and Deb, S., 2009, December. Cuckoo search via Lévy flights. In 2009 World
    congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). Ieee.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_a: float = 0.3, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_a (float): probability a, default=0.3
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_a = self.validator.check_float("p_a", p_a, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_a"])
        self.n_cut = int(self.p_a * self.pop_size)
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            ## Generate levy-flight solution
            levy_step = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = self.pop[idx].solution + 1.0 / np.sqrt(epoch) * np.sign(self.generator.random() - 0.5) * \
                      levy_step * (self.pop[idx].solution - self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        ## Abandoned some worst nests
        pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)
        pop_new = []
        for idx in range(0, self.n_cut):
            pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = pop[:(self.pop_size - self.n_cut)] + pop_new
