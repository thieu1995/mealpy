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
    >>> from mealpy import FloatVar, SHIO
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
    >>> model = SHIO.OriginalSHIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of Supercomputing, 1-42.
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
        _, (b1, b2, b3), _ = self.get_special_agents(self.pop, n_best=3, n_worst=1, minmax=self.problem.minmax)
        a = 1.5
        pop_new = []
        for idx in range(0, self.pop_size):
            a = a - 0.04
            x1 = b1.solution + (a*2*self.generator.random(self.problem.n_dims) - a)*np.abs(self.generator.random(self.problem.n_dims) * b1.solution - self.pop[idx].solution)
            x2 = b2.solution + (a*2*self.generator.random(self.problem.n_dims) - a)*np.abs(self.generator.random(self.problem.n_dims) * b2.solution - self.pop[idx].solution)
            x3 = b3.solution + (a*2*self.generator.random(self.problem.n_dims) - a)*np.abs(self.generator.random(self.problem.n_dims) * b3.solution - self.pop[idx].solution)
            pos_new = (x1 + x2 + x3) / 3
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new
