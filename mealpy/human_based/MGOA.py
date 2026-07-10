#!/usr/bin/env python
# Created by "Thieu" at 10:30, 02/01/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMGOA(Optimizer):
    """
    The original version of: Market Game Optimization Algorithm (MGOA)

    Links:
        1. https://doi.org/10.3390/sym17122118

    Notes:
        + Paper parameters: A = pi, n = 2, alpha = 1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MGOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = MGOA.OriginalMGOA(epoch=5000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Symmetry 2025, 17(12), 2118.
    """

    def __init__(self, epoch: int = 5000, pop_size: int = 50, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 5000
            pop_size (int): number of population size, default = 50
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        A = np.pi
        n = 2.0
        alpha = 1.0
        ratio = epoch / self.epoch
        S = A * np.sin(2 * np.pi * (ratio * n)) * np.exp(-alpha * ratio)

        phase = "attraction" if self.generator.random() < 0.5 else "collaboration"
        k = 5

        for idx in range(0, self.pop_size):
            pos = self.pop[idx].solution
            pos_new = pos.copy()

            if phase == "attraction":
                n_select = min(self.problem.n_dims, k)
                selected_dims = self.generator.choice(self.problem.n_dims, n_select, replace=False)
                pos_new[selected_dims] = pos[selected_dims] - S * (self.g_best.solution[selected_dims] - pos[selected_dims])
            else:
                candidates = np.arange(self.pop_size)
                candidates = candidates[candidates != idx]
                rA, rB = self.generator.choice(candidates, 2, replace=False)
                dm1 = self.pop[rA].solution - pos
                dm2 = self.pop[rB].solution - pos
                r1 = self.generator.random()
                r2 = self.generator.random()
                pos_new = pos + r1 * dm1 + r2 * dm2

            pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
