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

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 5000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 50.
    attract_dim_rate : float
        Number of dims will be changed in attraction phase under ratio format, in range (0.0, 1.0). Default is 0.2.

    References
    ~~~~~~~~~~
    1. Liu, S., Xiang, Y., Guo, X., Zhao, F., Zhao, A., & Wu, W. (2025).
       Market Game Optimization Algorithm: A Metaheuristic Inspired by Symmetric Competitive Behavior of Merchants and Consumers.
       Symmetry, 17(12), 2118. https://doi.org/10.3390/sym17122118

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
    >>> model = MGOA.OriginalMGOA(epoch=1000, pop_size=50, attract_dim_rate=0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 5000, pop_size: int = 50, attract_dim_rate=0.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 5000
            pop_size (int): number of population size, default = 50
            attract_dim_rate (float): number of dims will be changed in attraction phase under ratio format, default = 0.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.attract_dim_rate = self.validator.check_float("attract_dim_rate", attract_dim_rate, (0, 1))
        self.set_parameters(["epoch", "pop_size", "attract_dim_rate"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        n = 2.0
        alpha = 1.0
        ratio = epoch / self.epoch
        S = np.pi * np.sin(2 * np.pi * (ratio * n)) * np.exp(-alpha * ratio)
        phase = "attraction" if self.generator.random() < 0.5 else "collaboration"
        n_attract_dims = int(np.ceil(self.attract_dim_rate * self.problem.n_dims))

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            if phase == "attraction":
                selected_dims = self.generator.choice(self.problem.n_dims, n_attract_dims, replace=False)
                pos_new[selected_dims] = pos_new[selected_dims] - S * (self.g_best.solution[selected_dims] - pos_new[selected_dims])
            else:
                rA, rB = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=2)
                dm1 = self.pop[rA].solution - self.pop[idx].solution
                dm2 = self.pop[rB].solution - self.pop[idx].solution
                pos_new = self.pop[idx].solution + self.generator.random() * dm1 + self.generator.random() * dm2

            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
