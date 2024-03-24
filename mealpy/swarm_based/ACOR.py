#!/usr/bin/env python
# Created by "Thieu" at 14:14, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalACOR(Optimizer):
    """
    The original version of: Ant Colony Optimization Continuous (ACOR)

    Notes:
        + Use Gaussian Distribution (np.random.normal() function) instead of random number (np.random.rand())
        + Amend solution when they went out of space

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + sample_count (int): [2, 10000], Number of Newly Generated Samples, default = 25
        + intent_factor (float): [0.2, 1.0], Intensification Factor (Selection Pressure), (q in the paper), default = 0.5
        + zeta (float): [1, 2, 3], Deviation-Distance Ratio, default = 1.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ACOR
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
    >>> model = ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
    European journal of operational research, 185(3), pp.1155-1173.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, sample_count: int = 25,
                 intent_factor: float = 0.5, zeta: float = 1.0, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            sample_count: Number of Newly Generated Samples, default = 25
            intent_factor: Intensification Factor (Selection Pressure) (q in the paper), default = 0.5
            zeta: Deviation-Distance Ratio, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.sample_count = self.validator.check_int("sample_count", sample_count, [2, 10000])
        self.intent_factor = self.validator.check_float("intent_factor", intent_factor, (0, 1.0))
        self.zeta = self.validator.check_float("zeta", zeta, (0, 5))
        self.set_parameters(["epoch", "pop_size", "sample_count", "intent_factor", "zeta"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Selection Probabilities
        pop_rank = np.array([idx for idx in range(1, self.pop_size + 1)])
        qn = self.intent_factor * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)  # Normalize to find the probability.
        # Means and Standard Deviations
        matrix_pos = np.array([agent.solution for agent in self.pop])
        matrix_sigma = []
        for idx in range(0, self.pop_size):
            matrix_i = np.repeat(self.pop[idx].solution.reshape((1, -1)), self.pop_size, axis=0)
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (self.pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)

        # Generate Samples
        pop_new = []
        for idx in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)
            for jdx in range(0, self.problem.n_dims):
                rdx = self.get_index_roulette_wheel_selection(matrix_p)
                child[jdx] = self.pop[rdx].solution[jdx] + self.generator.normal() * matrix_sigma[rdx, jdx]  # (1)
            pos_new = self.correct_solution(child)      # (2)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
