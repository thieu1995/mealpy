#!/usr/bin/env python
# Created by "Thieu" at 09:56, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalAOA(Optimizer):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2020.113609

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): [3, 8], fixed parameter, sensitive exploitation parameter, Default: 5,
        + miu (float): [0.3, 1.0], fixed parameter , control parameter to adjust the search process, Default: 0.5,
        + moa_min (float): [0.1, 0.4], range min of Math Optimizer Accelerated, Default: 0.2,
        + moa_max (float): [0.5, 1.0], range max of Math Optimizer Accelerated, Default: 0.9,

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AOA
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
    >>> model = AOA.OriginalAOA(epoch=1000, pop_size=50, alpha = 5, miu = 0.5, moa_min = 0.2, moa_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M. and Gandomi, A.H., 2021. The arithmetic
    optimization algorithm. Computer methods in applied mechanics and engineering, 376, p.113609.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: float = 5,
                 miu: float = 0.5, moa_min: float = 0.2, moa_max: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5,
            miu (float): fixed parameter, control parameter to adjust the search process, Default: 0.5,
            moa_min (float): range min of Math Optimizer Accelerated, Default: 0.2,
            moa_max (float): range max of Math Optimizer Accelerated, Default: 0.9,
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.alpha = self.validator.check_int("alpha", alpha, [2, 10])
        self.miu = self.validator.check_float("miu", miu, [0.1, 2.0])
        self.moa_min = self.validator.check_float("moa_min", moa_min, (0, 0.41))
        self.moa_max = self.validator.check_float("moa_max", moa_max, (0.41, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "miu", "moa_min", "moa_max"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        moa = self.moa_min + epoch * ((self.moa_max - self.moa_min) / self.epoch)  # Eq. 2
        mop = 1 - ((epoch+1) ** (1.0 / self.alpha)) / (self.epoch ** (1.0 / self.alpha))  # Eq. 4
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                r1, r2, r3 = self.generator.random(3)
                if r1 > moa:  # Exploration phase
                    if r2 < 0.5:
                        pos_new[j] = self.g_best.solution[j] / (mop + self.EPSILON) * \
                                     ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
                    else:
                        pos_new[j] = self.g_best.solution[j] * mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
                else:  # Exploitation phase
                    if r3 < 0.5:
                        pos_new[j] = self.g_best.solution[j] - mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
                    else:
                        pos_new[j] = self.g_best.solution[j] + mop * ((self.problem.ub[j] - self.problem.lb[j]) * self.miu + self.problem.lb[j])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
