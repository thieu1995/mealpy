#!/usr/bin/env python
# Created by "Eren Kayacilar" at 20:50, 09/12/2025 ----------%
#       Email: serenkay01@gmail.com                          %
#       Github: https://github.com/ErenKayacilar             %
# -----------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalDBO(Optimizer):
    """
    The original version of: Dung Beetle Optimizer (DBO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    kk : float
        Deflection coefficient in rolling behavior, in range [0.0, 2.0]. Default is 0.1.
    bb : float
        Attraction toward worst position, in range [0.0, 1.0]. Default is 0.3.
    ss : float
        Attraction factor toward local best position, in range [0.0, 1.0]. Default is 0.3.

    Links
    -----
    1. https://doi.org/10.1007/s11227-022-04959-6
    2. https://github.com/Lancephil/Dung-Beetle-Optimizer

    References
    ~~~~~~~~~~
    1. Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic
       algorithm for global optimization. The Journal of Supercomputing, 79, 7305–7336.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DBO
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
    >>> model = DBO.OriginalDBO(epoch=1000, pop_size=50, kk=0.1, bb=0.5, ss=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, kk: float = 0.1, bb: float = 0.3, ss: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.kk = self.validator.check_float("kk", kk, [0.0, 2.0])
        self.bb = self.validator.check_float("bb", bb, [0.0, 1.0])
        self.ss = self.validator.check_float("ss", ss, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "kk", "bb", "ss"])

        self.sort_flag = True
        # Previous positions x(t−1), used in rolling behavior
        self._prev_positions = None

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        # Initialize previous positions x(t−1) on the first call
        if self._prev_positions is None:
            self._prev_positions = np.array([agent.solution.copy() for agent in self.pop])

    def evolve(self, epoch: int):
        """
        The main operations (equations) of the algorithm.
        Inherited from Optimizer class.

        Args:
            epoch (int): The current iteration.
        """
        # Local best and local worst
        _, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        x_best = best[0].solution
        x_worst = worst[0].solution

        # Split population into four behavioral groups: ball-rolling, breeding, foraging, and stealing dung beetles.
        n_roll = self.pop_size // 4
        n_breed = self.pop_size // 4 + n_roll
        n_forage = self.pop_size // 4 + n_breed

        pop_new = []
        ## R for area
        RR = 1.0 - epoch / self.epoch
        ## Bound for breeding and spawning area (Eq. 3)
        Lb_star = np.maximum(x_best * (1 - RR), self.problem.lb)
        Ub_star = np.minimum(x_best * (1 + RR), self.problem.ub)
        ## Bound for foraging area (Eq. 5)
        Lb_b = np.maximum(self.g_best.solution * (1 - RR), self.problem.lb)
        Ub_b = np.minimum(self.g_best.solution * (1 + RR), self.problem.ub)

        for idx in range(self.pop_size):
            x_curr = self.pop[idx].solution
            x_new = x_curr.copy()  # Default to current position if no update occurs

            if idx <= n_roll:
                # Ball-rolling dung beetles
                delta = self.generator.random()
                if delta < 0.9:     # Eq. 1
                    alpha = 1 if self.generator.random() > 0.5 else -1
                    delta_x = np.abs(x_curr - x_worst)
                    x_new = x_curr + alpha * self.kk * self._prev_positions[idx] + self.bb * delta_x
                else:       # Eq. 2
                    theta = self.generator.random() * np.pi
                    if np.abs(theta) > 1e-6 and np.abs(theta - np.pi / 2) > 1e-6 and np.abs(theta - np.pi) > 1e-6:
                        x_new = x_curr + np.tan(theta) * np.abs(x_curr - self._prev_positions[idx])
            elif idx <= n_breed:
                # Breeding dung beetles. Eq. 4
                b1 = self.generator.random(self.problem.n_dims)
                b2 = self.generator.random(self.problem.n_dims)
                x_new = x_best + b1 * (x_curr - Lb_star) + b2 * (x_curr - Ub_star)
                x_new = np.maximum(x_new, Lb_star)
                x_new = np.minimum(x_new, Ub_star)
            elif idx <= n_forage:
                # Small dung beetle (Eq 6)
                C1 = self.generator.normal()
                C2 = self.generator.random(self.problem.n_dims)
                x_new = x_curr + C1 * (x_curr - Lb_b) + C2 * (x_curr - Ub_b)
            else:
                # Stealing dung beetles (Eq. 7)
                g_vec = self.generator.normal(size=self.problem.n_dims)
                x_new = self.g_best.solution + self.ss * g_vec * (np.abs(x_curr - x_best) + np.abs(x_curr - self.g_best.solution))

            ## Set up bound
            if idx <= n_roll or idx > n_breed:
                x_new = np.maximum(x_new, self.problem.lb)
                x_new = np.minimum(x_new, self.problem.ub)
            ## Correct solution based on problem
            x_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(x_new)
            pop_new.append(agent)
            self._prev_positions[idx] = x_curr.copy()  # Update previous position for next iteration
            # In sequential modes: evaluate and greedy-update immediately
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(x_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        # In parallel modes: evaluate batch and greedy-select
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
