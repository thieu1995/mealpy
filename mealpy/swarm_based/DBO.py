#!/usr/bin/env python
# Created by "Eren Kayacilar" at 20:50, 09/12/2025 ----------%
#       Email: serenkay01@gmail.com                          %
#       Github: https://github.com/ErenKayacilar             %
# -----------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalDBO(Optimizer):
    """
    The original version of: Dung Beetle Optimizer (DBO)

    Links:
        1. https://doi.org/10.1007/s11227-022-04959-6
        2. https://github.com/Lancephil/Dung-Beetle-Optimizer

    Hyper-parameters should be fine-tuned in approximate ranges to obtain
    faster convergence toward the global optimum:
        + alpha (float): [-2.0, 2.0], direction / step influence, default = 1.0
        + k (float): (0.0, 0.2], deflection coefficient in rolling behavior,
                     default = 0.1
        + b_const (float): (0.0, 1.0], attraction toward best / worst positions,
                           default = 0.5

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
    >>> model = DBO.OriginalDBO(epoch=1000, pop_size=50,
    >>>                          alpha=1.0, k=0.1, b_const=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, "
    >>>       f"Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic
        algorithm for global optimization. The Journal of Supercomputing,
        79, 7305–7336.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 1.0,
        k: float = 0.1,
        b_const: float = 0.5,
        **kwargs: object,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): population size, default = 100
            alpha (float): direction / step influence, default = 1.0
            k (float): deflection coefficient in rolling behavior, default = 0.1
            b_const (float): attraction coefficient, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])

        self.alpha = self.validator.check_float("alpha", alpha, (-2.0, 2.0))
        self.k = self.validator.check_float("k", k, (0.0, 0.2))
        self.b_const = self.validator.check_float("b_const", b_const, (0.0, 1.0))

        self.set_parameters(["epoch", "pop_size", "alpha", "k", "b_const"])

        # Similar to some other swarm-based optimizers
        self.sort_flag = True
        self.is_parallelizable = False

        # Previous positions x(t−1), used in rolling behavior
        self._prev_positions = None

    def initialization(self):
        """
        Initialization step of the algorithm.
        This method is automatically called inside solve().
        """
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        # Initialize previous positions x(t−1) on the first call
        if self._prev_positions is None:
            self._prev_positions = np.array(
                [agent.solution.copy() for agent in self.pop]
            )

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate an empty agent (without target).

        DBO does not require any additional attributes per agent,
        therefore only the solution vector is stored.
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return Agent(solution=solution)

    def evolve(self, epoch: int):
        """
        The main operations (equations) of the algorithm.
        Inherited from Optimizer class.

        Args:
            epoch (int): The current iteration.
        """
        # Make sure previous positions array is aligned with the population
        if self._prev_positions is None or len(self._prev_positions) != self.pop_size:
            self._prev_positions = np.array(
                [agent.solution.copy() for agent in self.pop]
            )

        pop_array = np.array([agent.solution.copy() for agent in self.pop])
        prev_array = self._prev_positions.copy()

        # Global best / worst positions (bestX and worstX in the paper)
        g_best = self.g_best.solution.copy()
        if self.problem.minmax == "min":
            worst_agent = max(self.pop, key=lambda ag: ag.target.fitness)
        else:
            worst_agent = min(self.pop, key=lambda ag: ag.target.fitness)
        g_worst = worst_agent.solution.copy()

        n = self.pop_size
        idx = np.arange(n)
        self.generator.shuffle(idx)

        # Split population into four behavioral groups:
        # ball-rolling, breeding, foraging, and stealing dung beetles.
        n_roll = n // 4
        n_breed = n // 4
        n_forage = n // 4
        n_steal = n - (n_roll + n_breed + n_forage)

        idx_roll = idx[0:n_roll]
        idx_breed = idx[n_roll : n_roll + n_breed]
        idx_forage = idx[n_roll + n_breed : n_roll + n_breed + n_forage]
        idx_steal = idx[n_roll + n_breed + n_forage :]

        pop_new = []

        # ===== 1) Ball-rolling dung beetles =====
        for i in idx_roll:
            x_t = pop_array[i]
            x_t_1 = prev_array[i]

            # Rolling behavior: a simple approximation of the original equations
            step = self.alpha * self.k * x_t_1 + self.b_const * np.abs(x_t - g_worst)
            new_pos = x_t + step
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(agent.solution)
            pop_new.append(agent)

        # ===== 2) Breeding (reproduction) dung beetles =====
        for i in idx_breed:
            x_t = pop_array[i]
            R = 1.0 - epoch / self.epoch
            lb = self.problem.lb
            ub = self.problem.ub

            Lb_star = np.maximum(g_best * (1 - R), lb)
            Ub_star = np.minimum(g_best * (1 + R), ub)

            low = np.minimum(Lb_star, Ub_star)
            high = np.maximum(Lb_star, Ub_star)

            new_pos = self.generator.uniform(low, high)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(agent.solution)
            pop_new.append(agent)

        # ===== 3) Foraging dung beetles =====
        for i in idx_forage:
            x_t = pop_array[i]
            R = 1.0 - epoch / self.epoch
            lb = self.problem.lb
            ub = self.problem.ub

            Lb_b = np.maximum(g_best * (1 - R), lb)
            Ub_b = np.minimum(g_best * (1 + R), ub)

            low_b = np.minimum(Lb_b, Ub_b)
            high_b = np.maximum(Lb_b, Ub_b)

            rand_pos = self.generator.uniform(low_b, high_b)
            new_pos = x_t + self.generator.random() * (rand_pos - x_t)

            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(agent.solution)
            pop_new.append(agent)

        # ===== 4) Stealing dung beetles =====
        for i in idx_steal:
            x_t = pop_array[i]
            S = 1.0
            g_vec = self.generator.normal(size=self.problem.n_dims)
            best_x_star = g_best

            step = S * g_vec * (
                np.abs(x_t - best_x_star) + np.abs(x_t - g_best)
            )
            new_pos = g_best + step

            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(agent.solution)
            pop_new.append(agent)

        # Evaluate the new population (in parallel modes this is done inside)
        pop_new = self.update_target_for_population(pop_new)

        # Merge old and new populations, then sort and trim to pop_size
        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, self.pop_size, self.problem.minmax
        )

        # Update previous positions x(t−1) for the next iteration
        self._prev_positions = np.array(
            [agent.solution.copy() for agent in self.pop]
        )

