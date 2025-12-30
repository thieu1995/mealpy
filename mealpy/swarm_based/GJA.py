#!/usr/bin/env python
# Created by "Thieu" at 09:00, 09/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from math import gamma, cosh
from mealpy.optimizer import Optimizer


class OriginalGJA(Optimizer):
    """
    The original version of: Gekko Japonicus Algorithm (GJA)

    Links:
        1. https://doi.org/10.1007/s42235-025-00805-6
        2. https://github.com/zhy1109/Gekko-japonicusalgorithm

    The algorithm draws inspiration from the predation strategies and survival behaviors
    of the Gekko japonicus (Japanese gecko). It simulates various biological behaviors including:
        - Hybrid locomotion patterns (Levy flight + Gaussian perturbation)
        - Directional olfactory guidance
        - Implicit group advantage tendencies
        - Tail autotomy mechanism for escaping local optima
        - Historical memory injection for maintaining diversity

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + beta_start (float): [1.0, 1.5], starting value for beta parameter, default=1.2
        + beta_end (float): [0.1, 0.5], ending value for beta parameter, default=0.3
        + alpha_ratio (float): [0.4, 0.8], ratio to calculate alpha from beta, default=0.6

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GJA
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
    >>> model = GJA.OriginalGJA(epoch=1000, pop_size=30)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, K., Zhao, H., Li, X., Fu, C. and Jin, J., 2025. Gekko Japonicus Algorithm: A Novel
        Nature-inspired Algorithm for Engineering Problems and Path Planning. Journal of Bionic Engineering.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 beta_start: float = 1.2, beta_end: float = 0.3,
                 alpha_ratio: float = 0.6, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            beta_start (float): Starting value of beta parameter, default = 1.2
            beta_end (float): Ending value of beta parameter, default = 0.3
            alpha_ratio (float): Ratio to calculate alpha from beta, default = 0.6
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.beta_start = self.validator.check_float("beta_start", beta_start, [0.5, 2.0])
        self.beta_end = self.validator.check_float("beta_end", beta_end, [0.1, 1.0])
        self.alpha_ratio = self.validator.check_float("alpha_ratio", alpha_ratio, [0.1, 1.0])
        self.set_parameters(["epoch", "pop_size", "beta_start", "beta_end", "alpha_ratio"])
        self.sort_flag = False

    def initialize_variables(self) -> None:
        """Initialize variables before the main loop."""
        self.memory = []
        # FIX: Use epoch + 1 to handle mealpy's 0-to-epoch range (inclusive)
        self.decay = np.linspace(1, 0, self.epoch + 1) ** 2
        self.range_bound = self.problem.ub - self.problem.lb

    def levy_flight_(self, beta: float) -> np.ndarray:
        """
        Lévy flight operator - Equations (6) and (7) in the paper.

        Args:
            beta (float): Lévy exponent

        Returns:
            np.ndarray: Lévy flight step vector
        """
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.generator.normal(0, sigma, self.problem.n_dims)
        v = self.generator.normal(0, 1, self.problem.n_dims)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Cosine-hyperbolic driven decay - Equation (4)
        t_ratio = (epoch + 1) / self.epoch
        beta = self.beta_start - (self.beta_start - self.beta_end) * \
               ((cosh(t_ratio * 5) - 1) / (cosh(5) - 1))
        alpha = beta * self.alpha_ratio
        decay_value = self.decay[epoch]

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_current = self.pop[idx].solution.copy()

            # Stage 1: Hybrid Movement Strategy - Equations (3), (6), (7), (9)
            if self.generator.random() < 0.5:
                step = self.levy_flight_(beta) * alpha
            else:
                step = self.generator.normal(0, 1, self.problem.n_dims) * alpha * \
                       self.range_bound / np.sqrt(epoch + 2)
            pos_new = pos_current + step

            # Stage 2: Olfactory Assisted Guidance - Equations (10), (11)
            if (epoch + 1) % 3 == 0:
                dir_factor = np.sign(self.g_best.solution - pos_new)
                pos_new = pos_new + 0.3 * self.generator.random(self.problem.n_dims) * \
                          dir_factor * self.range_bound

            # Stage 3: Implicit Group Profit Seeking - Equations (12), (13), (14)
            fitness_values = np.array([agent.target.fitness for agent in self.pop])
            if self.problem.minmax == "min":
                sorted_idx = np.argsort(fitness_values)
            else:
                sorted_idx = np.argsort(fitness_values)[::-1]

            scent_idx = int(np.ceil(self.generator.random() ** 2 * self.pop_size)) - 1
            scent_idx = max(0, min(scent_idx, self.pop_size - 1))
            scent_source = self.pop[sorted_idx[scent_idx]].solution
            pos_new = pos_new + 0.5 * decay_value * (scent_source - pos_new)

            # Stage 4: Tail Autotomy Mutation - Equations (15), (16), (17)
            dist_to_best = np.linalg.norm(pos_new - self.g_best.solution)
            dist_threshold = 0.1 * np.linalg.norm(self.range_bound)

            if dist_to_best > dist_threshold:
                mut_prob = (1 - decay_value) * np.ones(self.problem.n_dims)
                mut_mask = self.generator.random(self.problem.n_dims) < mut_prob
                mut_dims = np.where(mut_mask)[0]

                if len(mut_dims) > 0:
                    pos_new[mut_dims] = self.g_best.solution[mut_dims] + \
                                        self.generator.normal(0, 1, len(mut_dims)) * \
                                        self.range_bound[mut_dims] / (epoch + 2)

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Stage 5: Historical Memory Injection - Algorithm 1
        if (epoch + 1) % 5 == 0:
            fitness_values = np.array([agent.target.fitness for agent in self.pop])
            if self.problem.minmax == "min":
                top5_idx = np.argsort(fitness_values)[:5]
            else:
                top5_idx = np.argsort(fitness_values)[-5:][::-1]

            for idx in top5_idx:
                pos = self.pop[idx].solution.copy()
                is_duplicate = any(np.allclose(pos, m, rtol=1e-10) for m in self.memory)
                if not is_duplicate:
                    self.memory.append(pos)

            self.memory = self.memory[:5]

            if len(self.memory) > 0:
                n_worst = int(np.ceil(0.1 * self.pop_size))
                if self.problem.minmax == "min":
                    worst_idx = np.argsort(fitness_values)[-n_worst:]
                else:
                    worst_idx = np.argsort(fitness_values)[:n_worst]

                for idx in worst_idx:
                    mem_idx = self.generator.integers(0, len(self.memory))
                    new_pos = self.memory[mem_idx].copy()
                    new_pos = self.correct_solution(new_pos)
                    self.pop[idx] = self.generate_agent(new_pos)