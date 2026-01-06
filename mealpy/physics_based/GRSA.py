#!/usr/bin/env python
# Created by "Eriş" for GRSA integration ---------%
#       Email: erssylmz12@gmail.com                %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevGRSA(Optimizer):
    """
    The developed version: General Relativity Search Algorithm (GRSA)

    Hyper-parameters (typical ranges):
        + w_max (float): [0.5, 1.0], default = 0.9
            Initial step-length / inertia-like coefficient.
        + w_min (float): [0.0, 0.5], default = 0.1
            Final step-length / inertia-like coefficient.
        + k_g (float): [0.1, 1.0], default = 0.5
            Relativistic factor scaling in the gamma term.
        + mutation_rate (float): (0, 1], default = 0.1
            Fraction of individuals to perturb each iteration.

    """

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 50,
        w_max: float = 0.9,
        w_min: float = 0.1,
        k_g: float = 0.5,
        mutation_rate: float = 0.1,
        **kwargs: object,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): population size, default = 50
            w_max (float): initial inertia-like factor in step length
            w_min (float): final inertia-like factor in step length
            k_g (float): relativistic factor scaling
            mutation_rate (float): ratio of mutated individuals per epoch
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.w_max = self.validator.check_float("w_max", w_max, (0.0, 2.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0.0, 2.0))
        self.k_g = self.validator.check_float("k_g", k_g, (0.0, 5.0))
        self.mutation_rate = self.validator.check_float(
            "mutation_rate", mutation_rate, (0.0, 1.0)
        )

        self.set_parameters(
            ["epoch", "pop_size", "w_max", "w_min", "k_g", "mutation_rate"]
        )

        self.prev_positions = None  
        self.sort_flag = True       


    def _compute_inertia_weight(self, epoch: int) -> float:
        """
        Linearly decreasing inertia-like coefficient w_t.
        """
        if self.epoch <= 1:
            return self.w_min
        return self.w_max - (self.w_max - self.w_min) * (epoch / (self.epoch - 1))

    def _init_prev_positions(self) -> None:
        """
        Initialize previous positions with current population solutions.
        Called implicitly in the first evolve call.
        """
        if self.prev_positions is None:
            self.prev_positions = np.array(
                [agent.solution.copy() for agent in self.pop],
                dtype=float,
            )


    def evolve(self, epoch: int):
        """
        The main operations (equations) of GRSA. Inherit from Optimizer class.

        Args:
            epoch (int): The current iteration (1-based)
        """
        self._init_prev_positions()

        lb = self.problem.lb
        ub = self.problem.ub
        n_dims = self.problem.n_dims

      
        positions = np.array([agent.solution for agent in self.pop], dtype=float)
        prev_positions = self.prev_positions
        g_best = self.g_best.solution.copy()

        w_t = self._compute_inertia_weight(epoch)
        pop_new = []

        for idx in range(self.pop_size):
            x = positions[idx]
            x_prev = prev_positions[idx]

        
            r1 = self.generator.integers(0, self.pop_size)
            r2 = self.generator.integers(0, self.pop_size)
            x_rand1 = positions[r1]
            x_rand2 = positions[r2]

            
            K_v = np.abs(g_best - x_rand1)
            xi = self.generator.random()
            gamma = 1.0 + xi * (1.0 - self.k_g)
            v_rel = np.sqrt(max(1.0 - 1.0 / (gamma ** 2), 0.0))
            lam = w_t * K_v * v_rel
            delta_tl = np.sign(x - x_prev)
            delta_sl = np.sign(x - g_best)
            delta_null = np.sign(x_rand2 - x)
            K_f = self.generator.integers(0, 2, size=n_dims)
            mix = delta_tl + K_f * delta_sl + (1 - K_f) * delta_null
            delta = -np.sign(mix)

            pos_new = x + lam * delta
            if self.generator.random() < self.mutation_rate:
               
                noise = 0.1 * (ub - lb) * self.generator.normal(size=n_dims)
                pos_new = g_best + noise

           
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

           
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

       
        self.prev_positions = np.array(
            [agent.solution.copy() for agent in self.pop],
            dtype=float,
        )


class OriginalGRSA(DevGRSA):
    """
    The original version of: General Relativity Search Algorithm (GRSA)

    Notes
    -----
    * This class reuses DevGRSA's implementation but exposes it as the "original"
      version for compatibility with MEALPY's naming convention.
    * If you later match the exact paper / MATLAB implementation, you can
      override `evolve` and/or add pre-generated random structures 

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GRSA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>>
    >>> model = GRSA.OriginalGRSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 50,
        w_max: float = 0.9,
        w_min: float = 0.1,
        k_g: float = 0.5,
        mutation_rate: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(epoch, pop_size, w_max, w_min, k_g, mutation_rate, **kwargs)
  
        self.support_parallel_modes = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Boundary handling for the original version.

        """
        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rd)
