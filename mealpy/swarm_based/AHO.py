#!/usr/bin/env python
# Created by "Hasna Sena Kaymak" on 06/01/2026
# Github: https://github.com/SenaKymk
# --------------------------------------------------%
# Updated by "Thieu" on 12/07/2026
# Github: https://github.com/thieu1995
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalAHO(Optimizer):
    """
    The original version of: Archerfish Hunting Optimizer (AHO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    theta : float
        Good range [0, pi], The swapping angle between exploration and exploitation, default: pi/12.
    omega : float
        Good range [0, 100.], The attractiveness rate, default: 0.01.

    Danger
    ------
    1. Empirical evaluations have exposed several critical flaws in its fundamental design:
    2. Architectural Inefficiency: Unlike standard metaheuristic algorithms that
       operate with O(N) population loops, AHO explicitly employs deeply nested
       O(N^2) population loops during its exploration (shooting) phase. This
       unorthodox design causes severe computational bottlenecking and wastes
       resources without yielding proportional exploration benefits.
    3. Convergence Failure & Literature Discrepancy: Independent testing reveals
       that AHO struggles to converge even on simple unimodal landscapes (e.g.,
       the Sphere function), failing to reach the global optimum after 10,000+
       iterations. These empirical outcomes strongly contradict the high-performance
       claims published in the original paper.
    4. Production Unsuitability: Due to the extreme computational overhead and
       stagnation risks, this implementation is strictly provided for academic
       reproducibility and critical analysis. It is NOT recommended for solving
       practical, large-scale, or real-world optimization problems.

    References
    ----------
    1. Zitouni, F., Harous, S., Belkeram, A., & Hammou, L. E. B. (2022).
       The archerfish hunting optimizer: A novel metaheuristic algorithm for global optimization.
       Arabian Journal for Science and Engineering, 47(2), 2513-2553. https://doi.org/10.1007/s13369-021-06208-z

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, AHO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem = {
    >>>     "obj_func": objective_function,
    >>>     "bounds": FloatVar(lb=[-10., ]*10, ub=[10., ]*10),
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AHO.OriginalAHO(epoch=100, pop_size=50, theta=0.26, omega=0.01)
    >>> g_best = model.solve(problem)
    >>> print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Archerfish Hunting Optimizer", year=2022, difficulty="hard", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.LACK_OF_NOVELTY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH
                       ))
    
    def __init__(self, epoch: int = 10000, pop_size: int = 100, theta: float = 0.26, omega: float = 0.01, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size, default = 100
            theta (float): Swapping angle between exploration and exploitation, default = pi/12=0.26
            omega (float): Attractiveness rate, default = 0.01
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.theta = self.validator.check_float("theta", theta, [0., np.pi])
        self.omega = self.validator.check_float("omega", omega, [0., 100.0])
        self.set_parameters(["epoch", "pop_size", "theta", "omega"])
        self.sort_flag = False
        self.is_parallelizable = False
    
    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        self.stagnation_counter = np.zeros(self.pop_size, dtype=int)
        # Threshold for triggering Lévy flight (d × N as per paper)
        self.limit_stagnation = self.problem.n_dims * self.pop_size
    
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        
        Args:
            epoch (int): The current iteration
        """
        for idx in range(self.pop_size):
            # Generate perceiving angle theta_0 (Equation 6)
            b = self.generator.integers(0, 2)  # Bernoulli distribution (0 or 1)
            alpha_random = self.generator.random()
            theta_0 = np.power(-1, b) * alpha_random * np.pi
            abs_theta_0 = np.abs(theta_0)       # Determine exploration or exploitation based on theta_0

            # --- Phase 1: Shooting behavior (Exploration) ---
            # Equation 2 and 3
            if (abs_theta_0 >= 0 and abs_theta_0 < self.theta) or (abs_theta_0 > np.pi - self.theta and abs_theta_0 <= np.pi):
                # Compute X_prey using Eq 3
                sparse_vec = np.zeros(self.problem.n_dims)
                kdx = self.generator.integers(0, self.problem.n_dims)
                sparse_vec[kdx] = self.omega * np.sin(2 * theta_0)
                X_prey = self.pop[idx].solution + sparse_vec + self.generator.uniform(-0.5, 0.5, self.problem.n_dims)
                X_prey = self.correct_solution(X_prey)
                target_prey = self.get_target(X_prey)

                for jdx in range(self.pop_size):
                    # Check fitness
                    if self.compare_target(target_prey, self.pop[jdx].target, self.problem.minmax):
                        # Update the location unconditionally using Eq 2
                        dist_sq = np.sum((X_prey - self.pop[jdx].solution) ** 2)
                        X_new = self.pop[jdx].solution + np.exp(-dist_sq) * (X_prey - self.pop[jdx].solution)
                        X_new = self.correct_solution(X_new)
                        self.pop[jdx] = self.generate_agent(X_new)
                        self.stagnation_counter[jdx] = 0  # Location changed
                    else:
                        # Location has not been changed
                        self.stagnation_counter[jdx] += 1
                        # Check stagnation threshold
                        if self.stagnation_counter[jdx] >= self.limit_stagnation:
                            X_new = self.pop[jdx].solution + self.get_levy_flight_step(beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=0)
                            X_new = self.correct_solution(X_new)
                            self.pop[jdx] = self.generate_agent(X_new)
                            self.stagnation_counter[jdx] = 0  # Reset after applying Levy

            # --- Phase 2: Jumping behavior (Exploitation) ---
            if self.theta <= abs_theta_0 <= (np.pi - self.theta):
                # Compute X_prey using Eq 5
                sparse_vec = np.zeros(self.problem.n_dims)
                if self.problem.n_dims > 1:
                    indices = self.generator.choice(self.problem.n_dims, 2, replace=False)
                    sparse_vec[indices[0]] = self.omega * np.sin(2 * theta_0)
                    sparse_vec[indices[1]] = -self.omega * (np.sin(theta_0) ** 2)
                else:
                    sparse_vec[0] = self.omega * np.sin(2 * theta_0) - self.omega * (np.sin(theta_0) ** 2)
                epsilon = self.generator.uniform(-0.5, 0.5, self.problem.n_dims)
                X_prey = self.pop[idx].solution + sparse_vec + epsilon
                X_prey = self.correct_solution(X_prey)
                target_prey = self.get_target(X_prey)

                if self.compare_target(target_prey, self.pop[idx].target, self.problem.minmax):
                    # Update the location unconditionally using Eq 4
                    dist_sq = np.sum((X_prey - self.pop[idx].solution) ** 2)
                    X_new = self.pop[idx].solution + np.exp(-dist_sq) * (X_prey - self.pop[idx].solution)
                    X_new = self.correct_solution(X_new)
                    self.pop[idx] = self.generate_agent(X_new)
                    self.stagnation_counter[idx] = 0  # Location changed
                else:
                    # Location has not been changed
                    self.stagnation_counter[idx] += 1
                    # Check stagnation threshold
                    if self.stagnation_counter[idx] >= self.limit_stagnation:
                        X_new = self.pop[idx].solution + self.get_levy_flight_step(beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=0)
                        X_new = self.correct_solution(X_new)
                        self.pop[idx] = self.generate_agent(X_new)
                        self.stagnation_counter[idx] = 0  # Reset after applying Levy
