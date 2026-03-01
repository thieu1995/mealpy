#!/usr/bin/env python
# Created by "Thieu" at 10:47, 31/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalDandelionO(Optimizer):
    """
    Dandelion Optimizer (DandelionO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105075
        2. https://www.mathworks.com/matlabcentral/fileexchange/114680-dandelion-optimizer

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): Maximum number of iterations, default = 10000
        + pop_size (int): Population size, default = 100

    Notes:
        1. The Levy flight step is calculated using the built-in method get_levy_flight_step() from the Optimizer class.
        2. 'q' factor calculation strictly follows Eq. (12) using quadratic coefficients a, b, c.
        3. Vectorized implementation allows for efficient execution without explicit loops.
        4. This code is based on the original MATLAB code and the original paper[1].

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.bio_based.DandelionO import OriginalDandelionO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = OriginalDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shijie Zhao, Tianran Zhang, Shilin Ma, Miao Chen,
    Dandelion Optimizer: A nature-inspired metaheuristic algorithm for engineering applications,
    Engineering Applications of Artificial Intelligence,Volume 114,2022,105075,ISSN 0952-1976,
    https://doi.org/10.1016/j.engappai.2022.105075.

    """
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def get_lognormal_distribution(self):
        """
        Calculate Log-normal distribution components based on Eq. (7).
        Using standard normal distribution μ=0, σ=1 with numpy.
        """
        # Eq. (7) formula implementation using numpy
        # Generate random variable from standard normal distribution
        y = self.generator.standard_normal((self.pop_size, self.problem.n_dims))

        # Standard Log-normal PDF formula: (1 / (y * sigma * sqrt(2*pi))) * exp(- (ln(y) - mu)^2 / (2*sigma^2))
        # We use standard normal (mu=0, sigma=1)
        sigma = 1.0
        mu = 0.0
        # Avoid negative/zero values for log calculation
        y_abs = np.abs(y) + 1e-100
        pdf = (1 / (y_abs * sigma * np.sqrt(2 * np.pi))) * \
              np.exp(-((np.log(y_abs) - mu) ** 2) / (2 * sigma ** 2))
        return pdf

    def evolve(self, epoch):
        """
        The main evolution step.
        """
        # --- Parameters ---
        # Eq. (8) Dynamic alpha parameter, factored version of ((1/Maxiteration^2)*t^2-2/Maxiteration*t+1)
        alpha = self.generator.random() * ((1 - epoch / self.epoch) ** 2)

        # Current population positions
        pop_pos = np.array([agent.solution for agent in self.pop])

        # --- 1. Rising Stage (Eq. 12) ---
        # Eq. (9) Calculate lift components vx, vy
        # theta randomly in [-pi, pi]
        theta = self.generator.uniform(-np.pi, np.pi, (self.pop_size, self.problem.n_dims))
        r = 1 / np.exp(theta)
        v_x = r * np.cos(theta)
        v_y = r * np.sin(theta)

        # Eq. (7) Log-normal distribution factor
        ln_y = self.get_lognormal_distribution()

        # Eq. (6) Random positions for exploration
        X_s = self.generator.uniform(self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims))

        # Eq. (11) Local search domain factor k
        # Factored version of (c+a*t^2+b*t) from the paper
        q_coef = (epoch / self.epoch) ** 2 - 2 * (epoch / self.epoch) + 1
        k = 1 - self.generator.random() * q_coef

        # Weather conditions: Random check
        # Assuming standard normal distribution check < 1.5 as in many implementations
        weather_check = self.generator.standard_normal((self.pop_size, 1))

        # Calculate steps for both conditions
        # Eq. (5) Clear weather (Exploration)
        step_clear = pop_pos + alpha * v_x * v_y * ln_y * (X_s - pop_pos)

        # Eq. (10) Rainy weather (Exploitation)
        step_rainy = pop_pos * k

        # Select based on weather condition
        # If check < 1.5 -> Clear weather, else -> Rainy weather
        pop_rise = np.where(weather_check < 1.5, step_clear, step_rainy)

        # --- 2. Descending Stage (Eq. 13) ---
        # Eq. (14) Mean position after rising stage
        mean_pos = np.mean(pop_rise, axis=0)

        # Brownian motion (Standard Normal Distribution)
        brownian = self.generator.standard_normal((self.pop_size, self.problem.n_dims))

        # Eq. (13) Update positions
        pop_descend = pop_rise - alpha * brownian * (mean_pos - alpha * brownian * pop_rise)

        # --- 3. Landing Stage (Eq. 15) ---
        # Eq. (16) Levy flight step
        # Using mealpy's built-in get_levy_flight_step from Optimizer class
        # beta=1.5, multiplier=0.01 as per paper Eq. (16) (s=0.01)
        # case=-1 returns the step vector directly
        levy_step = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)

        # Eq. (18) Linear increasing function delta (approx 2*t/T)
        delta = 2 * epoch / self.epoch

        # Elite position (Global Best)
        elite_pos = self.g_best.solution

        # Eq. (15) Final position update
        pop_new_pos = elite_pos + levy_step * alpha * (elite_pos - pop_descend * delta)

        # --- Final Update ---
        # Check boundaries
        pop_new_pos = np.clip(pop_new_pos, self.problem.lb, self.problem.ub)
        pop_new_pos = self.amend_solution(pop_new_pos)

        # Create new population agents
        pop_new = []
        for i in range(self.pop_size):
            # Using generate_agent implicitly calculates the fitness/target
            agent = self.generate_agent(pop_new_pos[i])
            pop_new.append(agent)

        # Update population
        self.pop = pop_new