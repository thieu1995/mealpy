#!/usr/bin/env python
# Created by "Halil" at 9:35, 03/01/2026 ----------%
#       Email: halilakbas11@outlook.com            %
#       Github: https://github.com/halilakbas11    %
# -------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalDandelionO(Optimizer):
    """
    The original version: Dandelion Optimizer (DandelionO)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100

    Warnings
    --------
    1. This version is implemented exactly as described in the paper and the author's original MATLAB code.
    2. However, in the MATLAB code, the author omitted the 0.01 multiplier in the Levy function,
       despite it being explicitly mentioned in the paper.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/114680-dandelion-optimizer
    2. https://doi.org/10.1016/j.engappai.2022.105075

    References
    ----------
    1. Zhao, S., Zhang, T., Ma, S., & Chen, M. (2022). Dandelion Optimizer: A nature-inspired metaheuristic
       algorithm for engineering applications. Engineering Applications of Artificial Intelligence, 114, 105075.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DandelionO
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
    >>> model = DandelionO.OriginalDandelionO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Dandelion Optimizer", year=2022, difficulty="medium", kind="original")

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main evolution step.
        """
        # --- Parameters ---
        # Eq. (8) Dynamic alpha parameter
        alpha = self.generator.random() * ((epoch / self.epoch)**2 - 2.0 * epoch / self.epoch + 1)

        # Eq. (11) Local search domain factor k
        aa = 1.0 / (self.epoch ** 2 - 2 *self.epoch + 1)
        bb = -2 * aa
        cc = 1 - aa - bb
        kk = 1 - self.generator.random() * (cc + aa*epoch**2 + bb * epoch)

        # Current population positions
        pop_pos = np.array([agent.solution for agent in self.pop])

        if self.generator.standard_normal() < 1.5:
            lamd = np.abs(self.generator.standard_normal(size=(self.pop_size, self.problem.n_dims)))
            theta = (2 * self.generator.random(self.pop_size) - 1) * np.pi
            row = 1 / np.exp(theta)
            vx = row * np.cos(theta)
            vy = row * np.sin(theta)
            pop_pos_new = self.generator.random(size=(self.pop_size, self.problem.n_dims)) * (self.problem.ub - self.problem.lb) + self.problem.lb
            # lognpdf equivalent (mu=0, sigma=1)
            lognpdf_vals = np.exp(-0.5 * (np.log(lamd)) ** 2) / (lamd * np.sqrt(2 * np.pi))
            # Eq. 5
            pop_pos_new = pop_pos + np.dot(alpha * vx * vy, lognpdf_vals * (pop_pos_new - pop_pos))
        else:
            pop_pos_new = pop_pos * kk
        ## Check bound
        pop_pos = np.clip(pop_pos_new, self.problem.lb, self.problem.ub)

        ## Decline stage
        pop_mean = np.mean(pop_pos, axis=0)  # Shape: (Dim,)
        beta_randn = self.generator.standard_normal(size=(self.pop_size, self.problem.n_dims))
        dd2 = pop_pos - beta_randn * alpha * (pop_mean - beta_randn * alpha * pop_mean)        # Eq.(13)
        # Check boundaries
        pop_pos = np.clip(dd2, self.problem.lb, self.problem.ub)

        ## Landing stage
        levy_step = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(self.pop_size, self.problem.n_dims), case=-1)
        pop_elite = np.tile(self.g_best.solution, (self.pop_size, 1))
        pop_pos = pop_elite + levy_step * alpha * (pop_elite - pop_pos * (2 * epoch / self.epoch))

        # Create new population agents
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(pop_pos[idx])
            self.pop[idx] = self.generate_empty_agent(pos_new)
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(pos_new)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)


class DevDandelionO(Optimizer):
    """
    The developed version: Dandelion Optimizer (DandelionO)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100

    Danger
    ------
    1. This dev version was contributed by the user "Halil". Several parameters—such as alpha, a, b,
       and k, differ from the original paper.
    2. Furthermore, the Levy function is applied to the entire population simultaneously, whereas the paper
       specifies generating a Levy step for each individual. If you choose to use this version,
       it must be clearly stated that it is not the original implementation.

    References
    ----------
    1. Zhao, S., Zhang, T., Ma, S., & Chen, M. (2022). Dandelion Optimizer: A nature-inspired
       metaheuristic algorithm for engineering applications. Engineering Applications of
       Artificial Intelligence, 114, 105075. https://doi.org/10.1016/j.engappai.2022.105075

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DandelionO
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
    >>> model = DandelionO.DevDandelionO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Dandelion Optimizer (Dev)", difficulty="medium", kind="developed")

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def get_lognormal_distribution(self):
        """
        Calculate Log-normal distribution components based on Eq. (7).
        Using standard normal distribution mu=0, sigma=1 with numpy.
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
        pdf = (1 / (y_abs * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(y_abs) - mu) ** 2) / (2 * sigma ** 2))
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
        levy_step = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)

        # Eq. (18) Linear increasing function delta (approx 2*t/T)
        delta = 2 * epoch / self.epoch

        # Elite position (Global Best)
        elite_pos = self.g_best.solution

        # Eq. (15) Final position update
        pop_new_pos = elite_pos + levy_step * alpha * (elite_pos - pop_descend * delta)

        # --- Final Update ---
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(pop_new_pos[idx])
            self.pop[idx] = self.generate_empty_agent(pos_new)
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(pos_new)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)
