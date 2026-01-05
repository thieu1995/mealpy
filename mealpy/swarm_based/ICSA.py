#!/usr/bin/env python
# Created by "Thieu" at 11:30, 05/01/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalICSA(Optimizer):
    """
    Improved Chameleon Swarm Algorithm (ICSA)

    Links:
        1. https://doi.org/10.3390/biomimetics9100583
        2. https://doi.org/10.1016/j.eswa.2021.114685
        3. https://www.mathworks.com/matlabcentral/fileexchange/98014-chameleon-swarm-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): Maximum number of iterations, default = 1000
        + pop_size (int): Population size, default = 100
        + beta (float): Lévy flight constant (Eq. 13), default = 1.5
        + r_chaos (float): Control parameter for logistic mapping (Eq. 10), default = 0.3
        + k_spiral (int): Variation coefficient for spiral search (Eq. 11), default = 5

    Notes:
        1. Logistic mapping initialization for population diversity.
        2. Sub-population spiral search strategy implemented for prey-search phase.
        3. Lévy flight strategy combined with cosine adaptive factor for eyes' rotation stage.
        4. Refraction reverse-learning strategy applied during prey-capture to avoid local optima.
        5. Vectorized implementation ensures high computational efficiency.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.swarm_based.ICSA import OriginalICSA
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
    >>> model = OriginalICSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, Y.; Cao, L.; Yue, Y. Hybrid Multi-Objective Chameleon Optimization Algorithm Based on
        Multi-Strategy Fusion and Its Applications. Biomimetics 2024, 9, 583.
    [2] Malik, B.S. Chameleon Swarm Algorithm: A bio-inspired optimizer for solving engineering
        design problems. Expert Systems with Applications, 2021, 174, 114685.
    """

    def __init__(self, epoch=1000, pop_size=100, beta=1.5, r_chaos=0.3, k_spiral=5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.beta = self.validator.check_float("beta", beta, [1.0, 2.0])
        self.r_chaos = self.validator.check_float("r_chaos", r_chaos, [0.0, 1.0])
        self.k_spiral = self.validator.check_int("k_spiral", k_spiral, [1, 20])
        self.set_parameters(["epoch", "pop_size", "beta", "r_chaos", "k_spiral"])
        self.sort_flag = True

    def initialize_variables(self):
        # 4.1. Logistic Chaotic Map Initialization (Eq. 9 & 10)
        l_seq = self.generator.random((self.pop_size, self.problem.n_dims))
        for j in range(self.pop_size - 1):
            l_seq[j + 1] = self.r_chaos * l_seq[j] * (1 - l_seq[j])  # Eq. (10)

        pop_pos = self.problem.lb + l_seq * (self.problem.ub - self.problem.lb)  # Eq. (9)
        self.pop = [self.generate_agent(pos) for pos in pop_pos]

    def evolve(self, epoch):
        t, t_max = epoch, self.epoch
        mu = np.exp(-(3.5 * t / t_max) ** 3)  # Eq. (2)

        pop_pos = np.array([agent.solution for agent in self.pop])
        pop_fit = np.array([agent.target.fitness for agent in self.pop])
        mean_fit = np.mean(pop_fit)

        # --- 1. Search for Prey: Sub-Population Hybrid Strategy (Eq. 11) ---
        mask_a = pop_fit < mean_fit
        mask_b = ~mask_a
        new_pos = pop_pos.copy()

        if np.any(mask_a):
            rv = self.generator.random(2)
            direction = self.g_best.solution - pop_pos[mask_a]
            new_pos[mask_a] = pop_pos[mask_a] + mu * (rv[1] * direction + rv[0] * direction)

        if np.any(mask_b):
            l_val = self.generator.uniform(-1, 1, (np.sum(mask_b), self.problem.n_dims))
            pk = self.k_spiral * np.cos(np.pi * (1 - t / t_max))  # Eq. (11) variation factor
            spiral = np.exp(pk * l_val) * np.cos(2 * np.pi * l_val)  # Spiral search
            sgn = np.sign(self.generator.random(new_pos[mask_b].shape) - 0.5)
            new_pos[mask_b] = self.g_best.solution + spiral * (self.g_best.solution - pop_pos[mask_b]) * sgn

        new_pos = self.amend_solution(new_pos)
        for i in range(self.pop_size):
            agent_new = self.generate_agent(new_pos[i])
            if agent_new.target.fitness < self.pop[i].target.fitness:
                self.pop[i] = agent_new

        # --- 2. Eyes' Rotation Stage: Lévy Flight (Eq. 15) ---
        ct = 0.075 * (1 + np.cos(np.pi * t / t_max))  # Eq. (12) adaptive factor
        levy_step = self.get_levy_flight_step(beta=self.beta, multiplier=0.01, case=-1)

        current_pos = np.array([a.solution for a in self.pop])
        xl = current_pos + ct * (self.g_best.solution - current_pos) * levy_step  # Eq. (15)

        xl = self.amend_solution(xl)
        for i in range(self.pop_size):
            agent_levy = self.generate_agent(xl[i])
            if agent_levy.target.fitness < self.pop[i].target.fitness:
                self.pop[i] = agent_levy

        # --- 3. Hunting Prey: Refraction Reverse-Learning (Eq. 20) ---
        current_pos = np.array([a.solution for a in self.pop])
        mid = (self.problem.lb + self.problem.ub) / 2
        x_star = mid + mid / (t + 1e-10) - current_pos / (t + 1e-10)  # Eq. (20)

        x_star = self.amend_solution(x_star)
        for i in range(self.pop_size):
            agent_star = self.generate_agent(x_star[i])
            if agent_star.target.fitness < self.pop[i].target.fitness:
                self.pop[i] = agent_star