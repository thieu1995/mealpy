#!/usr/bin/env python
# Created by "Thieu" at 23:01, 09/07/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGRSA(Optimizer):
    """
    The original version of: General Relativity Search Algorithm (GRSA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/57520-general-relativity-search-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_geometry (int): (1, pop_size/2) -> Number of geometries used to partition the population.
        + w_max (float): (0, 1) -> Maximum weight factor for kinetic energy calculation.
        + w_min (float): (0, 1) -> Minimum weight factor for kinetic energy calculation.
        + g_max (float): (0, 1) -> Maximum kinetic energy coefficient.
        + g_min (float): (0, 1) -> Minimum kinetic energy coefficient.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GRSA
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
    >>> model = GRSA.OriginalGRSA(epoch=100, pop_size=50, n_geometry=5, w_max=0.9, w_min=0.4, g_max=0.5, g_min=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Beiranvand, Hamzeh, and Esmaeel Rokrok. "General relativity search algorithm: a global
    optimization approach." International journal of computational intelligence and applications 14.03 (2015): 1550017.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_geometry: int = 5, w_max: float = 0.9,
                 w_min: float = 0.4, g_max: float = 0.5, g_min: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_geometry = self.validator.check_int("n_geometry", n_geometry, [1, int(self.pop_size / 2)])
        self.w_max = self.validator.check_float("w_max", w_max, (0., 1.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0., 1.0))
        self.g_max = self.validator.check_float("g_max", g_max, (0., 1.0))
        self.g_min = self.validator.check_float("g_min", g_min, (0., 1.0))
        self.set_parameters(["epoch", "pop_size", "n_geometry", "w_max", "w_min", "g_max", "g_min"])
        # State variables
        self.geodesic_x_p = None
        self.geodesic_f = None
        self.prev_pop = None

        self.sort_flag = False

    def before_main_loop(self):
        self.geodesic_x_p = np.array([agent.solution for agent in self.pop])
        self.geodesic_f = np.array([agent.target.fitness for agent in self.pop])
        self.prev_pop = np.array([agent.solution.copy() for agent in self.pop])

    def calculate_x_direction(self, b_sign, x_p, geodesic_x_p, x_p_prev, x_p_gbest):
        s_sign = np.sign(x_p - x_p_prev)
        g_sign = np.sign(x_p - x_p_gbest)
        p_sign = np.sign(x_p - geodesic_x_p)
        dx = -np.sign(s_sign + np.abs(b_sign) * g_sign + (1 - np.abs(b_sign)) * p_sign)
        # Boundary constraints
        dx[x_p > self.problem.ub] = -1
        dx[x_p < self.problem.lb] = 1
        zero_idx = (dx == 0)
        dx[zero_idx] = -1
        return dx

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        n_dims = self.problem.n_dims
        n_orn_g = self.pop_size // self.n_geometry
        wt = self.w_max - epoch * (self.w_max - self.w_min) / self.epoch

        # Current status
        x_p = np.array([agent.solution for agent in self.pop])
        fitness_list = np.array([agent.target.fitness for agent in self.pop])
        sorted_indices = np.argsort(fitness_list)

        # Precompute geometry indices
        sg_idx = np.array([i * n_orn_g for i in range(self.n_geometry)])
        eg_idx = np.array([(i + 1) * n_orn_g for i in range(self.n_geometry)])

        pop_updated = x_p.copy()
        # Main geometry loop
        for g in range(self.n_geometry):
            indices = sorted_indices[sg_idx[g]:eg_idx[g]]

            # Kv calculation (Distance between best and random in group)
            rand_st = self.generator.integers(0, n_orn_g - 1)
            kv = np.abs(x_p[indices[-1]] - x_p[indices[rand_st]])

            # Update agents in geometry
            for jj in indices:
                i_g = np.where(indices == jj)[0][0] + 1

                ek_mc2 = (i_g - self.pop_size) / (self.pop_size - 1) * self.g_max + (1 - i_g) / (self.pop_size - 1) * self.g_min
                gama0 = 1 + ek_mc2
                kg = self.generator.random(n_dims)
                gama = 1 * kg + gama0 * (1 - kg)

                v_drag = np.sqrt(np.maximum(0, (1 - gama ** 2) / (gama ** 2 + self.EPSILON)))
                landa = wt * kv * v_drag

                b_sign = self.generator.random(n_dims) < 0.5
                dx = self.calculate_x_direction(b_sign, x_p[jj], self.geodesic_x_p[jj], self.prev_pop[jj], self.g_best.solution)

                uc_max = 0.5 * (self.problem.ub - self.problem.lb)
                upx = np.clip(landa * dx, -uc_max, uc_max)
                pop_updated[jj] = np.clip(x_p[jj] + upx, self.problem.lb, self.problem.ub)

            # Mutation step (Intra-geometry)
            ii = 0
            for gg in range(self.n_geometry):
                if gg != g:
                    for _ in range(n_orn_g):
                        if ii < n_orn_g:
                            alfa = self.generator.random(n_dims) < 0.5
                            target_idx = indices[ii]
                            pop_updated[target_idx] = alfa * self.g_best.solution + (1 - alfa) * pop_updated[target_idx]
                            ii += 1

        # Update Geodesic and Global status
        pop_new = []
        for idx in range(self.pop_size):
            self.prev_pop[idx] = x_p[idx].copy()
            pos_new = self.correct_solution(pop_updated[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target(pos_new)
                self.pop[idx].update(solution=pos_new, target=target)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

        for idx in range(self.pop_size):
            if self.compare_fitness(self.pop[idx].target.fitness, self.geodesic_f[idx]):
                self.geodesic_f[idx] = self.pop[idx].target.fitness
                self.geodesic_x_p[idx] = self.pop[idx].solution.copy()

        # Reset condition
        if np.std(np.array([agent.solution for agent in self.pop]), axis=0).mean() < self.EPSILON:
            self.pop = self.generate_population(self.pop_size)
