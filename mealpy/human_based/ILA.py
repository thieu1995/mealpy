#!/usr/bin/env python
# Created by "Thieu" at 14:07, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalILA(Optimizer):
    """
    The original version of: Incomprehensible but Intelligible-in-time Logics Algorithm (ILA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    n_models : int
        Number of models for grouping in Stage 1, in range [2, int(pop_size / 2)]. Default is 5.
    p_s1 : float
        Maximum percentage of iterations in Stage 1, in range [0.01, 0.5]. Default is 0.33.
    p_s2 : float
        Maximum percentage of iterations in Stage 2, in range [0.01, 0.5]. Default is 0.33.
    b_min : float
        The minimum boundary for the parameters of IbI, in range [-10.0, 10.0]. Default is 0.4.
    b_max : float
        The maximum boundary for the parameters of IbI, in range [-10.0, 10.0]. Default is 0.6.


    .. attention::
       1. This is one of the most complex and lengthiest algorithms we have ever implemented.
          The complexity stems from the group partitioning logic and various logical operations
          that do not match the real-world behaviors described in the paper.
       2. This algorithm cannot be parallelized; furthermore, it evaluates a high number of
          function evaluations (NFEs) within a single iteration. Users should exercise caution
          when applying it to large-scale problems.
       3. Aside from being complex, it also involves numerous parameters that heavily impact overall performance.

    References
    ~~~~~~~~~~
    1. Mirrashid, M., & Naderpour, H. (2023). Incomprehensible but Intelligible-in-time logics: Theory and
       optimization algorithm. Knowledge-Based Systems, 264, 110305.
       https://doi.org/10.1016/j.knosys.2023.110305

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ILA
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
    >>> model = ILA.OriginalILA(epoch=1000, pop_size=50, n_models=5, p_s1=0.33, p_s2=0.33, b_min=0.4, b_max=0.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_models: int = 5, p_s1: float=0.33, p_s2: float=0.33,
                 b_min: float=0.4, b_max: float=0.6, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_models = self.validator.check_int("n_models", n_models, [2, int(self.pop_size / 2)])
        self.p_s1 = self.validator.check_float("p_s1", p_s1, [0.01, 0.5])
        self.p_s2 = self.validator.check_float("p_s2", p_s2, [0.01, 0.5])
        self.b_min = self.validator.check_float("b_min", b_min, [-10., 10.])
        self.b_max = self.validator.check_float("b_max", b_max, [-10., 10.])
        self.set_parameters(["epoch", "pop_size", "n_models", "p_s1", "p_s2", "b_min", "b_max"])
        self.sort_flag = False

        # Iteration tracking
        self.n_g_max = max(1, self.pop_size // 2)
        self.n_t1 = int(self.p_s1 * self.epoch)
        self.n_t2 = int(self.p_s2 * self.epoch)
        self.n_t3 = self.epoch - self.n_t1 - self.n_t2
        # Calculate iterations per model for Stage 1
        self.t_m = max(1, self.n_t1 // self.n_models) if self.n_models > 0 else 1

        # Initialize population
        self.experts = None
        self.fitness = None
        self.pop_prev = None
        self.current_best = None
        self.current_n_g = None

    def before_main_loop(self):
        self.experts = np.array([agent.solution for agent in self.pop])
        self.fitness = np.array([agent.target.fitness for agent in self.pop])
        self.pop_prev = self.pop.copy()
        self.current_best = self.g_best.copy()
        self.current_n_g = 1

    def normalize(self, values):
        """Normalize an array of values to [0, 1] range."""
        val_min, val_max = np.min(values), np.max(values)
        if val_max == val_min:
            return np.zeros_like(values)
        return (values - val_min) / (val_max - val_min)

    def kmeans_simple(self, data, k, max_iters=100):
        """Lightweight k-means clustering."""
        centroids = data[self.generator.choice(data.shape[0], k, replace=False)]
        clusters = np.zeros(data.shape[0])
        for _ in range(max_iters):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            clusters = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [data[clusters == i].mean(axis=0) if np.any(clusters == i) else data[self.generator.choice(data.shape[0])]
                 for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return clusters

    def stage_1_step(self, epoch):
        """Stage 1: Groupwork (Exploration)."""
        if (epoch-1) % self.t_m == 0:
            self.current_n_g = self.generator.integers(1, self.n_g_max + 1)
            self.current_clusters = self.kmeans_simple(self.experts, self.current_n_g)

        for g in range(self.current_n_g):
            group_indices = np.where(self.current_clusters == g)[0]
            if len(group_indices) == 0:
                continue

            group_experts = self.experts[group_indices]
            group_fitness = self.fitness[group_indices]

            if self.problem.minmax == "min":
                e_best_g = group_experts[np.argmin(group_fitness)]
            else:
                e_best_g = group_experts[np.argmax(group_fitness)]
            e_avg_g = np.mean(group_experts, axis=0)

            c_i = np.linalg.norm(group_experts - self.current_best.solution, axis=1)
            group_pos = np.array([self.pop_prev[idx].solution for idx in group_indices])
            d_i = np.linalg.norm(group_experts - group_pos, axis=1)
            p_i = np.linalg.norm(group_experts - e_best_g, axis=1)

            r_c = self.normalize(c_i)
            r_d = self.normalize(d_i)
            r_p = self.normalize(p_i)

            for idx, real_idx in enumerate(group_indices):
                b_c = self.generator.uniform(self.b_min, self.b_max)
                b_p = self.generator.uniform(self.b_min, self.b_max)
                b_d = self.generator.uniform(self.b_min, self.b_max)

                e_i = self.experts[real_idx]
                e_r = group_experts[self.generator.integers(len(group_experts))]
                e_u = self.generator.uniform(self.problem.lb, self.problem.ub)

                # K0
                if r_c[idx] <= b_c and r_p[idx] <= b_p:
                    k_0 = r_p[idx] * (e_i + e_r) / 2
                elif r_c[idx] <= b_c and r_p[idx] > b_p:
                    k_0 = r_p[idx] * (e_i + e_avg_g) / 2
                elif r_c[idx] > b_c and r_p[idx] <= b_p:
                    k_0 = r_p[idx] * (e_best_g + e_r) / 2
                else:
                    k_0 = r_p[idx] * (e_best_g + e_avg_g) / 2

                # K1
                c1 = self.generator.random()
                k_1 = c1 * e_avg_g if r_d[idx] <= b_d else c1 * e_u
                k_s1 = np.abs(k_0 + k_1) / 2

                c2 = self.generator.uniform(-1.5, 1.5, self.problem.n_dims)
                e_s1_new1 = e_i + c2 * k_s1

                c3 = self.generator.uniform(-1.5, 1.5)
                c4 = self.generator.random()
                e_s1_new2 = c3 * e_s1_new1 + c4 * e_best_g

                # Boundary clipping & Selection
                e_s1_new1 = self.correct_solution(e_s1_new1)
                e_s1_new2 = self.correct_solution(e_s1_new2)
                agent1 = self.generate_agent(e_s1_new1)
                agent2 = self.generate_agent(e_s1_new2)

                if self.compare_target(agent1.target, agent2.target, self.problem.minmax):
                    best_new = agent1
                else:
                    best_new = agent2
                if self.compare_fitness(best_new.target.fitness, self.fitness[real_idx], self.problem.minmax):
                    self.pop_prev[real_idx] = self.pop[real_idx].copy()
                    self.pop[real_idx] = best_new

    def stage_2_step(self, epoch):
        """Perform a single iteration of Stage 2: Integration."""
        e_avg = np.mean(self.experts, axis=0)

        c_i = np.linalg.norm(self.experts - self.current_best.solution, axis=1)
        pop_prev = np.array([agent.solution for agent in self.pop_prev])
        d_i = np.linalg.norm(self.experts - pop_prev, axis=1)
        p_i = np.linalg.norm(self.experts - self.g_best.solution, axis=1)

        r_c = self.normalize(c_i)
        r_d = self.normalize(d_i)
        r_p = self.normalize(p_i)

        for idx in range(self.pop_size):
            b_c = self.generator.uniform(self.b_min, self.b_max)
            b_p = self.generator.uniform(self.b_min, self.b_max)
            b_d = self.generator.uniform(self.b_min, self.b_max)

            e_i = self.experts[idx]
            e_r = self.experts[self.generator.integers(self.pop_size)]
            e_u = self.generator.uniform(self.problem.lb, self.problem.ub)

            if r_c[idx] <= b_c and r_p[idx] <= b_p:
                k_0 = r_p[idx] * (e_i + e_r) / 2
            elif r_c[idx] <= b_c and r_p[idx] > b_p:
                k_0 = r_p[idx] * (e_i + e_avg) / 2
            elif r_c[idx] > b_c and r_p[idx] <= b_p:
                k_0 = r_p[idx] * (self.g_best.solution + e_r) / 2
            else:
                k_0 = r_p[idx] * (self.g_best.solution + e_avg) / 2

            c5 = self.generator.random()
            k_1 = c5 * e_avg if r_d[idx] <= b_d else c5 * e_u
            k_s2 = np.abs(k_0 + k_1) / 2
            c6 = self.generator.uniform(-0.75, 0.75, self.problem.n_dims)
            e_s2_new1 = e_i + c6 * k_s2

            c7 = self.generator.uniform(-0.75, 0.75)
            c8 = self.generator.random()
            e_s2_new2 = c7 * e_s2_new1 + c8 * self.g_best.solution
            agent1 = self.generate_agent(self.correct_solution(e_s2_new1))
            agent2 = self.generate_agent(self.correct_solution(e_s2_new2))

            if self.compare_target(agent1.target, agent2.target, self.problem.minmax):
                best_new = agent1
            else:
                best_new = agent2
            if self.compare_fitness(best_new.target.fitness, self.fitness[idx], self.problem.minmax):
                self.pop_prev[idx] = self.pop[idx].copy()
                self.pop[idx] = best_new

    def stage_3_step(self, epoch):
        """Perform a single iteration of Stage 3: IbI Logic Search."""
        e_avg = np.mean(self.experts, axis=0)

        for idx in range(self.pop_size):
            e_i = self.experts[idx]
            e_r = self.experts[self.generator.integers(self.pop_size)]
            knowledge_factor = self.generator.choice([1, 2])
            if knowledge_factor == 1:
                k_s3 = np.abs(e_avg - e_r)
            else:
                k_s3 = np.abs(e_avg - self.g_best.solution)
            c9 = self.generator.uniform(-0.25, 0.25, self.problem.n_dims)
            e_s3_new1 = e_i + c9 * k_s3

            c10 = self.generator.uniform(-0.25, 0.25)
            c11 = self.generator.random()
            e_s3_new2 = c10 * e_s3_new1 + c11 * self.g_best.solution
            agent1 = self.generate_agent(self.correct_solution(e_s3_new1))
            agent2 = self.generate_agent(self.correct_solution(e_s3_new2))
            if self.compare_target(agent1.target, agent2.target, self.problem.minmax):
                best_new = agent1
            else:
                best_new = agent2
            if self.compare_fitness(best_new.target.fitness, self.fitness[idx], self.problem.minmax):
                self.pop_prev[idx] = self.pop[idx].copy()
                self.pop[idx] = best_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        if epoch < self.n_t1:
            self.stage_1_step(epoch)
        elif epoch < (self.n_t1 + self.n_t2):
            self.stage_2_step(epoch)
        else:
            self.stage_3_step(epoch)

        self.experts = np.array([agent.solution for agent in self.pop])
        self.fitness = np.array([agent.target.fitness for agent in self.pop])
        self.current_best, _ = self.get_best_agent(self.pop, self.problem.minmax)
