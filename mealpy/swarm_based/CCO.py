import numpy as np
from math import gamma
from mealpy.optimizer import Optimizer


class OriginalCCO(Optimizer):
    """
    The original version of: Cuckoo-Catfish Optimizer (CCO)

    Notes:
        - Mealpy conventions:
          + Uses ``self.generator`` for all stochasticity (reproducible via seed)
          + Uses ``self.correct_solution`` for boundary handling
          + Uses ``self.compare_target`` for min/max-safe selection

    References:
        Wang, T. L., Gu, S. W., Liu, R. J., Chen, L. Q., Wang, Z., & Zeng, Z. Q. (2025).
        Cuckoo Catfish Optimizer: A New Meta-Heuristic Optimization Algorithm.
        Artificial Intelligence Review. DOI: 10.1007/s10462-025-11291-x

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.swarm_based.CCO import OriginalCCO
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
    >>> model = OriginalCCO(epoch=1000, pop_size=50, alpha=0.5, beta=1.0)
    >>> best = model.solve(problem_dict)
    >>> print(best.solution, best.target.fitness)
    """

    def __init__(self, epoch=10000, pop_size=100, alpha=0.5, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 1000000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 100000])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 10.0))
        self.beta = self.validator.check_float("beta", beta, (0, 10.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta"])
        self.sort_flag = False

        # Stagnation counter for catfish mechanism
        self.t_counter = 0

    def _levy_flight(self, n, m, beta=1.5):
        """
        Generate Lévy flight steps using Mantegna's algorithm.

        Args:
            n (int): Number of step vectors
            m (int): Dimension
            beta (float): Lévy exponent

        Returns:
            np.ndarray: (n, m) Lévy steps
        """
        num = gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
        den = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
        sigma_u = (num / den) ** (1.0 / beta)

        u = self.generator.normal(0.0, sigma_u, (n, m))
        v = self.generator.normal(0.0, 1.0, (n, m))
        return 0.05 * u / (np.abs(v) ** (1.0 / beta))

    def evolve(self, epoch):
        """
        The main evolution step called by the Mealpy framework.
        """
        rng = self.generator
        epoch_i = int(epoch)
        epoch_safe = max(epoch_i, 1)  # prevent division by zero

        pop_pos = np.asarray([agent.solution for agent in self.pop])
        pop_fits = np.asarray([agent.target.fitness for agent in self.pop])
        best_x = self.g_best.solution

        # Time-dependent parameters
        C = 1.0 - (epoch_i / self.epoch)
        T = (1.0 - np.sin((np.pi * epoch_i) / (2.0 * self.epoch))) ** (epoch_i / self.epoch)

        # Dynamic probability for catfish mechanism (die probability)
        if self.t_counter < 15:
            die = 0.02 * T
        else:
            die = 0.02
            C = 0.8

        # Spiral path vectors
        indices = np.arange(1, self.pop_size + 1)
        theta = (1.0 - 10.0 * indices / self.pop_size) * np.pi
        r_spiral = self.alpha * np.exp(self.beta * theta / 3.0)
        x_spiral = r_spiral * np.cos(theta)
        y_spiral = r_spiral * np.sin(theta)

        # Lévy steps
        levy_steps = self._levy_flight(self.pop_size, self.problem.n_dims, beta=1.5)

        new_pop = []
        for i in range(self.pop_size):
            agent = self.pop[i]
            pos_i = agent.solution.copy()

            # Stochastic switch
            Q = 0
            if (epoch_i / self.epoch) < rng.random():
                Q = 1

            # Random distinct indices
            rand_idx = rng.choice(self.pop_size, 3, replace=False)
            pop_pos_rand = pop_pos[rand_idx]
            pop_fit_rand = pop_fits[rand_idx]

            # Random coefficient and distance metric
            wRand = rng.random() * (2.0 - 2.0 * epoch_i / self.epoch)
            Dis = rng.random() * (epoch_i / self.epoch) ** 2

            # Random vectors
            F = rng.random(self.problem.n_dims)
            R1 = rng.random(self.problem.n_dims)

            # E, j parameters
            E = 0.3 if (epoch_i / self.epoch) > rng.random() else 0.9
            j_val = 1.0 if (epoch_i / self.epoch) < rng.random() else 2.0

            if rng.random() < 0.5:
                U = 2.0 * rng.random(self.problem.n_dims)
                V = 0.0
            else:
                U = 0.0
                V = 2.0 * rng.random(self.problem.n_dims)

            if Q == 1:
                # Strategy A: Levy-based exploration around best
                step = levy_steps[i] * (best_x - pos_i)
                new_pos = best_x + step * C * (1.0 - 2.0 * rng.random(self.problem.n_dims))
            else:
                if rng.random() < 0.5:
                    # Catfish disturbance / local randomization
                    if rng.random() < die:
                        new_pos = rng.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
                    else:
                        mean_pos = np.mean(pop_pos, axis=0)
                        step = wRand * (mean_pos - pos_i) + (1.0 - wRand) * (best_x - pos_i)
                        new_pos = pos_i + C * step * (2.0 * rng.random(self.problem.n_dims) - 1.0)
                else:
                    if rng.random() < rng.random():
                        # Strategy C: Chaotic local search
                        if j_val < Dis:
                            mean_pos = np.mean(pop_pos, axis=0)
                            V2 = 2.0 * (rng.random() * (mean_pos - pos_i) + rng.random() * (best_x - pos_i))
                        else:
                            V2 = 2.0 * (rng.random() * (pop_pos_rand[1] - pop_pos_rand[2]) +
                                        rng.random() * (pop_pos_rand[0] - pos_i))

                        # Compare with first sampled peer (keeps shapes consistent)
                        r = 0
                        fit_i = agent.target.fitness
                        step_base = pos_i if fit_i <= pop_fit_rand[r] else pop_pos_rand[r]
                        target_base = pop_pos_rand[r] if fit_i <= pop_fit_rand[r] else pos_i

                        step = step_base - E * target_base
                        factor = y_spiral[i] if ((i + 1) % 2 == 1) else x_spiral[i]
                        new_pos = (step_base + (T ** 2) * factor *
                                   (R1 * (best_x - step_base) + (1.0 - R1) * np.abs(step))
                                   + F * R1 * step / 2.0
                                   + V2 * j_val / epoch_safe)
                    else:
                        # Strategy D: Top-k guided update (top3 + mean)
                        if self.problem.minmax == "min":
                            sorted_indices = np.argsort(pop_fits)
                        else:
                            sorted_indices = np.argsort(-pop_fits)

                        top3 = pop_pos[sorted_indices[:3]]
                        mean_pos = np.mean(pop_pos, axis=0)
                        D = np.vstack((top3, mean_pos))
                        B = D[rng.integers(0, 4)]

                        Rt1 = rng.permutation(360)[:self.problem.n_dims] * np.pi / 360.0
                        Rt2 = rng.permutation(360)[:self.problem.n_dims] * np.pi / 360.0
                        w = 1.0 - ((np.exp(epoch_i / self.epoch) - 1.0) / (np.exp(1.0) - 1.0)) ** 2

                        rand_trig = rng.random()
                        if rand_trig < 0.33:
                            new_pos = B + 2.0 * w * F * np.cos(Rt1) * np.sin(Rt2) * (B - pos_i)
                        elif rand_trig < 0.66:
                            new_pos = B + 2.0 * w * F * np.sin(Rt1) * np.cos(Rt2) * (B - pos_i)
                        else:
                            new_pos = B + 2.0 * w * F * np.cos(Rt1) * np.cos(Rt2) * (B - pos_i)

            # Mealpy boundary handling + evaluation
            new_pos = self.correct_solution(new_pos)
            new_agent = self.generate_agent(new_pos)

            # Greedy selection (min/max safe)
            if self.compare_target(new_agent.target, agent.target, self.problem.minmax):
                new_pop.append(new_agent)
                self.t_counter = 0
            else:
                new_pop.append(agent)
                self.t_counter += 1

        self.pop = new_pop
