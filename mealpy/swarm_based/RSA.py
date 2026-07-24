#!/usr/bin/env python
# Created by "https://github.com/beratcalik" at 2025
# -------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalRSA(Optimizer):
    """
    The original version of: Reptile Search Algorithm (RSA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    alpha : float
        Current range from (0.0, 100.0).
    beta : float
        Current range from (0.0, 100.0).

    References
    ~~~~~~~~~~
    1. Abualigah, L., Abd Elaziz, M., Sumari, P., Geem, Z. W., & Gandomi, A. H. (2022).
        Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer.
        Expert Systems with Applications, 191, 116158. https://doi.org/10.1016/j.eswa.2021.116158

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, RSA
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
    >>> model = RSA.OriginalRSA(epoch=1000, pop_size=50, alpha=0.1, beta=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Reptile Search Algorithm", year=2022, difficulty="medium", kind="original")

    def __init__(self, epoch=10000, pop_size=100, alpha=0.1, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, (0.0, 100.0))
        self.beta = self.validator.check_float("beta", beta, (0.0, 100.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta"])
        self.sort_flag = False

    def evolve(self, epoch):
        best = self.g_best.solution

        r3 = self.generator.integers(-1, 2)   # {-1, 0, 1}
        ES = 2.0 * r3 * (1.0 - 1.0 / self.epoch)
        pop_new = []
        for i in range(self.pop_size):
            x = self.pop[i].solution
            mx = np.mean(x)

            r1 = self.generator.integers(0, self.pop_size)
            r2 = self.generator.integers(0, self.pop_size)

            x_r1 = self.pop[r1].solution
            x_r2 = self.pop[r2].solution

            rand = self.generator.random(self.problem.n_dims)

            denom = best * (self.problem.ub - self.problem.lb) + self.EPSILON
            P = self.alpha + (x - mx) / denom
            eta = best * P
            R = (best - x_r2) / (best + self.EPSILON)

            if epoch <= self.epoch / 4:                     # High walking
                pos_new = best * (-eta * self.beta - R * rand)
            elif epoch <= 2 * self.epoch / 4:               # Belly walking
                pos_new = best * (x_r1 * ES * rand)
            elif epoch <= 3 * self.epoch / 4:               # Hunting coordination
                pos_new = best * (P * rand)
            else:                              # Hunting cooperation
                pos_new = best - eta * self.EPSILON - R * rand

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[i] = self.get_better_agent(agent, self.pop[i], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
