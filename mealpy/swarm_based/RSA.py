#!/usr/bin/env python
# --------------------------------------------------%
# Created by contributor for MEALPY
# Reptile Search Algorithm (RSA)
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRSA(Optimizer):
    """
    The original version of: Reptile Search Algorithm (RSA)

    Links:
        https://doi.org/10.1016/j.eswa.2021.116158

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Abd Elaziz, M., Sumari, P., Geem, Z. W., & Gandomi, A. H. (2022).
        Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer.
        Expert Systems with Applications, 191, 116158.
    """

    def __init__(self, epoch=10000, pop_size=100,
                 alpha=0.1, beta=0.1, eps=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, (0.0, 10.0))
        self.beta = self.validator.check_float("beta", beta, (0.0, 10.0))
        self.eps = self.validator.check_float("eps", eps, (1e-30, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta", "eps"])
        self.sort_flag = False

    def evolve(self, epoch):
        best = self.g_best.solution
        T = self.epoch
        t = epoch + 1

        r3 = self.generator.integers(-1, 2)   # {-1, 0, 1}
        ES = 2.0 * r3 * (1.0 - 1.0 / T)

        lb = self.problem.lb
        ub = self.problem.ub

        pop_new = []
        for i in range(self.pop_size):
            x = self.pop[i].solution
            mx = np.mean(x)

            r1 = self.generator.integers(0, self.pop_size)
            r2 = self.generator.integers(0, self.pop_size)

            x_r1 = self.pop[r1].solution
            x_r2 = self.pop[r2].solution

            rand = self.generator.random(self.problem.n_dims)

            denom = best * (ub - lb) + self.eps
            P = self.alpha + (x - mx) / denom
            eta = best * P
            R = (best - x_r2) / (best + self.eps)

            if t <= T / 4:                     # High walking
                pos_new = best * (-eta * self.beta - R * rand)
            elif t <= 2 * T / 4:               # Belly walking
                pos_new = best * (x_r1 * ES * rand)
            elif t <= 3 * T / 4:               # Hunting coordination
                pos_new = best * (P * rand)
            else:                              # Hunting cooperation
                pos_new = best - eta * self.eps - R * rand

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[i] = self.get_better_agent(
                    agent, self.pop[i], self.problem.minmax
                )

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )
