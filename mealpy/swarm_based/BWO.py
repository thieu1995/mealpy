#!/usr/bin/env python
#
# Beluga Whale Optimization (BWO)
#
# Author: Enes Cabbar AKÇA (Github : enescabbarakca29)
#
# Reference:
# Zhong, C., Li, G., & Meng, Z. (2022). Beluga whale optimization: A novel nature‑inspired metaheuristic algorithm.
# Knowledge‑Based Systems, 251, 109215.
# https://doi.org/10.1016/j.knosys.2022.109215
#
# Beluga Whale Optimization (BWO) - Paper-faithful (Eq. 3–10)
#
# Notes:
# - Exploration (Eq.4) implemented per-dimension with separate pj and p1 (critical fix).
# - Whale fall (Eq.8–10) uses probability Wf and greedy acceptance.
# - Levy flight (Eq.6–7) with beta=1.5 and scale=0.05.
#
# Usage:
# >>> from mealpy import FloatVar, BWO
# >>> model = BWO.OriginalBWO(epoch=1000, pop_size=50)
# >>> g_best = model.solve(problem_dict)

from __future__ import annotations

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBWO(Optimizer):
    """
    The original version of: Beluga Whale Optimization (BWO)

    Paper-faithful core (Eq. 3–10):
        - Exploration (Eq.4)
        - Exploitation with Lévy flight (Eq.5–7)
        - Whale fall strategy (Eq.8–10)

    Hyper-parameters:
        epoch (int): maximum number of iterations
        pop_size (int): population size
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        # validate and store parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        # BWO does not sort population between iterations
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration (starts from 0 or 1 depending on Mealpy loop)
        """
        n = self.pop_size
        d = self.problem.n_dims

        # Ensure lb/ub are vectors (Mealpy commonly stores arrays)
        lb = np.asarray(self.problem.lb, dtype=float)
        ub = np.asarray(self.problem.ub, dtype=float)

        # Eq. (3): Bf = B0*(1 - T/(2*Tmax)), with B0 in (0,1) per individual
        B0 = self.generator.random(n)
        # NOTE: Mealpy's `epoch` can be 0-based in some loops; paper is 1..Tmax.
        # Using current `epoch` directly keeps consistent with other Mealpy implementations.
        Bf = B0 * (1.0 - epoch / (2.0 * self.epoch))

        pop_new = []

        # -------------------------
        # Main move: Eq.4 or Eq.5
        # -------------------------
        for idx in range(n):
            # choose r != idx
            r = int(self.generator.integers(0, n))
            while r == idx:
                r = int(self.generator.integers(0, n))

            x_i = self.pop[idx].solution
            x_r = self.pop[r].solution
            x_best = self.g_best.solution

            # Start from copy
            pos_new = x_i.copy()

            if Bf[idx] > 0.5:
                # ===================== Exploration (Eq. 4) =====================
                r1, r2 = self.generator.random(2)

                for j in range(d):
                    # per-dimension pj and separate p1 (critical)
                    pj = int(self.generator.integers(0, d))
                    p1 = int(self.generator.integers(0, d))

                    base = x_i[pj]
                    diff = x_r[p1] - base

                    # even/odd w.r.t 1-indexed j (paper)
                    trig = (np.sin if ((j + 1) % 2 == 0) else np.cos)(2.0 * np.pi * r2)
                    pos_new[j] = base + diff * (1.0 + r1) * trig
            else:
                # ===================== Exploitation (Eq. 5) =====================
                r3, r4 = self.generator.random(2)
                C1 = 2.0 * r4 * (1.0 - epoch / self.epoch)
                # Use built‑in Levy-flight function from Optimizer
                # Beta=1.5 (Eq.6), multiplier=0.05 (scale), case=-1 returns multiplier * s only
                LF = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=d, case=-1)

                pos_new = r3 * x_best - r4 * x_i + C1 * LF * (x_r - x_i)

            # Bound control (Mealpy way)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            # In sequential modes: evaluate and greedy-update immediately
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        # In parallel modes: evaluate batch and greedy-select
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # -------------------------
        # Whale fall (Eq. 8–10)
        # -------------------------
        # Eq. (10): Wf = 0.1 - 0.05*T/Tmax
        Wf = 0.1 - 0.05 * (epoch / self.epoch)
        # Eq. (9): Xstep = (ub - lb)*exp(-C2*T/Tmax), C2 = 2*Wf*n
        C2 = 2.0 * Wf * n
        X_step = (ub - lb) * np.exp(-C2 * epoch / self.epoch)

        # Sequential mode: evaluate per candidate
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(n):
                if float(self.generator.random()) < Wf:
                    r = int(self.generator.integers(0, n))
                    while r == idx:
                        r = int(self.generator.integers(0, n))

                    r5, r6, r7 = self.generator.random(3)
                    x_i = self.pop[idx].solution
                    x_r = self.pop[r].solution

                    cand = r5 * x_i - r6 * x_r + r7 * X_step
                    cand = self.correct_solution(cand)

                    agent_cand = self.generate_empty_agent(cand)
                    agent_cand.target = self.get_target(cand)
                    self.pop[idx] = self.get_better_agent(agent_cand, self.pop[idx], self.problem.minmax)

        # Parallel mode: batch-evaluate only the whale-fall candidates
        else:
            idxs = []
            cand_agents = []
            for idx in range(n):
                if float(self.generator.random()) < Wf:
                    r = int(self.generator.integers(0, n))
                    while r == idx:
                        r = int(self.generator.integers(0, n))

                    r5, r6, r7 = self.generator.random(3)
                    x_i = self.pop[idx].solution
                    x_r = self.pop[r].solution

                    cand = r5 * x_i - r6 * x_r + r7 * X_step
                    cand = self.correct_solution(cand)

                    idxs.append(idx)
                    cand_agents.append(self.generate_empty_agent(cand))

            if len(cand_agents) > 0:
                cand_agents = self.update_target_for_population(cand_agents)
                for k, idx in enumerate(idxs):
                    self.pop[idx] = self.get_better_agent(cand_agents[k], self.pop[idx], self.problem.minmax)
