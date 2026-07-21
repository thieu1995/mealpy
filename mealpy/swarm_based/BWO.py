#!/usr/bin/env python
# Created by "Enes Cabbar AKÇA" in 2024
# Github: https://github.com/enescabbarakca29
# ------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBWO(Optimizer):
    """
    The original version of: Beluga Whale Optimization (BWO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.

    References
    ~~~~~~~~~~
    1. Zhong, Changting, Gang Li, and Zeng Meng. "Beluga whale optimization: A novel nature-inspired metaheuristic
       algorithm." Knowledge-based systems 251 (2022): 109215. https://doi.org/10.1016/j.knosys.2022.109215

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BWO
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
    >>> model = BWO.OriginalBWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration (starts from 0 or 1 depending on Mealpy loop)
        """
        # Eq. (3): Bf = B0*(1 - T/(2*Tmax)), with B0 in (0,1) per individual
        B0 = self.generator.random(self.pop_size)
        Bf = B0 * (1.0 - epoch / (2.0 * self.epoch))
        ndim = self.problem.n_dims

        pop_new = []
        # -------------------------
        # Main move: Eq.4 or Eq.5
        # -------------------------
        for idx in range(self.pop_size):
            rr = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos_rr = self.pop[rr].solution
            pos_ii = self.pop[idx].solution
            if Bf[idx] > 0.5:
                # ===================== Exploration (Eq. 4) =====================
                r1, r2 = self.generator.random(2)
                pj = self.generator.integers(0, ndim, size=ndim)
                p1 = self.generator.integers(0, ndim, size=ndim)
                base = pos_ii[pj]
                diff = pos_ii[p1] - base
                j_indices = np.arange(1, ndim + 1)
                sin_val = np.sin(2.0 * np.pi * r2)
                cos_val = np.cos(2.0 * np.pi * r2)
                trig = np.where(j_indices % 2 == 0, sin_val, cos_val)
                pos_new = base + diff * (1.0 + r1) * trig
            else:
                # ===================== Exploitation (Eq. 5) =====================
                r3, r4 = self.generator.random(2)
                C1 = 2.0 * r4 * (1.0 - epoch / self.epoch)
                # Use built‑in Levy-flight function from Optimizer
                # Beta=1.5 (Eq.6), multiplier=0.05 (scale), case=-1 returns multiplier * s only
                LF = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=ndim, case=-1)
                pos_new = r3 * self.g_best.solution - r4 * pos_ii + C1 * LF * (pos_rr - pos_ii)
            # Bound control
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
        C2 = 2.0 * Wf * self.pop_size
        X_step = (self.problem.ub - self.problem.lb) * np.exp(-C2 * epoch / self.epoch)

        # Sequential mode: evaluate per candidate
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(self.pop_size):
                if float(self.generator.random()) < Wf:
                    rr = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                    r5, r6, r7 = self.generator.random(3)
                    cand = r5 * self.pop[idx].solution - r6 * self.pop[rr].solution + r7 * X_step
                    cand = self.correct_solution(cand)
                    agent_cand = self.generate_agent(cand)
                    self.pop[idx] = self.get_better_agent(agent_cand, self.pop[idx], self.problem.minmax)

        # Parallel mode: batch-evaluate only the whale-fall candidates
        else:
            idxs = []
            cand_agents = []
            for idx in range(self.pop_size):
                if float(self.generator.random()) < Wf:
                    rr = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                    r5, r6, r7 = self.generator.random(3)
                    cand = r5 * self.pop[idx].solution - r6 * self.pop[rr].solution + r7 * X_step
                    cand = self.correct_solution(cand)
                    idxs.append(idx)
                    cand_agents.append(self.generate_empty_agent(cand))

            if len(cand_agents) > 0:
                cand_agents = self.update_target_for_population(cand_agents)
                for k, idx in enumerate(idxs):
                    self.pop[idx] = self.get_better_agent(cand_agents[k], self.pop[idx], self.problem.minmax)
