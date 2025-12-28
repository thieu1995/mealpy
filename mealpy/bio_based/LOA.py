#!/usr/bin/env python
# Created for Mealpy-style implementation of LOA (Lyrebird Optimization Algorithm)
# Created by "MehdiSndg" at 2025
#       Email: mehdisindag@gmail.com
#       Github: https://github.com/MehdiSndg
# Based on: Dehghani et al., Biomimetics 2023 (LOA)

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalLOA(Optimizer):
    """
    The original version of: Lyrebird Optimization Algorithm (LOA)

    Paper operators (summary):
        - Pick phase by rp:
            Phase 1 (Escape / Exploration) if rp <= 0.5
            Phase 2 (Hide / Exploitation) otherwise
        - Phase 1 uses a randomly selected "safe area" (better solution) SSA
          and random I in {1,2} per dimension.
        - Phase 2 uses shrinking step (ub-lb)/t.

    References (equations):
        - Phase choice (Eq. 4)
        - Safe areas (Eq. 5)
        - Escape update (Eq. 6) + replacement rule (Eq. 7)
        - Hide update (Eq. 8) + replacement rule (Eq. 9)

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, LOA
    >>>
    >>> def objective_function(x):
    >>>     return np.sum(x**2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = LOA.OriginalLOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem)
    >>> print(g_best.solution, g_best.target.fitness)
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default=10000
            pop_size (int): population size, default=100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True  # Mealpy will keep population sorted by fitness each iteration

    def initialize_variables(self):
        # cache bounds range for Phase 2
        self.lb = np.array(self.problem.lb, dtype=float)
        self.ub = np.array(self.problem.ub, dtype=float)
        self.bw = self.ub - self.lb  # bound width vector

    def _pick_safe_area(self, idx: int):
        """
        Safe areas for ith member: all members with better fitness than it (Eq. 5 in min form).
        Since sort_flag=True, better members appear before idx in self.pop.
        """
        if idx <= 0:
            return None
        safe_idx = self.generator.integers(0, idx)  # pick one from [0, idx-1]
        return self.pop[safe_idx].solution

    def evolve(self, epoch: int):
        """
        Main LOA operators. Inherit from Optimizer class

        Args:
            epoch (int): current iteration index (usually 0..epoch-1 in Mealpy)
        """
        t = epoch + 1  # LOA paper uses iteration counter starting from 1

        pop_new = []
        for idx in range(self.pop_size):
            x = self.pop[idx].solution

            rp = self.generator.random()  # rp in [0,1]
            if rp <= 0.5:
                # -------- Phase 1: Escape (Exploration) --------
                ssa = self._pick_safe_area(idx)

                if ssa is None:
                    # Fallback: if no safe area exists (e.g., best agent), re-randomize to keep exploration
                    pos_new = self.problem.generate_solution()
                else:
                    r = self.generator.random(self.problem.n_dims)  # ri,j in [0,1]
                    I = self.generator.integers(1, 3, size=self.problem.n_dims)  # Ii,j ∈ {1,2}
                    # Eq. (6): x_new = x + r * (SSA - I*x)
                    pos_new = x + r * (ssa - I * x)
            else:
                # -------- Phase 2: Hide (Exploitation) --------
                r = self.generator.random(self.problem.n_dims)  # ri,j in [0,1]
                # Eq. (8): x_new = x + (1 - 2r) * (ub-lb)/t
                pos_new = x + (1.0 - 2.0 * r) * (self.bw / t)

            # Bound handling + greedy selection like other Mealpy optimizers (e.g., DevSMA)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)