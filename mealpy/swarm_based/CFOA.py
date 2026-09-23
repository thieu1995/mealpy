#!/usr/bin/env python
# Created by "Thieu" at 16:16, 23/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalCFOA(Optimizer):
    """
    The original version of: Catch Fish Optimization Algorithm (CFOA)

    Parameters
    ----------
    epoch : int
        Maximum number of optimization iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Population size, in range [5, 10000]. Default is 30.


    Links
    -----
    1. Paper: https://doi.org/10.1007/s10586-024-04618-w
    2. Matlab code: https://seyedalimirjalili.com/morealgorithms

    Note
    ----
    This implementation primarily follows Eqs. (1)-(11), Algorithm 1, and
    the procedural description in the paper. Several differences between the
    paper and the released MATLAB code should be noted:

    - CFOA is formulated using the number of fitness evaluations (`EFs`).
      Mealpy is iteration-based, so this implementation maps the current
      evaluation count to `pop_size * epoch`, with the initial population
      counted as the first `pop_size` evaluations.
    - Eq. (8) defines the group target as the arithmetic mean of the group.
      The MATLAB code instead uses a raw-fitness-weighted center. This
      implementation follows Eq. (8).
    - Eq. (9) multiplies the random offset by
      `(1 - 2*EFs/MaxEFs)**2`. The MATLAB source omits the square. The
      published equation is followed here.
    - In independent search, the MATLAB source calculates the reference
      direction using shuffled indices but calculates the Euclidean distance
      using the unshuffled loop indices. This implementation consistently
      uses the current fisherman and its selected reference, as required by Eqs. (5)-(7).
    - The MATLAB code applies greedy parent-child selection before updating
      the population. Such a selection rule is not stated in Eqs. (5)-(11),
      Algorithm 1, or the procedural description. New positions therefore
      replace the current population directly here, while the global best is
      retained historically by Mealpy.
    - The paper does not specify boundary handling. The MATLAB source clips
      violated dimensions to their corresponding bounds, which is adopted
      here.
    - During exploration, the MATLAB source samples one probability `p` for
      the whole population update. The paper and flowchart do not clearly
      specify whether `p` is global or individual. The source behavior is
      retained.
    - Groups contain randomly 3 or 4 fishermen. When fewer than the sampled
      group size remain at the end of the shuffled population, the MATLAB
      source applies independent search to the remaining fisherman. This
      source-defined behavior is retained.
    - Eq. (5) is undefined when all current fitness values are identical.
      In that numerical corner case, `Exp = 0` is used.
    - The original formulation is explicitly written for minimization.
      This implementation therefore supports minimization problems only.

    References
    ----------
    1. Jia, H., Wen, Q., Wang, Y. and Mirjalili, S., 2024.
       Catch fish optimization algorithm: a new human behavior algorithm for solving clustering problems.
       Cluster Computing, 27(9), pp.13295-13332.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CFOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = CFOA.OriginalCFOA(epoch=1000, pop_size=30)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Catch Fish Optimization Algorithm", year=2024, difficulty="medium", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.QUESTIONABLE_MATH, ScientificConcern.INCORRECT_EQUATIONS,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH
                       ))

    def __init__(self, epoch: int = 1000, pop_size: int = 30, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Population size, default = 30.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations of CFOA.

        Args:
            epoch (int): The current iteration.
        """
        positions = np.array([agent.solution for agent in self.pop])
        fitness = np.array([agent.target.fitness for agent in self.pop])

        # The initial population already consumes pop_size evaluations.
        # With epoch iterations, total evaluations are approximately: pop_size * (epoch + 1).
        current_evals = epoch * self.pop_size
        max_evals = (self.epoch + 1) * self.pop_size
        progress = current_evals / max_evals
        pop_new = [agent.copy() for agent in self.pop]

        # ==============================================================
        # Exploration phase
        # ==============================================================
        if progress < 0.5:
            # Eq. (4)
            exponent = (3.0 * current_evals / (2.0 * max_evals))
            alpha = (1.0 - 3.0 * current_evals / (2.0 * max_evals)) ** exponent
            # The MATLAB code samples p once for the whole exploration
            # population update. The paper is not fully explicit here.
            p = self.generator.random()
            # Randomly shuffle all fishermen.
            order = self.generator.permutation(self.pop_size)
            pointer = 0
            fit_max = np.max(fitness)
            fit_min = np.min(fitness)
            fit_range = fit_max - fit_min
            while pointer < self.pop_size:
                # Paper/source: random groups of 3 or 4.
                group_size = int(self.generator.integers(3, 5))
                remaining = (self.pop_size - pointer)
                # ======================================================
                # Independent search -- Eqs. (5)-(7)
                if (p < alpha or remaining < group_size):
                    idx = int(order[pointer])
                    # Select a reference fisherman different from idx.
                    candidates = np.delete(np.arange(self.pop_size), idx, )
                    ref_idx = int(self.generator.choice(candidates))
                    # Eq. (5)
                    if fit_range <= self.EPSILON:
                        exp_value = 0.0
                    else:
                        exp_value = (fitness[idx] - fitness[ref_idx]) / fit_range
                    current_pos = positions[idx]
                    ref_pos = positions[ref_idx]
                    difference = (ref_pos - current_pos)
                    # Euclidean distance Dis in Eq. (6)
                    distance = np.linalg.norm(difference)
                    # Random unit vector s.
                    random_direction = (2.0 * self.generator.random(self.problem.n_dims) - 1.0)
                    direction_norm = np.linalg.norm(random_direction)
                    if direction_norm <= self.EPSILON:
                        unit_direction = np.zeros(self.problem.n_dims)
                    else:
                        unit_direction = (random_direction / direction_norm)
                    # rs in Eq. (7)
                    random_scale = (self.generator.random())
                    # Eq. (6)
                    radius = (distance * np.sqrt(np.abs(exp_value)) * (1.0 - progress))
                    # Eq. (7)
                    pos_new = (current_pos + difference * exp_value + random_scale * unit_direction * radius)
                    pos_new = self.correct_solution(pos_new)
                    pop_new[idx] = self.generate_empty_agent(pos_new)
                    pointer += 1
                # ======================================================
                # Group capture -- Eqs. (8)-(9)
                # ======================================================
                else:
                    group_indices = order[pointer:pointer + group_size]
                    group_positions = positions[group_indices]
                    # Eq. (8)
                    centre = np.mean(group_positions, axis=0)
                    # r2 varies from fisherman to fisherman.
                    r2 = self.generator.random((group_size, 1))
                    # r3 is an offset in (-1, 1).
                    r3 = (2.0 * self.generator.random((group_size, self.problem.n_dims,)) - 1.0)
                    # Eq. (9)
                    offset_scale = (1.0 - 2.0 * progress) ** 2
                    new_positions = (group_positions + r2 * (centre - group_positions) + offset_scale * r3)
                    for local_idx, idx in enumerate(group_indices):
                        pos_new = self.correct_solution(new_positions[local_idx])
                        pop_new[int(idx)] = self.generate_empty_agent(pos_new)
                    pointer += group_size
        # ==============================================================
        # Exploitation phase: Collective capture -- Eqs. (10)-(11)
        # ==============================================================
        else:
            remaining_ratio = (1.0 - progress)
            # Eq. (10)
            sigma = np.sqrt((2.0 * remaining_ratio) / (remaining_ratio ** 2 + 1.0))
            mean_position = np.mean(positions, axis=0)
            best_position = (self.g_best.solution.copy())
            distance_to_mean = np.abs(best_position - mean_position)
            for idx in range(self.pop_size):
                # r4 belongs to {1, 2, 3}.
                r4 = int(self.generator.integers(1, 4))
                # Standard deviation vector from Eq. (11).
                std = (distance_to_mean * (r4 / 3.0) * sigma)
                pos_new = best_position + self.generator.normal(0.0, std, self.problem.n_dims)
                pos_new = self.correct_solution(pos_new)
                pop_new[idx] = self.generate_empty_agent(pos_new)
        # The published algorithm does not specify greedy parent-child selection.
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(self.pop_size):
                pop_new[idx].target = self.get_target(pop_new[idx].solution)
                self.pop[idx] = pop_new[idx]
        else:
            self.pop = self.update_target_for_population(pop_new)
