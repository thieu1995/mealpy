#!/usr/bin/env python
# Created by "Thieu" at 14:58, 23/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalEGO(Optimizer):
    """
    The original version of: Eel and Grouper Optimizer (EGO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The main experiments use 500 iterations. Default is 500.
    pop_size : int
        Number of search agents, in range [5, 10000].
        The main experiments use 30 search agents. Default is 30.

    Links
    -----
    1. Paper: https://doi.org/10.1007/s10586-024-04545-w
    2. Matlab code: https://nl.mathworks.com/matlabcentral/fileexchange/164316-the-eel-and-grouper-optimizer

    Note
    ----
    This implementation primarily follows Eqs. (1)-(12) and the algorithmic
    description in the paper. The released MATLAB code is used only where
    the paper does not provide sufficient implementation details.

    Several differences and ambiguities exist between the paper and the
    released MATLAB implementation:

    - Eq. (1) uses
      `|X_i - C2 * X_rand|`, while the MATLAB code implements
      `|C2 * X_i - X_rand|`. This implementation follows Eq. (1).
    - Eq. (4) defines `C2 = 2*r2`, whereas the MATLAB code uses
      `C2 = 2*r1`. This implementation follows Eq. (4).
    - The paper defines `X_rand` as a random position vector. The MATLAB
      code selects a new random population member separately for every
      dimension. Here, one random agent is selected for the whole vector.
    - Algorithm 1 and the flowchart omit the exploration update in Eq. (1),
      although Sect. 3.2.1 explicitly defines it and the MATLAB source
      executes it. Eq. (1) is included here.
    - Eq. (2) uses a `>` fitness comparison even though the experiments and
      MATLAB implementation solve minimization problems. Mealpy's `minmax`
      semantics are used instead.
    - The paper denotes `C1` and `C2` as coefficient vectors but describes
      `r1` and `r2` as random numbers. The MATLAB code uses scalar values
      shared by all dimensions. This implementation follows that source-backed
      interpretation.
    - The coefficient `b` appearing in Eq. (10) is not defined anywhere in
      the mathematical formulation. The MATLAB source uses `b = a*r2`, which
      is therefore adopted here.
    - Eq. (7) defines `r4 = 100*rand`. The MATLAB code reuses `r2`, making
      `r4`, `C2`, and `b` statistically dependent. Here, an independent
      random value is used as stated in Eq. (7).
    - Eq. (8) only defines the eel position when the grouper successfully
      signals it. The paper merely states that otherwise the eel moves to a
      new random position. The MATLAB implementation uses
      `C2 * X_random_agent`; this behavior is retained because no alternative
      equation is provided.
    - Eqs. (10) and (11) use distances
      `|X_eel - X_i|` and `|X_grouper - X_i|`, respectively. The MATLAB code
      instead inserts `C2` inside both distances. The published equations are
      followed here.
    - Eq. (12) uses a single probability `p` to choose between its two
      branches. Algorithm 1 also updates `p` once per agent. The MATLAB code
      redraws `p` separately for every dimension. This implementation uses
      one `p` per agent.
    - The MATLAB code internally changes `Max_iter` to `Max_iter + 1`.
      Consequently, `a` never reaches exactly zero and the starvation rate
      never reaches exactly 100%. The paper defines both schedules using the
      actual maximum iteration count; the paper formulation is followed here.
    - Both the MATLAB source and the procedural description evaluate an
      intermediate position after Eq. (1) and may immediately update the
      grouper before the cooperative attack. This produces approximately
      `2 * pop_size` new objective evaluations per iteration.
    - Search agents are replaced by the generated positions without greedy
      parent-child selection. Only the best-so-far grouper is retained
      historically.
    - The paper performs boundary correction after the cooperative update.
      Intermediate Eq. (1) positions are therefore evaluated before the final
      boundary correction in the original procedure. This implementation
      preserves that ordering.

    References
    ----------
    1. Mohammadzadeh, A., & Mirjalili, S. (2024). Eel and grouper optimizer: a nature-inspired optimization algorithm.
       Cluster Computing, 27(9), 12745-12786.
       https://doi.org/10.1007/s10586-024-04545-w

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, EGO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = EGO.OriginalEGO(epoch=500, pop_size=30)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Eel and Grouper Optimizer", year=2024, difficulty="medium", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.QUESTIONABLE_MATH, ScientificConcern.INCORRECT_EQUATIONS,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.FABRICATED_RESULTS,
                           ScientificConcern.AMBIGUOUS_METHODOLOGY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH,
                       ))

    def __init__(self, epoch: int = 500, pop_size: int = 30, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 500.
            pop_size (int): Population size, default = 30.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch):
        """
        The main operations of EGO.

        Args:
            epoch (int): The current iteration.
        """
        # Eq. (5): decreases from approximately 2 to 0.
        a = 2.0 - 2.0 * epoch / self.epoch
        # Eq. (9): increases from approximately 0 to 100.
        starvation_rate = 100.0 * epoch / self.epoch
        for idx in range(self.pop_size):
            # ----------------------------------------------------------
            # Random coefficients, Eqs. (3)-(7)
            r1 = self.generator.random()
            r2 = self.generator.random()
            c1 = 2.0 * a * r1 - a  # Eq. (3)
            c2 = 2.0 * r2  # Eq. (4)
            r3 = (a - 2.0) * r1 + 2.0  # Eq. (6)
            r4 = 100.0 * self.generator.random()  # Eq. (7)

            # Eq. (10) contains b, but the paper does not define it.
            # The released MATLAB source uses b = a * r2.
            b = a * r2
            # ----------------------------------------------------------
            # Tracking the prey by grouper -- Eq. (1)
            # The paper defines X_rand as one random position vector.
            # ----------------------------------------------------------
            rand_idx = int(self.generator.integers(self.pop_size))
            x_rand = self.pop[rand_idx].solution
            pos_grouper = (x_rand + c1 * np.abs(self.pop[idx].solution - c2 * x_rand))
            pos_grouper = self.correct_solution(pos_grouper)
            grouper_agent = self.generate_agent(pos_grouper)
            # Eq. (2): retain the better solution.
            if self.compare_target(grouper_agent.target, self.g_best.target, self.problem.minmax):
                self.g_best = grouper_agent.copy()

            # The search agent itself moves to the Eq. (1) position.
            self.pop[idx] = grouper_agent

            # ----------------------------------------------------------
            # Signal to moray eel -- Eqs. (8)-(9)
            if r4 <= starvation_rate:
                # Eq. (8)
                eel_pos = c2 * self.g_best.solution
            else:
                # The paper gives no explicit equation for this case.
                # Follow the released MATLAB source.
                rand_idx = int(self.generator.integers(self.pop_size))
                eel_pos = (c2 * self.pop[rand_idx].solution)

            # ----------------------------------------------------------
            # Cooperative attack -- Eqs. (10)-(12)
            # XP is interpreted as the current search-agent position,
            # consistent with the role used by the released algorithm.
            prey_pos = self.pop[idx].solution
            # Eq. (10)
            distance_eel = np.abs(eel_pos - prey_pos)
            x1 = (np.exp(b * r3) * np.sin(2.0 * np.pi * r3) * c1 * distance_eel + eel_pos)
            # Eq. (11)
            distance_grouper = np.abs(self.g_best.solution - prey_pos)
            x2 = (self.g_best.solution + c1 * distance_grouper)
            # Eq. (12)
            if self.generator.random() < 0.5:
                pos_new = (0.8 * x1 + 0.2 * x2) / 2.0
            else:
                pos_new = (0.2 * x1 + 0.8 * x2) / 2.0

            # Boundary handling is placed after Eq. (12), matching the flowchart and released implementation.
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            # The original algorithm replaces the current search agent unconditionally.
            self.pop[idx] = agent
            if self.compare_target(agent.target, self.g_best.target, self.problem.minmax):
                self.g_best = agent.copy()
