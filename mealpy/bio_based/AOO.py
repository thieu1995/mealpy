#!/usr/bin/env python
# Created by "Thieu" at 16:01, 23/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalAOO(Optimizer):
    """
    The original version of: Animated Oat Optimization Algorithm (AOO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Population size, in range [10, 10000]. Default is 30.

    Links
    -----
    1. Paper: https://doi.org/10.1016/j.knosys.2025.113589
    2. Matlab code: https://seyedalimirjalili.com/morealgorithms

    Note
    ----
    This implementation primarily follows Eqs. (1)-(14) and Algorithm 1
    in the paper. The released MATLAB code contains several differences:

    - Algorithm 1 places the calculation of Eq. (3) inside the individual
      loop, while the MATLAB code samples `m`, `L`, and `e` only once
      before optimization. This implementation follows Algorithm 1.
    - Eq. (3) uses the same random symbol `r` for `m`, `L`, and `e`,
      whereas MATLAB samples `m` and `L` independently and then sets
      `e = m`. This implementation follows Eq. (3) literally.
    - Eq. (12) defines `x = 3*r/dim` as part of the projectile-motion
      operator. MATLAB instead samples one `x` per individual only once
      before optimization. This implementation follows Eq. (12).
    - The paper states that `r'` in Eq. (12) is sampled from `[0, T]`.
      MATLAB uses `randi([0, t])`, i.e., an integer from `[0, t]`.
      The published formulation is followed here.
    - The gravitational coefficient `g` appearing in Eq. (13) is not
      defined numerically in the paper. The MATLAB source uses
      `g = 9.8 / dim`; this source-defined value is therefore adopted.
    - The paper specifies the Levy step in Eqs. (8)-(9), including the
      factor `0.01`, but the supplied MATLAB snippet does not include the
      implementation of `levy()`. This implementation uses the standard
      Mantegna construction implied by Eq. (9), with beta = 1.5 and the
      `0.01` multiplier explicitly given in Eq. (8).
    - Eq. (5) is expressed using generation-t positions `X_t`. MATLAB
      updates `X` in-place, so `mean(X)` for later individuals contains a
      mixture of old and already-updated positions. This implementation
      uses a frozen population snapshot for each iteration, following the
      published equations.
    - The paper does not specify boundary handling. MATLAB clips violated
      components to the corresponding lower or upper bounds. The same
      behavior is used here.
    - Neither the paper nor MATLAB uses greedy parent-child selection.
      All generated positions form the next population; only the global
      best is retained historically.
    - MATLAB evaluates `feval(fhd, X', ...)` and then evaluates every
      individual again in the following loop. If the first call evaluates
      all population members, the source performs approximately twice the
      objective evaluations stated by Algorithm 1 and the paper. This
      redundant call is not reproduced here.

    References
    ----------
    1. Wang, R.B., Hu, R.B., Geng, F.D., Xu, L., Chu, S.C., Pan, J.S., Meng, Z.Y. and Mirjalili, S., 2025.
       The Animated Oat Optimization Algorithm: A nature-inspired metaheuristic for engineering optimization
       and a case study on Wireless Sensor Networks. Knowledge-Based Systems, 318, p.113589.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, AOO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = AOO.OriginalAOO(epoch=1000, pop_size=30)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Animated Oat Optimization Algorithm", year=2025, difficulty="medium", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.QUESTIONABLE_MATH, ScientificConcern.INCORRECT_EQUATIONS,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH
                       ))

    def __init__(self, epoch: int = 1000, pop_size: int = 30, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Population size, default = 30.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations of AOO.

        Args:
            epoch (int): The current iteration.
        """
        # Freeze X_t and X_best during generation, consistent with the
        # generation-index notation in Eqs. (5), (10), and (14).
        pop_old = [agent.copy() for agent in self.pop]
        positions = np.array([agent.solution for agent in pop_old])
        mean_pos = np.mean(positions, axis=0)
        best_pos = self.g_best.solution.copy()

        # Eq. (3)
        c = (1.0 - epoch / self.epoch) ** 3
        # One Levy vector for every individual.
        levy_steps = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
        pop_new = []
        for idx in range(self.pop_size):
            # ----------------------------------------------------------
            # Eq. (3)
            # The same r is used literally for m, L, and e.
            r_param = self.generator.random()
            mass = (0.5 * r_param / self.problem.n_dims)
            awn_length = (self.pop_size * r_param / self.problem.n_dims)
            eccentricity = (0.5 * r_param / self.problem.n_dims)

            # ----------------------------------------------------------
            # Exploration phase -- Eqs. (4)-(5)
            if self.generator.random() > 0.5:
                # Eq. (4)
                w = (c / np.pi * (2.0 * self.generator.random(self.problem.n_dims) - 1.0) * self.problem.ub)
                # Eq. (5) is written using 1-based individual indices.
                i = idx + 1
                interval = self.pop_size / 10.0
                remainder = np.mod(i, interval)
                if np.isclose(remainder, 0.0):
                    pos_new = (mean_pos + w)
                elif np.isclose(remainder, 1.0):
                    pos_new = (best_pos + w)
                else:
                    pos_new = (pop_old[idx].solution + w)
            # ----------------------------------------------------------
            # Exploitation phase
            else:
                # ------------------------------------------------------
                # Hygroscopic rolling -- Eqs. (6)-(10)
                if self.generator.random() > 0.5:
                    r = self.generator.random()
                    # Eq. (6)
                    a = (self.problem.ub - np.abs(self.problem.ub * epoch * np.sin(2.0 * np.pi * r) / self.epoch))
                    # Eq. (7)
                    random_vector = self.generator.uniform(-a, a)
                    rolling = ((mass * eccentricity + awn_length ** 2) / self.problem.n_dims * random_vector)
                    # Eq. (10)
                    pos_new = (best_pos + rolling + c * levy_steps[idx] * best_pos)
                # -----------------------------------------------------
                # Energy-storing ejection -- Eqs. (11)-(14)
                else:
                    r = self.generator.random()
                    # Eq. (11)
                    b = (self.problem.ub - np.abs(self.problem.ub * epoch * np.cos(2.0 * np.pi * r) / self.epoch))
                    # Eq. (12)
                    k = (0.5 + 0.5 * self.generator.random())
                    x = (3.0 * self.generator.random() / self.problem.n_dims)
                    theta = (np.pi * self.generator.random())
                    # The paper states r' is in [0, T].
                    r_prime = self.generator.uniform(0.0, self.epoch)
                    alpha = (np.exp(r_prime / self.epoch) / np.pi)
                    # g is not specified numerically in the paper.
                    # Follow the released MATLAB source.
                    gravity = 9.8 / self.problem.n_dims
                    random_vector = self.generator.uniform(-b, b)
                    # Eq. (13)
                    jump = (2.0 * k * x ** 2 * np.sin(2.0 * theta) / (mass * gravity + self.EPSILON) * (1.0 - alpha) / self.problem.n_dims * random_vector)
                    # Eq. (14)
                    pos_new = (best_pos + jump + c * levy_steps[idx] * best_pos)
            # Boundary treatment is not given in the paper. MATLAB uses clipping.
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                # ----------------------------------------------------------
                # Algorithm 1 updates the generated position directly.
                # It does not specify greedy parent-child survival.
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            # No greedy parent-child comparison is specified.
            self.pop = self.update_target_for_population(pop_new)
