#!/usr/bin/env python
# Created by "Thieu" at 09:51, 16/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalSRA(Optimizer):
    """
    The original version of: Schrodinger Optimizer (SRA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of search agents, in range [5, 10000].
        The paper uses 30 agents in the numerical experiments. Default is 50.

    Note
    ----
    This implementation primarily follows the mathematical equations and
    Algorithm 1 in the paper. The authors' released Python code differs from
    the published algorithm in several important places:

    - The paper uses `p[i] >= TF(t)` for wave-like exploration, where
      `TF(t) = (t / T)^3`. The source computes `TF(t)` but does not use it;
      instead, it switches phases using `p[i] >= 0.5`.
    - The paper states that `zc` decreases linearly from 1 to 0. The source
      instead uses `b = 1 - (t/T)^(1/5)` and samples a random vector from
      `[-b, b]`.
    - Eq. (19) defines `psi(x) = sin(x)`. The source initializes psi as
      `sqrt(2/L) * sin(x) * exp(2)` and later updates it as
      `sin(rand * x)`.
    - The source implementations of Eqs. (20)-(21) use different signs and
      coefficients for the wave-function terms than those printed in the
      paper.
    - Eq. (24) uses `x_best - randn * (x_rand1 - x_rand2)`. The source
      replaces it with a Levy-flight-based equation not given in the paper.
    - Eq. (25) specifies a standard-normal random variable, whereas the
      source performs uniform random reinitialization inside the bounds.
    - The paper derives `h = h0 / (2*pi)`, while the source sets
      `h = 6.625e-34`, approximately Planck's constant `h0`.
    - The paper does not define `x_{i-1}` in Eq. (23) for the first agent.
      Following the released source, cyclic indexing is used, so the previous
      position of agent 0 is the last population member.
    - The source applies greedy parent-child selection, while Algorithm 1
      describes updating the position and then updating the global best/worst,
      without specifying greedy survival. This implementation follows the
      paper and directly accepts the generated position.
    - The paper reports a maximum of 2500 evaluations in its experimental
      description, whereas the released script uses 2500 full iterations with
      30 agents, corresponding to about 75,000 new objective evaluations.

    Eq. (20) and Eq. (21) divide by `psi(x_i)`, which may be exactly or nearly
    zero. The paper provides no singularity treatment; Mealpy's numerical
    epsilon is used only to prevent division by zero.

    Links
    -----
    1. Code: https://github.com/MohammedQaraad/SRA/blob/main/SRA_framework.ipynb
    2. Paper: https://doi.org/10.1016/j.knosys.2025.114273

    References
    ----------
    1. Hussein, N.K., Qaraad, M., El Najjar, A.M., Farag, M.A., Elhosseini, M.A., Mirjalili, S. and Guinovart, D., 2025.
       Schrödinger optimizer: A quantum duality-driven metaheuristic for stochastic optimization and engineering challenges.
       Knowledge-Based Systems, p.114273.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, SRA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = SRA.OriginalSRA(epoch=2500, pop_size=30)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Schrodinger Optimizer", year=2025, difficulty="medium",
                       kind="original", scientific_status="questionable",
                       concerns=(ScientificConcern.POOR_REPRODUCIBILITY,
                                 ScientificConcern.CODE_PSEUDOCODE_MISMATCH, ScientificConcern.AMBIGUOUS_METHODOLOGY,
                                 ScientificConcern.INSUFFICIENT_VALIDATION, ScientificConcern.QUESTIONABLE_MATH)
                       )

    def __init__(self, epoch: int = 10000, pop_size: int = 50, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000.
            pop_size (int): Number of search agents, default = 50.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

        # Eq. (15) explicitly sets k = 1 in the proposed optimizer.
        self.k = 1.0
        # The paper defines h = h0 / (2*pi), where h0 is Planck's constant.
        h0 = 6.62607015e-34
        self.h = h0 / (2.0 * np.pi)
        self.worst = None

    @staticmethod
    def get_wave_function(solution):
        """
        Calculate the wave function according to Eq. (19).

        psi(x_i, t) = sin(x_i)
        """
        return np.sin(solution)

    def get_probability_sequence(self):
        """
        Calculate p[i] according to Eq. (22).

        p[i] = ((N - i) / N)^2, i = 0, ..., N - 1.
        """
        idx = np.arange(self.pop_size)
        return ((self.pop_size - idx) / self.pop_size) ** 2

    def evolve(self, epoch):
        """
        The main operations of SRA.

        Args:
            epoch (int): The current iteration.
        """
        # The original pseudo-code starts with t = 0 and performs iterations while t < T.
        progress = (epoch - 1.) / self.epoch
        # Eq. (26)
        threshold = progress ** 3
        # The paper states that zc decreases linearly from 1 to 0.
        zc = 1.0 - progress
        # Eq. (22)
        p = self.get_probability_sequence()
        pop_new = []
        for idx in range(self.pop_size):
            current = self.pop[idx].solution.copy()
            # ----------------------------------------------------------
            # Random exploration -- Eq. (25)
            if self.generator.random() <= 0.03:
                # Eq. (25) explicitly defines randu as a standard-normal random variable.
                randn = self.generator.normal()
                pos_new = (randn * (self.problem.ub - self.problem.lb) + self.problem.lb)
            # ----------------------------------------------------------
            # Wave-particle duality
            elif p[idx] >= threshold:
                # ------------------------------------------------------
                # Wave-like exploration -- Eqs. (20)-(21)
                id_1, id_2 = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=2, replace=False)
                psi_i = self.get_wave_function(current)
                psi_best = self.get_wave_function(self.g_best.solution)
                psi_worst = self.get_wave_function(self.g_worst.solution)
                psi_1 = self.get_wave_function(self.pop[id_1].solution)
                psi_2 = self.get_wave_function(self.pop[id_2].solution)

                # The printed equations are singular whenever sin(x_i)=0.
                # No treatment is provided in the paper.
                denominator = np.where(np.abs(psi_i) > self.EPSILON, psi_i, np.where(psi_i >= 0.0, self.EPSILON, -self.EPSILON))
                # Common term in Eqs. (20)-(21).
                wave_term = (self.h * (psi_best - psi_worst) + p[idx] * (psi_1 - psi_i + psi_2))
                step = (self.generator.random() * zc * wave_term / denominator)
                if self.generator.random() < 0.5:
                    # Eq. (20)
                    pos_new = (self.g_best.solution + step)
                else:
                    # Eq. (21)
                    pos_new = (current + step)
            else:
                # ------------------------------------------------------
                # Particle-like exploitation -- Eqs. (23)-(24)
                if self.generator.random() < 0.5:
                    # Eq. (23)
                    # The paper does not define x_{i-1} for i = 0.
                    # Cyclic indexing follows the released implementation.
                    previous = self.pop[(idx - 1) % self.pop_size].solution
                    pos_new = (self.k * self.generator.random() + 2.0 * current - previous)
                else:
                    # Eq. (24)
                    id_1, id_2 = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=2, replace=False)
                    # The paper explicitly defines randu as a standard normally distributed random number.
                    randn = self.generator.normal()
                    pos_new = (self.g_best.solution - randn * (self.pop[id_1].solution - self.pop[id_2].solution))
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
            self.pop = self.update_target_for_population(pop_new)
