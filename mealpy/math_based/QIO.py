#!/usr/bin/env python
# Created by "Thieu" at 15:21, 23/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalQIO(Optimizer):
    """
    The original version of: Quadratic Interpolation Optimization (QIO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The main benchmark experiments use a population size of 50 and a
        maximum budget of 25,000 fitness evaluations. Default is 500.
    pop_size : int
        Population size, in range [4, 10000]. Default is 50.

    Links
    -----
    1. Paper: https://doi.org/10.1016/j.cma.2023.116446
    2. Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/135627-quadratic-interpolation-optimization-qio

    Note
    ----
    This implementation follows the generalized quadratic interpolation formulation in
    Eqs. (5) and (19)-(23), the exploration strategy in Eqs. (25)-(29), the exploitation strategy
    in Eqs. (30)-(32), and the greedy selection rule in Eq. (33).

    The released MATLAB source is largely consistent with the paper, but
    several implementation details and inconsistencies should be noted:

    - The text following Eq. (26) states that the GQI minimizer is formed
      using the current individual, `x_rand2`, and `x_rand3`, whereas
      Eq. (26) itself uses the current individual, `x_rand1`, and
      `x_rand2`. The MATLAB code follows Eq. (26), while `x_rand3` is used
      only in the exploration perturbation of Eq. (25). This
      implementation follows Eq. (26).
    - The MATLAB comments swap the equation numbers of the exploration
      equations: the GQI call is labeled Eq. (25), while the exploration update is labeled
      Eq. (26). According to the paper, these are Eqs. (26) and (25), respectively.
    - The paper does not specify a boundary-handling mechanism. The
      released MATLAB `SpaceBound` function randomly reinitializes each
      violated dimension within its corresponding bounds. The same rule is used here.
    - Eq. (5) does not define behavior for a zero interpolation
      denominator. The MATLAB source adds machine epsilon to the denominator.
    - The MATLAB interpolation routine replaces the computed minimizer with
      a uniformly random value within the corresponding variable bounds if
      the result is NaN, infinite, or outside the search interval. This
      fallback is not described in the paper and is reproduced here for
      source-level reproducibility.
    - The theoretical GQI derivation assumes strict fitness ordering
      `f_i < f_j < f_k` and strict positional ordering. The MATLAB source
      sorts the three fitness values and uses non-strict positional
      comparisons, implicitly defining behavior for ties. This
      implementation follows the source for such cases.
    - In the exploration stage, the MATLAB source selects three distinct
      individuals that are also different from the current individual.
      This agrees with the paper's description of different randomly
      selected individuals and is enforced here.
    - In the exploitation stage, the MATLAB source selects two distinct
      individuals excluding the current individual. The paper only states
      that two individuals are randomly selected from the population and
      does not explicitly state this exclusion.
    - Eq. (31) writes the GQI inputs in the order
      `(best, rand1, rand2)`, while the MATLAB code passes
      `(rand1, rand2, best)`. Since GQI internally sorts the three points
      by fitness, these forms are equivalent except for possible exact
      fitness ties.
    - The paper states that the adaptive parameter `b` gradually increases
      as iterations progress, although Eqs. (28)-(29), Fig. 10, and the
      MATLAB source show an overall decrease toward zero. The equations
      are followed here.
    - The MATLAB implementation immediately applies Eq. (33) to each
      individual before processing the next one. Therefore, later
      individuals can sample already-updated population members from the
      same iteration. The population update is consequently sequential,
      not generational, and the algorithm is not parallelized here.
    - The global best used during an iteration is updated only after all
      individuals have been processed, as shown in Algorithm 1 and the
      MATLAB implementation. It is therefore kept fixed within each
      iteration.
    - The paper formulates GQI specifically for minimization and Eq. (33)
      explicitly accepts a candidate only when its fitness is smaller.
      This original implementation therefore supports minimization only.
    - The paper reports a maximum of 25,000 fitness evaluations with
      `pop_size = 50`. The released code evaluates the initial population
      and then evaluates one candidate per individual per iteration.
      Consequently, 500 iterations require 25,050 total objective calls
      if initialization is included in the NFE count.

    QIO evaluates exactly one new candidate per individual per iteration. Thus,
    excluding initialization, each iteration requires `pop_size` objective-function evaluations.

    References
    ----------
    1. Zhao, W., Wang, L., Zhang, Z., Mirjalili, S., Khodadadi, N., & Ge, Q. (2023).
       Quadratic Interpolation Optimization (QIO): A new optimization algorithm based on generalized
       quadratic interpolation and its applications to real-world engineering problems.
       Computer Methods in Applied Mechanics and Engineering, 417, 116446.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, QIO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = QIO.OriginalQIO(epoch=500, pop_size=50)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Quadratic Interpolation Optimization", year=2023, difficulty="hard", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.QUESTIONABLE_MATH, ScientificConcern.INCORRECT_EQUATIONS,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH
                       ))

    def __init__(self, epoch: int = 500, pop_size: int = 50, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 500.
            pop_size (int): Population size, default = 50.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [4, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def space_bound(self, solution):
        """
        Apply the boundary treatment used in the released MATLAB source.

        Each violated component is independently reinitialized uniformly
        within its corresponding search interval.
        """
        mask = (solution > self.problem.ub) | (solution < self.problem.lb)
        if np.any(mask):
            random_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
            solution = np.where(mask, random_pos, solution)
        return solution

    def interpolation(self, xi, xj, xk, fi, fj, fk, lower, upper):
        """
        Calculate the quadratic interpolation minimizer using Eq. (5).
        The numerical stabilization and random fallback follow the released MATLAB implementation.
        """
        numerator = ((xj ** 2 - xk ** 2) * fi + (xk ** 2 - xi ** 2) * fj + (xi ** 2 - xj ** 2) * fk)
        denominator = 2.0 * ((xj - xk) * fi + (xk - xi) * fj + (xi - xj) * fk)
        # MATLAB source:
        # L_xmin = a / (b + eps)
        x_min = numerator / (denominator + np.finfo(float).eps)
        # This fallback is defined by the released source, not the paper.
        if (not np.isfinite(x_min) or x_min > upper or x_min < lower):
            x_min = self.generator.uniform(lower, upper)
        return x_min

    def gqi(self, x1, x2, x3, f1, f2, f3):
        """
        Generalized Quadratic Interpolation according to Eq. (23).

        The three solutions are first reordered according to their fitness
        values such that fi <= fj <= fk, following the released MATLAB code.
        """
        points = np.vstack((np.asarray(x1, dtype=float), np.asarray(x2, dtype=float), np.asarray(x3, dtype=float)))
        fitness = np.asarray([f1, f2, f3], dtype=float, )

        # Stable ordering provides deterministic handling of exact ties.
        order = np.argsort(fitness, kind="stable")
        xi, xj, xk = points[order]
        fi, fj, fk = fitness[order]
        lb = np.asarray(self.problem.lb, dtype=float)
        ub = np.asarray(self.problem.ub, dtype=float)
        x_min = np.zeros(self.problem.n_dims, dtype=float)
        for dim in range(self.problem.n_dims):
            xi_d = xi[dim]
            xj_d = xj[dim]
            xk_d = xk[dim]

            # ----------------------------------------------------------
            # Eq. (23), first case:
            # xj <= xi <= xk or xk <= xi <= xj
            if ((xk_d >= xi_d >= xj_d) or (xj_d >= xi_d >= xk_d)):
                x_min[dim] = self.interpolation(xi_d, xj_d, xk_d, fi, fj, fk, lb[dim], ub[dim])
            # ----------------------------------------------------------
            # Eq. (19):
            # xi <= xj <= xk
            elif xk_d >= xj_d >= xi_d:
                value = self.interpolation(xi_d, xj_d, xk_d, fi, fj, fk, lb[dim], ub[dim])
                if value < xj_d:
                    x_min[dim] = value
                else:
                    xk_new = (3.0 * xi_d - 2.0 * xj_d)
                    x_min[dim] = self.interpolation(xi_d, xj_d, xk_new, fi, fj, fk, lb[dim], ub[dim])
            # ----------------------------------------------------------
            # Eq. (20):
            # xk <= xj <= xi
            elif xi_d >= xj_d >= xk_d:
                value = self.interpolation(xi_d, xj_d, xk_d, fi, fj, fk, lb[dim], ub[dim])
                if value > xj_d:
                    x_min[dim] = value
                else:
                    xk_new = (3.0 * xi_d - 2.0 * xj_d)
                    x_min[dim] = self.interpolation(xi_d, xj_d, xk_new, fi, fj, fk, lb[dim], ub[dim])

            # ----------------------------------------------------------
            # Eq. (21):
            # xi <= xk <= xj
            elif xj_d >= xk_d >= xi_d:
                xj_new = (2.0 * xi_d - xk_d)
                x_min[dim] = self.interpolation(xi_d, xj_new, xk_d, fi, fj, fk, lb[dim], ub[dim])

            # ----------------------------------------------------------
            # Eq. (22):
            # xj <= xk <= xi
            elif xi_d >= xk_d >= xj_d:
                xj_new = (2.0 * xi_d - xk_d)
                x_min[dim] = self.interpolation(xi_d, xj_new, xk_d, fi, fj, fk, lb[dim], ub[dim])
        return x_min

    def evolve(self, epoch):
        """
        The main operations of QIO.

        Args:
            epoch (int): The current iteration.
        """
        # The MATLAB source and Algorithm 1 update x_best only after
        # processing all individuals. Therefore, freeze the global best during the current iteration.
        best_pos = self.g_best.solution.copy()
        best_fit = self.g_best.target.fitness

        for idx in range(self.pop_size):
            # ----------------------------------------------------------
            # Exploration -- Eqs. (25)-(29)
            # ----------------------------------------------------------
            if self.generator.random() > 0.5:
                k1, k2, k3 = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=3, replace=False)
                agent_i = self.pop[idx]
                agent_1 = self.pop[k1]
                agent_2 = self.pop[k2]

                # Eq. (26)
                pos_gqi = self.gqi(agent_i.solution, agent_1.solution, agent_2.solution, agent_i.target.fitness, agent_1.target.fitness, agent_2.target.fitness)
                # Eq. (29)
                a = np.cos(np.pi * epoch / (2.0 * self.epoch))
                # Eq. (28)
                b = (0.7 * a + 0.15 * a * (np.cos(5.0 * np.pi * epoch / self.epoch) + 1.0))
                # Eq. (27)
                w1 = (3.0 * b * self.generator.normal())
                # MATLAB:
                # round(0.5 * (0.05 + rand))
                # This is 1 only when rand >= 0.95.
                gate = (1.0 if self.generator.random() >= 0.95 else 0.0)

                # Eq. (25) uses two independent random values in (0, 1).
                r2 = self.generator.random()
                r3 = self.generator.random()
                random_term = (gate * np.log(r2 / r3))
                # Eq. (25)
                pos_new = (pos_gqi + w1 * (self.pop[k3].solution - pos_gqi) + random_term)

            # ----------------------------------------------------------
            # Exploitation -- Eqs. (30)-(32)
            else:
                k1, k2 = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=2, replace=False)
                agent_1 = self.pop[k1]
                agent_2 = self.pop[k2]
                # Eq. (31)
                # The MATLAB source passes (rand1, rand2, best), whereas the paper writes (best, rand1, rand2).
                # GQI sorts the points internally by fitness, so the order is normally equivalent.
                pos_gqi = self.gqi(agent_1.solution, agent_2.solution, best_pos, agent_1.target.fitness, agent_2.target.fitness, best_fit)

                # Eq. (32)
                w2 = (3.0 * (1.0 - (epoch - 1.0) / self.epoch) * self.generator.normal())
                # rD in Eq. (30)
                r_dim = int(self.generator.integers(self.problem.n_dims))
                # MATLAB round(1 + rand) produces either 1 or 2.
                multiplier = (1.0 if self.generator.random() < 0.5 else 2.0)
                scale = ((self.problem.ub - self.problem.lb) / (self.problem.ub[r_dim] - self.problem.lb[r_dim]))
                # Eq. (30)
                pos_new = (pos_gqi + w2 * (best_pos - multiplier * scale * self.pop[idx].solution[r_dim]))

            # ----------------------------------------------------------
            # Boundary handling
            # Not defined in the paper. Follow the released SpaceBound() implementation.
            # ----------------------------------------------------------
            pos_new = self.space_bound(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)

            # ----------------------------------------------------------
            # Eq. (33): greedy replacement.
            # Strict inequality matches the MATLAB source.
            # ----------------------------------------------------------
            if (agent.target.fitness < self.pop[idx].target.fitness):
                self.pop[idx] = agent
