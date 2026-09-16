#!/usr/bin/env python
# Created by "Thieu" at 05:08, 16/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalCLCO(Optimizer):
    """
    The original version of: Coordination Core-Ligand Collaborative Optimizer (CLCO)

    This implementation follows the authors' released MATLAB source code rather
    than relying solely on the published equations. The paper is not sufficient
    to reproduce CLCO exactly because several implementation-critical details
    are missing or ambiguous.

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The benchmark experiments use 500 iterations. Default is 500.
    pop_size : int
        Number of ligands, in range [5, 10000].
        The benchmark experiments use 30 ligands. Default is 30.
    alpha0 : float
        Base coordination-field factor. Default is 1.8.
    beta0 : float
        Base Jahn-Teller factor. Default is 0.3.
    gamma0 : float
        Base crystal-field factor. Default is 0.6.
    delta0 : float
        Base ligand-exchange rate. Default is 0.2.

    Warnings
    --------
    The paper alone cannot uniquely reproduce the released CLCO implementation.
    Therefore, this class follows the authors' MATLAB source where the two differ
    or where the paper omits necessary details. Important discrepancies include:

    - The paper does not provide explicit schedules for `alpha`, `beta`, and
      `gamma`. The MATLAB code introduces additional coefficients and a
      discontinuous switch at 70% progress.
    - Several equations use `(ub - lb)` where a scalar is required. The MATLAB
      code resolves this as `norm(ub - lb)`, introducing a dimension-dependent
      scale proportional to sqrt(D) for equal-width bounds.
    - The MATLAB implementation modifies Eq. (3) by computing the fitness ratio
      with `center_f + 1e-12`.
    - The crystal-field ranking is computed before that phase and reused later
      by the Jahn-Teller reset, so its bottom-50% set and `x_best` may be stale.
    - The ligand-exchange phase is sequential and in-place, while its mean
      fitness threshold is computed only once before the exchange loop.
    - The initial global-best tracker is the randomly generated central ion,
      even when an initialized ligand is better.
    - The source counts every objective call and states that one iteration costs
      roughly 2.2-2.5 * pop_size evaluations, whereas the paper describes
      `pop_size * epoch` as the common benchmark FE budget. The released source
      itself does not enforce such a `max_fe` stopping rule.

    Consequently, equal iteration counts should not be interpreted as equal
    objective-function evaluation budgets for CLCO.

    Links
    -----
    1. https://github.com/JunhaoWei-mpu/ROBIS-Lab/tree/Coordination-Core-Ligand-Collaborative-Optimizer-(CLCO)


    References
    ----------
    1. Li, Y., Wei, J., Mirjalili, S., Zhao, Y., Li, Z., Wang, Z., Im, S.K., Yang, X. and Wang, Y., 2026.
       Coordination Core-Ligand Collaborative Optimizer (CLCO): A chemistry-inspired metaheuristic
       for global numerical optimization, space–air–ground path planning, and constrained engineering design.
       Advanced Engineering Informatics, 76, p.105092.
       https://doi.org/10.1016/j.aei.2026.105092

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CLCO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = CLCO.OriginalCLCO(epoch=500, pop_size=30, alpha0=1.5, beta0=0.5, gamma0=0.6, delta0=0.2)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Coordination Core-Ligand Collaborative Optimizer",
                       year=2026, difficulty="hard", kind="original", scientific_status="questionable",
                       concerns=(ScientificConcern.INCORRECT_EQUATIONS, ScientificConcern.POOR_REPRODUCIBILITY,
                                 ScientificConcern.CODE_PSEUDOCODE_MISMATCH, ScientificConcern.AMBIGUOUS_METHODOLOGY,
                                 ScientificConcern.INSUFFICIENT_VALIDATION, ScientificConcern.QUESTIONABLE_MATH)
                       )

    def __init__(self, epoch: int = 500, pop_size: int = 30, alpha0: float = 1.8, beta0: float = 0.3,
            gamma0: float = 0.6, delta0: float = 0.2, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 500.
            pop_size (int): Number of ligands, default = 30.
            alpha0 (float): Base coordination-field factor, default = 1.8.
            beta0 (float): Base Jahn-Teller factor, default = 0.3.
            gamma0 (float): Base crystal-field factor, default = 0.6.
            delta0 (float): Base ligand-exchange rate, default = 0.2.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha0 = self.validator.check_float("alpha0", alpha0, [0.0, 10.0])
        self.beta0 = self.validator.check_float("beta0", beta0, [0.0, 10.0])
        self.gamma0 = self.validator.check_float("gamma0", gamma0, [0.0, 10.0])
        self.delta0 = self.validator.check_float("delta0", delta0, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "alpha0", "beta0", "gamma0", "delta0"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        """Initialize CLCO-specific state variables."""
        self.center = None
        self.stall = 0
        self.box = None

    def initialization(self):
        ranges = self.problem.ub - self.problem.lb
        self.box = np.linalg.norm(ranges)
        # Eq. (1): central ion.
        center_pos = self.problem.generate_solution(encoded=True)
        self.center = self.generate_empty_agent(center_pos)
        self.center.target = self.get_target(center_pos)
        pop = []
        for _ in range(self.pop_size):
            direction = self.generator.normal(0.0, 1.0, self.problem.n_dims)
            norm = np.linalg.norm(direction)
            if norm > 0.0:
                direction = direction / norm
            else:
                direction = np.zeros(self.problem.n_dims)
            # MATLAB source interpretation of Eq. (2).
            distance = (0.3 * self.box * self.generator.random())
            pos_new = (self.center.solution + distance * direction)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop.append(agent)
        self.pop = pop

    def before_main_loop(self):
        """
        Reproduce the MATLAB initialization of the best tracker.

        The reference source initializes the global best with the central ion,
        even if an initial ligand already has better fitness.
        """
        self.g_best = self.center.copy()
        self.stall = 0

    def get_dynamic_parameters(self, epoch):
        """
        Calculate dynamic parameters exactly as in the released MATLAB source.
        """
        progress = epoch / self.epoch
        # These numerical schedules are present in the MATLAB source but are
        # not fully specified by the published paper.
        if progress < 0.7:
            alpha = self.alpha0 * (1.0 - 0.6 * progress)
            beta = self.beta0 * (1.0 - 0.4 * progress)
            gamma = self.gamma0 * (0.6 + 0.4 * progress)
        else:
            alpha = 0.3 * self.alpha0
            beta = 0.4 * self.beta0
            gamma = 0.8 * self.gamma0
        # Eq. (15).
        delta = self.delta0 * (0.7 + 0.3 * np.sin(2.0 * np.pi * epoch / 100.0))
        return progress, alpha, beta, gamma, delta

    def get_coordination_strengths(self):
        """
        Calculate coordination strengths following the MATLAB implementation
        of Eqs. (3)-(4).
        """
        positions = np.asarray([agent.solution for agent in self.pop])
        fitness = np.asarray([agent.target.fitness for agent in self.pop])
        distances = np.linalg.norm(positions - self.center.solution, axis=1)
        # Source differs slightly from the printed Eq. (3):
        #   fitness_ratio = ligand_f / (center_f + EPS)
        fitness_ratio = (fitness / (self.center.target.fitness + self.EPSILON))
        strength = (np.exp(-distances / self.box) / (fitness_ratio + self.EPSILON))
        # Eq. (4)
        strength = (strength / (np.max(strength) + self.EPSILON))
        return strength

    def coordination_field_phase(self, progress, alpha, ):
        """
        Coordination-field formation following Eqs. (3)-(7).
        """
        strengths = self.get_coordination_strengths()
        for idx in range(self.pop_size):
            current = self.pop[idx].solution
            if (strengths[idx] > 0.7 and self.generator.random() > 0.3 * progress):
                # Eq. (5)
                step = (alpha * strengths[idx] * (1.5 + self.generator.random()))
                direction = (self.center.solution - current)
            elif (self.generator.random() < 0.2 + 0.3 * (1.0 - progress)):
                # Eq. (6)
                step = (alpha * strengths[idx] * (1.5 + self.generator.random()))
                direction = self.generator.normal(0.0, 1.0, self.problem.n_dims, )
            else:
                # Eq. (7)
                step = (alpha * strengths[idx] * (0.3 + 0.2 * self.generator.random()))
                direction = (self.center.solution - current + 0.3 * self.generator.normal(0.0, 1.0, self.problem.n_dims))
            pos_new = current + step * direction
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Greedy acceptance.
            if agent.target.fitness < self.pop[idx].target.fitness:
                self.pop[idx] = agent

    def crystal_field_phase(self, progress, gamma, ):
        """
        Crystal-field splitting following Eqs. (8)-(11).

        Returns the ranking, best position, and worst position computed before
        the phase. The MATLAB source later reuses these stale values during the Jahn-Teller reset.
        """
        fitness = np.asarray([agent.target.fitness for agent in self.pop])
        order = np.argsort(fitness)
        rank_of = np.empty(self.pop_size, dtype=int)
        rank_of[order] = np.arange(1, self.pop_size + 1)
        best_idx = order[0]
        worst_idx = order[-1]
        x_best = self.pop[best_idx].solution.copy()
        x_worst = self.pop[worst_idx].solution.copy()

        for idx in range(self.pop_size):
            # MATLAB source carries the best ligand unchanged.
            if idx == best_idx:
                continue

            current = self.pop[idx].solution
            rank_ratio = (rank_of[idx] / self.pop_size)
            if rank_ratio < 0.3:
                # Eq. (8)
                field = gamma * 0.3
                direction = (x_best - current)
            elif rank_ratio > 0.7:
                field = (gamma * (1.5 + 0.5 * self.generator.random()))
                if (self.generator.random() < 0.4 * (1.0 - progress)):
                    # Eq. (9)
                    direction = self.generator.normal(0.0, 1.0, self.problem.n_dims)
                else:
                    # Eq. (10)
                    direction = (x_best - x_worst + self.generator.normal(0.0, 1.0, self.problem.n_dims))
            else:
                # Eq. (11)
                field = (gamma * (0.8 + 0.4 * self.generator.random()))
                direction = (x_best - current + 0.5 * self.generator.normal(0.0, 1.0, self.problem.n_dims))
            # Table 5: multiplicative step jitter.
            step = (field * (1.0 + 0.2 * self.generator.normal()))
            pos_new = current + step * direction
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if agent.target.fitness < self.pop[idx].target.fitness:
                self.pop[idx] = agent
        return order, x_best, x_worst

    def jahn_teller_phase(self, epoch, progress, beta, order, x_best, ):
        """
        Jahn-Teller distortion following the released MATLAB implementation of Eqs. (12)-(14).

        The ranking and `x_best` are intentionally those calculated before the
        crystal-field updates, matching the source code.
        """
        positions = np.asarray([agent.solution for agent in self.pop])
        fitness = np.asarray([agent.target.fitness for agent in self.pop])
        centroid = np.mean(positions, axis=0)
        sym_dev = np.mean(np.linalg.norm(positions - centroid, axis=1))
        # MATLAB: std(ligand_f, 1)
        diversity = np.std(fitness, ddof=0, )
        needs_jt = (diversity < 1e-8 or sym_dev < 0.1 * self.box or epoch % 30 == 0)
        if not needs_jt:
            return
        jt = (beta * (1.0 + 0.5 * (1.0 - progress)))
        # Eq. (13): one-axis distortion.
        distortion = np.zeros(self.problem.n_dims)
        axis_idx = self.generator.integers(self.problem.n_dims)
        sign = (1.0 if self.generator.random() > 0.5 else -1.0)
        distortion[axis_idx] = (sign * jt * self.box * 0.15)
        distortion += (0.2 * jt * (self.generator.random(self.problem.n_dims) - 0.5))
        center_pos = (self.center.solution + distortion)
        center_pos = self.correct_solution(center_pos)
        center_new = self.generate_agent(center_pos)
        if center_new.target.fitness < self.center.target.fitness:
            self.center = center_new

        # MATLAB source reuses `order` computed before crystal-field updates.
        reset_prob = (0.4 + 0.3 * (1.0 - progress))
        start = self.pop_size // 2
        for pos in range(start, self.pop_size):
            idx = order[pos]
            if self.generator.random() >= reset_prob:
                continue
            if self.generator.random() < 0.7:
                # Eq. (14): global reset.
                pos_new = (self.problem.lb + (self.problem.ub - self.problem.lb) * self.generator.random(self.problem.n_dims))
            else:
                # Eq. (14): local reset around the stale x_best.
                direction = self.generator.normal(0.0, 1.0, self.problem.n_dims, )
                norm = np.linalg.norm(direction)
                if norm > 0.0:
                    direction = direction / norm
                else:
                    direction = np.zeros(self.problem.n_dims)
                distance = (0.3 * self.box * self.generator.random())
                pos_new = (x_best + distance * direction)
                pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Reset is non-greedy in the MATLAB source.
            self.pop[idx] = agent

    def ligand_exchange_phase(self, delta, ):
        """
        Perform the sequential in-place ligand exchange of Eqs. (15)-(18).
        """
        # MATLAB source freezes mean fitness before the exchange loop.
        mean_f = np.mean([agent.target.fitness for agent in self.pop])
        for idx in range(self.pop_size):
            if self.generator.random() >= delta:
                continue
            fitness = np.asarray([agent.target.fitness for agent in self.pop])
            u = self.generator.random()
            if u < 0.4:
                pool = np.flatnonzero(fitness < mean_f)
            elif u < 0.7:
                # Empty pool deliberately triggers uniform fallback.
                pool = np.asarray([], dtype=int)
            else:
                pool = np.flatnonzero(fitness > mean_f)
            pool = pool[pool != idx]
            if pool.size > 0:
                partner = self.generator.choice(pool)
            else:
                candidates = np.delete(np.arange(self.pop_size), idx)
                partner = self.generator.choice(candidates)
            mask_prob = (0.4 + 0.3 * self.generator.random())
            mask = (self.generator.random(self.problem.n_dims) < mask_prob)
            # MATLAB source skips a zero mask without evaluating either ligand.
            if not np.any(mask):
                continue
            pos_i = self.pop[idx].solution.copy()
            pos_j = self.pop[partner].solution.copy()
            temp = pos_i[mask].copy()
            pos_i[mask] = pos_j[mask]
            pos_j[mask] = temp
            # Both exchanged ligands are re-evaluated unconditionally.
            agent_i = self.generate_agent(pos_i)
            agent_j = self.generate_agent(pos_j)
            self.pop[idx] = agent_i
            self.pop[partner] = agent_j

    def central_ion_phase(self, progress, ):
        """
        Update the central ion according to Eqs. (19)-(20).
        """
        fitness = np.asarray([agent.target.fitness for agent in self.pop])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.center.target.fitness:
            # Eq. (19)
            self.center = self.pop[best_idx].copy()
        elif self.generator.random() < 0.3 * (1.0 - progress):
            # Eq. (20)
            pos_new = (self.center.solution + 0.03 * (1.0 - progress) * (self.generator.random(self.problem.n_dims) - 0.5))
            pos_new = self.correct_solution(pos_new)
            center_new = self.generate_agent(pos_new)
            if center_new.target.fitness < self.center.target.fitness:
                self.center = center_new

    def evolve(self, epoch):
        """
        The main operations of CLCO.

        Args:
            epoch (int): The current iteration.
        """
        (progress, alpha, beta, gamma, delta) = self.get_dynamic_parameters(epoch)

        # MATLAB stagnation safeguard.
        if self.stall > 20:
            alpha = 0.8 * self.alpha0
            beta = 0.7 * self.beta0
            self.stall = 0

        # --------------------------------------------------------------
        # 1. Coordination-field formation
        # --------------------------------------------------------------
        self.coordination_field_phase(progress, alpha)

        # --------------------------------------------------------------
        # 2. Crystal-field splitting
        #
        # Keep the old ranking and x_best because the MATLAB source
        # deliberately reuses them in the subsequent JT reset.
        # --------------------------------------------------------------
        (order, x_best, _,) = self.crystal_field_phase(progress, gamma)

        # --------------------------------------------------------------
        # 3. Jahn-Teller distortion
        # --------------------------------------------------------------
        self.jahn_teller_phase(epoch, progress, beta, order, x_best)

        # --------------------------------------------------------------
        # 4. Ligand exchange
        # --------------------------------------------------------------
        self.ligand_exchange_phase(delta)

        # --------------------------------------------------------------
        # 5. Central-ion update
        # --------------------------------------------------------------
        self.central_ion_phase(progress)

        # MATLAB source updates its best tracker from the central ion only.
        if self.center.target.fitness < self.g_best.target.fitness:
            self.g_best = self.center.copy()
            self.stall = 0
        else:
            self.stall += 1
