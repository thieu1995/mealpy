#!/usr/bin/env python
# Created by "Thieu" at 10:20, 16/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalSFOA(Optimizer):
    """
    The original version of: Superb Fairy-wren Optimization Algorithm (SFOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of search agents, in range [5, 10000].
        The paper uses 30 agents in the CEC experiments. Default is 30.
    c : float
        Breeding-and-feeding weighting coefficient in Eq. (6).
        The paper uses C = 0.8. Default is 0.8.
    levy_beta : float
        Levy-flight exponent used in the predator-avoidance stage.
        Subsequent descriptions of the original SFOA report beta = 1.5. Default is 1.5.

    Warnings
    --------
    This algorithm relies exclusively on the Lévy flight operator, whereas all other operators
    are overly simplistic. Moreover, the paper is ambiguous in numerous parts; therefore,
    users should exercise caution when employing this algorithm and thoroughly validate it on their specific problems.

    Note
    ----
    The original paper contains several ambiguities and inconsistencies:

    - The switching coefficient `r` is described as a population-proportion coefficient,
      but no equation is provided for it. The algorithm only tests `r > 0.5` or `r < 0.5`.
      This implementation uses a fresh U(0, 1) random value, consistent with later descriptions of SFOA.
    - Eq. (4) defines `s = 20*r1 + 20*r2`, with `r1` and `r2` normally distributed,
      but their mean and variance are not specified. Standard normal variables are used here.
    - Eq. (12) reverses the conditions of the breeding and predator-avoidance
      stages. Section 3.3 and Algorithm 2 consistently use breeding when
      `s < 20` and predator avoidance when `s >= 20`; this implementation follows those definitions.
    - The paper introduces a Levy-flight variable `l` but does not define its
      distribution or exponent. A standard Mantegna Levy flight with
      beta = 1.5 is used, consistent with later descriptions of the original SFOA.
    - The paper is inconsistent about when fitness evaluations and greedy
      replacement occur. This implementation evaluates one candidate per
      agent and uses greedy replacement, as implied by the stage descriptions.
    - Eqs. (7)-(8) depend explicitly on FEs / MaxFEs. Therefore, an NFE-based
      termination criterion is recommended for faithful reproduction.
    - The exact boundary-handling operator is not specified. Mealpy's
      standard solution correction is applied.

    SFOA generates one candidate per agent per population cycle, so the main
    search requires approximately `pop_size` objective-function evaluations per iteration.

    References
    ----------
    1. Jia, H., Zhou, X., Zhang, J., & Mirjalili, S. (2025).
       Superb Fairy-wren Optimization Algorithm: a novel metaheuristic algorithm for solving feature selection problems.
       Cluster Computing, 28(4), 246. https://doi.org/10.1007/s10586-024-04901-w

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, SFOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = SFOA.OriginalSFOA(epoch=1000, pop_size=30, c=0.8, levy_beta=1.2)
    >>> g_best = model.solve(problem)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Superb Fairy-wren Optimization Algorithm",
                       year=2025, difficulty="easy", kind="original", scientific_status="questionable",
                       concerns=(ScientificConcern.POOR_REPRODUCIBILITY,
                                 ScientificConcern.CODE_PSEUDOCODE_MISMATCH, ScientificConcern.AMBIGUOUS_METHODOLOGY,
                                 ScientificConcern.INSUFFICIENT_VALIDATION, ScientificConcern.QUESTIONABLE_MATH)
                       )

    def __init__(self, epoch: int = 10000, pop_size: int = 30, c: float = 0.8, levy_beta: float = 1.5,
            **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000.
            pop_size (int): Number of search agents, default = 30.
            c (float): Weighting coefficient in Eq. (6), default = 0.8.
            levy_beta (float): Levy-flight exponent, default = 1.5.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c = self.validator.check_float("c", c, (0.0, 1.0))
        self.levy_beta = self.validator.check_float("levy_beta", levy_beta, (0., 3.0))
        self.set_parameters(["epoch", "pop_size", "c", "levy_beta", ])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch):
        """
        The main operations of SFOA.

        Args:
            epoch (int): The current iteration.
        """
        pop_new = []
        for idx in range(self.pop_size):
            current = self.pop[idx].solution
            # ----------------------------------------------------------
            # Switching coefficient r.
            # The original paper does not provide an equation for r.
            # Later descriptions of SFOA treat it as a random threshold.
            r = self.generator.random()
            if r > 0.5:
                # ------------------------------------------------------
                # Young-bird growth stage -- Eq. (3)
                random_pos = (self.problem.lb + (self.problem.ub - self.problem.lb) * self.generator.random(self.problem.n_dims))
                pos_new = (current + random_pos)
            else:
                # ------------------------------------------------------
                # Danger factor -- Eq. (4)
                # The paper only says r1 and r2 follow a normal
                # distribution. Standard normal is used here.
                r1 = self.generator.normal()
                r2 = self.generator.normal()
                s = (20.0 * r1 + 20.0 * r2)
                ratio = min((self.nfe_counter - 1) / (self.epoch * self.pop_size), 1.0)
                if s < 20.0:
                    # --------------------------------------------------
                    # Breeding and feeding stage -- Eqs. (5)-(8)
                    # Eq. (8)
                    m = 2.0 * ratio
                    # Eq. (7)
                    p = np.sin(2.0 * (self.problem.ub - self.problem.lb) + (self.problem.ub - self.problem.lb) * m)
                    # Eq. (6)
                    x_g = (self.g_best.solution * self.c)
                    # Eq. (5)
                    pos_new = (x_g + (self.g_best.solution - current) * p)
                else:
                    # --------------------------------------------------
                    # Predator-avoidance stage -- Eqs. (9)-(11)
                    # Eq. (11)
                    w = (np.pi / 2.0 * ratio)
                    # Eq. (10)
                    k = (0.2 * np.sin(np.pi / 2.0 - w))
                    # Eq. (9)
                    levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
                    pos_new = self.g_best.solution + current * k * levy
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
