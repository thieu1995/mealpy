#!/usr/bin/env python
# Created by "Thieu" at 06:08, 15/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalMO(Optimizer):
    """
    The original version of: Musk Ox Optimizer (MO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Number of musk oxen in the population, in range [5, 10000]. Default is 50.

    Note
    ----
    MO uses three behaviors: migration, foraging, and defense. The alert
    signal determines whether migration or roosting is performed, while
    the safety signal selects between foraging and defense during roosting.

    The original paper states that adult males and females each constitute
    40% of the population, while juveniles constitute the remaining 20%.
    Since the paper does not define persistent identities for individual
    agents, the defensive phase randomly assigns 40% of the population
    as males; all remaining individuals use the female/juvenile update.


    Warnings
    --------
    The original paper contains several ambiguities and typographical issues that
    should be considered when reproducing MO:

    - The paper states that males and females each represent 40% of the population
      and juveniles 20%, but does not specify how these identities are assigned or
      whether they remain fixed during optimization.
    - The definitions of the "second" and "worst" musk oxen in Eq. (14) are not
      sufficiently precise about whether they refer to the current population or
      historical solutions.
    - Eq. (16) updates female and juvenile positions without adding the current
      position. This implementation follows the equation literally.
    - The Levy-flight description contains notation inconsistencies; Eq. (18) is
      followed, with sigma_y = 1 and a = 1.5.
    - The paper does not specify a boundary-handling strategy. Mealpy's standard
      solution correction is therefore applied.
    - Eq. (2) incorrectly describes `MOlb` and `MOub` in the accompanying text;
      this implementation uses them conventionally as lower and upper bounds.

    The constants `G1 = 0.001`, `G2 = 100`, `beta = 0.01`, and Levy exponent
    `a = 1.5` are fixed by the original formulation and are not exposed as
    tunable parameters.


    References
    ----------
    1. Yuan, Y., Chong, G., Ren, J., Zhao, W., Li, Y., Wang, Z., & Mirjalili, S. (2025).
       Musk ox optimizer (MO): a novel optimization algorithm and its application. Cluster Computing, 28(16), 1041.
       https://doi.org/10.1007/s10586-025-05735-w

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, MO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>> model = MO.OriginalMO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Musk Ox Optimizer", year=2026, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Number of musk oxen in the population, default = 50.
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations of the Musk Ox Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        ratio = epoch / self.epoch
        # --------------------------------------------------------------
        # Eqs. (3)-(6): alert and safety signals
        alpha1 = 1.0 - ratio
        r1 = self.generator.random()
        # Eq. (5)
        alert_signal = (2.0 * alpha1 * (1.0 - r1))
        # Eq. (6)
        safety_signal = self.generator.random()
        # Historical best position.
        best_pos = self.g_best.solution.copy()
        pop_new = []
        # ==============================================================
        # Migration: exploration phase, Eqs. (7)-(9)
        # ==============================================================
        if np.abs(alert_signal) >= 1.0:
            # Eq. (9)
            alpha2 = (1.0 - 1.0 / (1.0 + np.exp(((self.epoch / 2.0 - epoch) / self.epoch) * 10.0)))
            for idx in range(self.pop_size):
                # Select two different guards.
                guard_indices = self.generator.choice(self.pop_size, size=2, replace=False)
                x_a = self.pop[guard_indices[0]].solution
                x_b = self.pop[guard_indices[1]].solution
                r3 = self.generator.random()
                # Eq. (8)
                step = (alpha2 ** 2 * r3 ** 4 * (x_a - x_b))
                # Eq. (7)
                pos_new = self.pop[idx].solution + step
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
        # ==============================================================
        # Roosting: exploitation phase
        # ==============================================================
        else:
            # ==========================================================
            # Foraging behavior, Eqs. (10)-(12)
            # ==========================================================
            if np.abs(safety_signal) >= 0.5:
                g1 = 0.001
                beta = 0.01
                # Eq. (12)
                wi = (g1 / np.exp(beta * epoch / self.epoch))
                for idx in range(self.pop_size):
                    current = self.pop[idx].solution
                    rand = self.generator.random()
                    r4 = self.generator.random()
                    # Eq. (11)
                    dx1 = (rand * (best_pos - current) + r4 ** 3 * wi * (current - best_pos))
                    # Eq. (10)
                    pos_new = current + dx1
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        agent.target = self.get_target(pos_new)
            # ==========================================================
            # Defensive behavior, Eqs. (13)-(18)
            # ==========================================================
            else:
                # Obtain the current second-best and worst musk oxen.
                _, list_best, list_worst = self.get_special_agents(self.pop, n_best=2, n_worst=1, minmax=self.problem.minmax)
                second_pos = list_best[1].solution
                worst_pos = list_worst[0].solution
                g2 = 100.0
                beta = 0.01
                # Eq. (15)
                mi = (g2 / np.exp(beta * epoch / self.epoch))
                # Adult males represent 40% of the population.
                n_males = int(np.round(0.4 * self.pop_size))
                n_males = max(1, min(n_males, self.pop_size))
                # The paper specifies population proportions but does not
                # prescribe permanent identities for individual agents.
                indices = self.generator.permutation(self.pop_size)
                male_indices = set(indices[:n_males].tolist())
                for idx in range(self.pop_size):
                    current = self.pop[idx].solution
                    # --------------------------------------------------
                    # Male musk oxen, Eqs. (13)-(15)
                    # --------------------------------------------------
                    if idx in male_indices:
                        r5 = self.generator.random()
                        # Eq. (14)
                        dx2 = (r5 ** 2 * (second_pos - best_pos) + r5 * mi * (best_pos - worst_pos))
                        # Eq. (13)
                        pos_new = current + dx2
                    # --------------------------------------------------
                    # Female and juvenile musk oxen, Eqs. (16)-(18)
                    # --------------------------------------------------
                    else:
                        r6 = self.generator.random()
                        levy = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=self.problem.n_dims, case=-1)
                        # Eq. (16)
                        pos_new = (r6 * (best_pos - current) * levy)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # The pseudocode updates the population directly. It does not
        # specify greedy parent-offspring survivor selection.
        self.pop = pop_new
