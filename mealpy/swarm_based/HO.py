#!/usr/bin/env python
# Created by "Thieu" at 03:37, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalHO(Optimizer):
    """
    The original version of: Hippopotamus Optimization Algorithm (HO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 500.
    pop_size : int
        Number of hippopotamuses in the population, in range [4, 10000]. The population size
        must be even because the first two exploration phases operate on two equal
        halves of the population. Default is 24.

    Note
    ----
    HO performs approximately 3 * pop_size objective-function evaluations per iteration (NFE):
    pop_size evaluations in Phase 1, pop_size evaluations in Phase 2, and pop_size evaluations in Phase 3.
    Therefore, using the same `epoch` and `pop_size` as other optimizers does not imply the same computational budget.
    For fair comparisons, the total number of function evaluations should be controlled.

    The three phases are executed sequentially with greedy replacement, so later
    phases may operate on solutions already updated by earlier phases.

    The local bounds in Phase 3 follow Eq. (16) exactly: `lb_local = lb / t` and `ub_local = ub / t`.
    The Levy exponent is fixed at 1.5 as specified in the original paper.

    References
    ----------
    1. Amiri, M.H., Mehrabi Hashjin, N., Montazeri, M., Mirjalili, S., & Khodadadi, N. (2024).
       Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm.
       Scientific Reports, 14, 5032. https://doi.org/10.1038/s41598-024-54910-3

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, HO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>>
    >>> model = HO.OriginalHO(epoch=500, pop_size=24)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Hippopotamus Optimization Algorithm", year=2024, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 500, pop_size: int = 24, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [4, 10000])
        if self.pop_size % 2 != 0:
            raise ValueError("'pop_size' must be even because HO divides the population into two equal halves.")
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        # The three phases contain sequential greedy updates.
        self.is_parallelizable = False

    def generate_h_scenarios(self, i1, i2):
        """
        Generate the five h scenarios defined in Eq. (4).
        """
        rho1 = self.generator.integers(0, 2)
        rho2 = self.generator.integers(0, 2)
        h = [i2 * self.generator.random(self.problem.n_dims) + (1 - rho1),
             2.0 * self.generator.random(self.problem.n_dims) - 1.0,
             self.generator.random(self.problem.n_dims),
             i1 * self.generator.random(self.problem.n_dims) + (1 - rho2),
             self.generator.random()]
        return h

    def evolve(self, epoch):
        """
        The main operations of the Hippopotamus Optimization Algorithm.

        Args:
            epoch (int): The current iteration.
        """
        n_half = self.pop_size // 2
        # ==============================================================
        # Phase 1: Hippopotamuses position update in the river or pond
        # Exploration, Eqs. (3)-(9)
        # ==============================================================
        dominant = self.g_best.solution.copy()
        for idx in range(n_half):
            current = self.pop[idx].solution.copy()
            # I1 and I2 are random integers in {1, 2}.
            i1 = self.generator.integers(1, 3)
            i2 = self.generator.integers(1, 3)
            # Random group used to calculate MG_i.
            group_size = self.generator.integers(1, self.pop_size + 1)
            group_indices = self.generator.choice(self.pop_size, size=group_size, replace=False)
            mean_group = np.mean(np.array([self.pop[j].solution for j in group_indices]), axis=0)
            h = self.generate_h_scenarios(i1, i2)
            h1 = h[self.generator.integers(0, 5)]
            h2 = h[self.generator.integers(0, 5)]

            # Eq. (3): male hippopotamus position.
            y1 = self.generator.random()
            pos_male = (current + y1 * (dominant - i1 * current))
            agent_male = self.generate_agent(pos_male)
            # Eq. (8): greedy update.
            if self.compare_target(agent_male.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent_male

            # Eq. (5)
            temperature = np.exp(-epoch / self.epoch)
            # Eq. (6)
            if temperature > 0.6:
                pos_female = (current + h1 * (dominant - i2 * mean_group))
            else:
                # Eq. (7)
                if self.generator.random() > 0.5:
                    pos_female = (current + h2 * (mean_group - dominant))
                else:
                    r7 = self.generator.random()
                    pos_female = (self.problem.lb + r7 * (self.problem.ub - self.problem.lb))
            pos_female = self.correct_solution(pos_female)
            agent_female = self.generate_agent(pos_female)
            # Eq. (9): greedy update.
            if self.compare_target(agent_female.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent_female

        # ==============================================================
        # Phase 2: Hippopotamus defence against predators
        # Exploration, Eqs. (10)-(15)
        # ==============================================================
        for idx in range(n_half, self.pop_size):
            current = self.pop[idx].solution.copy()
            # Eq. (10): random predator position.
            r8 = self.generator.random(self.problem.n_dims)
            predator = (self.problem.lb + r8 * (self.problem.ub - self.problem.lb))
            predator_agent = self.generate_agent(predator)

            # Eq. (11)
            distance = np.abs(predator - current)
            # Parameters defined directly in the paper.
            f = self.generator.uniform(2.0, 4.0)
            c = self.generator.uniform(1.0, 1.5)
            d = self.generator.uniform(2.0, 3.0)
            g = self.generator.uniform(-1.0, 1.0)
            levy = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=self.problem.n_dims, case=-1)
            coefficient = (f / (c - d * np.cos(2.0 * np.pi * g)))
            predator_is_better = self.compare_target(predator_agent.target, self.pop[idx].target, self.problem.minmax)
            # Eq. (12)
            factor = np.where(distance == 0, self.EPSILON, distance)
            if not predator_is_better:
                r9 = self.generator.random(self.problem.n_dims)
                factor = 2.0 * distance + r9
            pos_new = levy * predator + coefficient * (1.0 / factor)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Eq. (15)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

        # ==============================================================
        # Phase 3: Hippopotamus escaping from the predator
        # Exploitation, Eqs. (16)-(19)
        # ==============================================================
        local_lb = self.problem.lb / epoch
        local_ub = self.problem.ub / epoch
        for idx in range(self.pop_size):
            current = self.pop[idx].solution.copy()
            # Eq. (18): randomly select one of the three scenarios.
            scenario = self.generator.integers(0, 3)
            if scenario == 0:
                s1 = (2.0 * self.generator.random(self.problem.n_dims) - 1.0)
            elif scenario == 1:
                s1 = self.generator.normal()
            else:
                s1 = self.generator.random()
            # Eq. (17)
            r10 = self.generator.random()
            pos_new = (current + r10 * (local_lb + s1 * (local_ub - local_lb)))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Eq. (19)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax, ):
                self.pop[idx] = agent
