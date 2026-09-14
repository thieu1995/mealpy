#!/usr/bin/env python
# Created by "Thieu" at 05:48, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalAHA(Optimizer):
    """
    The original version of: Artificial Hummingbird Algorithm (AHA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Number of hummingbirds and food sources, in range [2, 10000]. Default is 50.

    Note
    ----
    The migration coefficient is not exposed as a control parameter because
    the original paper recommends `M = 2 * pop_size`.

    Each regular iteration evaluates one new candidate for every hummingbird, while migration introduces
    one additional evaluation every `2 * pop_size` iterations. Therefore, the exact function-evaluation
    budget is slightly larger than `epoch * pop_size`.

    The visit table is updated sequentially after each hummingbird performs guided or territorial foraging.
    This sequential behavior is preserved in this implementation.

    The paper does not explicitly specify a boundary-handling rule for newly generated food sources.
    Mealpy's standard solution correction is applied before objective-function evaluation.

    References
    ----------
    1. Zhao, W., Wang, L., & Mirjalili, S. (2022).
       Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications.
       Computer Methods in Applied Mechanics and Engineering, 388, 114194.
       https://doi.org/10.1016/j.cma.2021.114194

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, AHA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>> model = AHA.OriginalAHA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Artificial Hummingbird Algorithm", year=2022, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Number of hummingbirds and food sources, default = 50.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        # The visit table and population are updated sequentially.
        self.is_parallelizable = False

    def initialize_variables(self):
        """
        Initialize the visit table according to Eq. (2).
        """
        self.visit_table = np.zeros((self.pop_size, self.pop_size), dtype=float)
        # VT(i, i) = null in Eq. (2).
        np.fill_diagonal(self.visit_table, np.nan)
        # Eq. (12)
        self.migration_coefficient = 2 * self.pop_size

    def get_direction_mask(self):
        """
        Generate the direction switch vector according to Eqs. (3)-(5).
        """
        n_dims = self.problem.n_dims
        r = self.generator.random()

        # Eq. (3): axial flight.
        if r < 1.0 / 3.0:
            direction = np.zeros(n_dims)
            idx = self.generator.integers(n_dims)
            direction[idx] = 1.0

        # Eq. (4): diagonal flight.
        elif r < 2.0 / 3.0:
            if n_dims == 1:
                direction = np.ones(1)
            elif n_dims == 2:
                direction = np.ones(2)
            else:
                r1 = self.generator.random()
                n_selected = int(np.ceil(r1 * (n_dims - 2))) + 1
                indices = self.generator.choice(n_dims, size=n_selected, replace=False)
                direction = np.zeros(n_dims)
                direction[indices] = 1.0
        # Eq. (5): omnidirectional flight.
        else:
            direction = np.ones(n_dims)
        return direction

    def get_target_source(self, idx):
        """
        Select the target food source for guided foraging.

        The food source with the highest visit level is selected. If several
        sources share that level, the one with the best fitness is selected.
        """
        row = self.visit_table[idx]
        max_visit = np.nanmax(row)
        candidates = np.flatnonzero(row == max_visit)
        target_idx = candidates[0]
        for candidate_idx in candidates[1:]:
            if self.compare_target(self.pop[candidate_idx].target, self.pop[target_idx].target, self.problem.minmax):
                target_idx = candidate_idx
        return target_idx

    def update_visit_row(self, idx, visited_idx=None):
        """
        Increase visit levels in one hummingbird row.

        If a target food source is visited, its visit level is reset to zero.
        """
        for jdx in range(self.pop_size):
            if jdx == idx:
                continue
            if visited_idx is not None and jdx == visited_idx:
                continue
            self.visit_table[idx, jdx] += 1
        if visited_idx is not None:
            self.visit_table[idx, visited_idx] = 0

    def update_visit_column(self, source_idx):
        """
        Update the visit level of a newly replaced food source for all
        other hummingbirds.
        """
        for idx in range(self.pop_size):
            if idx == source_idx:
                continue
            max_visit = np.nanmax(self.visit_table[idx])
            self.visit_table[idx, source_idx] = max_visit + 1

    def guided_foraging(self, idx):
        """
        Perform guided foraging according to Eqs. (6)-(8).
        """
        target_idx = self.get_target_source(idx)
        direction = self.get_direction_mask()
        # Eq. (7)
        a = self.generator.normal()
        # Eq. (6)
        pos_new = (self.pop[target_idx].solution + a * direction * (self.pop[idx].solution - self.pop[target_idx].solution))
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)

        # The target food source is considered visited regardless of whether
        # the newly generated food source replaces the current one.
        self.update_visit_row(idx, visited_idx=target_idx, )
        # Eq. (8)
        if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax, ):
            self.pop[idx] = agent
            # The food source occupied by hummingbird idx has changed.
            self.update_visit_column(idx)

    def territorial_foraging(self, idx):
        """
        Perform territorial foraging according to Eqs. (9)-(10).
        """
        direction = self.get_direction_mask()
        # Eq. (10)
        b = self.generator.normal()
        # Eq. (9)
        pos_new = (self.pop[idx].solution + b * direction * self.pop[idx].solution)
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)
        # All other food sources become one time unit less recently visited.
        self.update_visit_row(idx)
        if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
            self.pop[idx] = agent
            # The current food source has been replaced.
            self.update_visit_column(idx)

    def migration_foraging(self):
        """
        Perform migration foraging according to Eqs. (11)-(12).
        """
        worst_idx = 0
        for idx in range(1, self.pop_size):
            if self.compare_target(self.pop[worst_idx].target, self.pop[idx].target, self.problem.minmax):
                worst_idx = idx
        # Eq. (11)
        pos_new = self.problem.generate_solution(encoded=True)
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)
        # Migration replaces the worst food source unconditionally.
        self.pop[worst_idx] = agent
        # Algorithm 4: update the corresponding visit-table row.
        self.update_visit_row(worst_idx)
        # The migrated food source is new to every other hummingbird.
        self.update_visit_column(worst_idx)

    def evolve(self, epoch):
        """
        The main operations of the Artificial Hummingbird Algorithm.

        Args:
            epoch (int): The current iteration.
        """
        # Guided and territorial foraging are selected independently
        # for every hummingbird with equal probability.
        for idx in range(self.pop_size):
            if self.generator.random() < 0.5:
                self.guided_foraging(idx)
            else:
                self.territorial_foraging(idx)

        # Eq. (12) and Algorithm 4.
        if epoch % self.migration_coefficient == 0:
            self.migration_foraging()
