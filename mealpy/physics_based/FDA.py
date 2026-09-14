#!/usr/bin/env python
# Created by "Thieu" at 10:39, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalFDA(Optimizer):
    """
    The original version of: Flow Direction Algorithm (FDA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Number of flows in the population, in range [5, 10000]. Default is 30.
    beta : int
        Number of neighboring positions generated around each flow, in range [1, 100]. The FDA is inspired
        by the D8 method, where eight neighboring directions are considered. Default is 8.

    Warnings
    --------
    The mathematical expressions reported in the FDA paper differ from the authors' reference MATLAB
    implementation in the slope and flow-update formulas. This implementation is followed by the paper.

    Note
    ----
    FDA generates `beta` neighbors for every flow and then evaluates one
    additional candidate flow. Therefore, each iteration requires approximately
    `pop_size * (beta + 1)` objective-function evaluations.

    The original formulation is explicitly described for minimization problems
    and compares solutions using lower objective-function values. Therefore,
    this implementation requires `minmax="min"`.

    The neighborhood radius is not an independent control parameter. It is
    dynamically calculated using Eqs. (4) and (5).

    The original paper does not explicitly define a numerical strategy for
    divisions by zero in the slope and direction equations. Small numerical
    safeguards are applied only to avoid undefined floating-point operations.

    References
    ----------
    1. Karami, H., Valikhan Anaraki, M., Farzin, S., & Mirjalili, S. (2021).
       Flow Direction Algorithm (FDA): A Novel Optimization Approach for Solving Optimization Problems.
       Computers & Industrial Engineering, 156, 107224. https://doi.org/10.1016/j.cie.2021.107224

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, FDA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = FDA.OriginalFDA(epoch=1000, pop_size=30, beta=8)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Flow Direction Algorithm", year=2021, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 30, beta: int = 8, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Number of flows in the population, default = 30.
            beta (int): Number of neighboring positions for each flow, default = 8.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.beta = self.validator.check_int("beta", beta, [1, 100])
        self.set_parameters(["epoch", "pop_size", "beta"])
        self.sort_flag = False

    def calculate_weight(self, epoch):
        """
        Calculate the nonlinear weight W according to Eq. (5).
        """
        ratio = epoch / self.epoch
        # At the final iteration, the base 1 - ratio becomes zero.
        # A tiny positive value prevents undefined 0**negative operations
        # without changing the search equation during ordinary iterations.
        base = max(1.0 - ratio, np.finfo(float).tiny)
        randn = self.generator.normal()
        w = (base ** (2.0 * randn) * (self.generator.random(self.problem.n_dims) * ratio) * self.generator.random(self.problem.n_dims))
        return w

    def evolve(self, epoch):
        """
        The main operations of the Flow Direction Algorithm.

        Args:
            epoch (int): The current iteration.
        """
        # Eq. (5)
        w = self.calculate_weight(epoch)
        # Use the current best flow as the basin outlet.
        best_pos = self.g_best.solution.copy()
        pop_new = []
        for idx in range(self.pop_size):
            current = self.pop[idx].solution
            current_fit = self.pop[idx].target.fitness
            neighbors = []
            # ----------------------------------------------------------
            # Generate beta neighboring positions, Eqs. (3)-(5)
            # ----------------------------------------------------------
            for _ in range(self.beta):
                # Random position Xrand.
                x_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
                # Eq. (4)
                delta = ((self.generator.random(self.problem.n_dims) * x_rand - self.generator.random(self.problem.n_dims) * current) * np.linalg.norm(best_pos - current) * w)
                # Eq. (3)
                neighbor_pos = current + self.generator.normal(0.0, 1.0, self.problem.n_dims) * delta
                neighbor_pos = self.correct_solution(neighbor_pos)
                neighbor = self.generate_empty_agent(neighbor_pos)
                if self.mode not in self.AVAILABLE_MODES:
                    neighbor.target = self.get_target(neighbor_pos)
                neighbors.append(neighbor)
            if self.mode in self.AVAILABLE_MODES:
                neighbors = self.update_target_for_population(neighbors)

            # Select the neighbor with the lowest objective value.
            best_neighbor, _ = self.get_best_agent(neighbors, self.problem.minmax)

            # ----------------------------------------------------------
            # Best neighbor is better than current flow: Eqs. (6)-(8)
            # ----------------------------------------------------------
            if best_neighbor.target.fitness < current_fit:
                diff = (current - best_neighbor.solution)
                # Eq. (7)
                # The paper defines the slope component-wise through the
                # positional distance between the flow and its neighbor.
                denominator = np.abs(diff)
                denominator = np.where(denominator <= self.EPSILON, self.EPSILON, denominator)
                slope = (current_fit - best_neighbor.target.fitness) / denominator

                # Eq. (6)
                velocity = (self.generator.normal(0.0, 1.0, self.problem.n_dims, ) * slope)
                # Eq. (8)
                distance = np.linalg.norm(diff)
                if distance <= self.EPSILON:
                    pos_new = current.copy()
                else:
                    pos_new = (current + velocity * diff / distance)

            # ----------------------------------------------------------
            # Sink-filling mechanism, Eq. (9)
            # ----------------------------------------------------------
            else:
                r_idx = self.generator.integers(self.pop_size)
                if self.pop[r_idx].target.fitness < current_fit:
                    pos_new = current + self.generator.normal(0.0, 1.0, self.problem.n_dims) * (self.pop[r_idx].solution - current)
                else:
                    pos_new = current + 2.0 * self.generator.normal(0.0, 1.0, self.problem.n_dims) * (best_pos - current)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Step 8: update a flow only if the new flow is better.
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
