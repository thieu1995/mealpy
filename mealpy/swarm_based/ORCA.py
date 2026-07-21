#!/usr/bin/env python
# Created by "Thieu" at 17:45, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalOrcaOA(Optimizer):
    """
    The original version of: Orca Optimization Algorithm (OrcaOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    p_percent : float
        The percentage of worst orcas to remove and regenerate per iteration, default = 0.1.
    R0 : float
        The initial radius of the ice floe, default=2.0.

    References
    ~~~~~~~~~~
    1. Golilarz, N. A., Gao, H., Addeh, A., & Pirasteh, S. (2020, December).
       ORCA optimization algorithm: A new meta-heuristic tool for complex optimization problems.
       In 2020 17th International Computer Conference on Wavelet Active Media Technology and
       Information Processing (ICCWAMTIP) (pp. 198-204). IEEE. https://doi.org/10.1109/ICCWAMTIP51612.2020.9317473

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ORCA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = ORCA.OriginalOrcaOA(epoch=1000, pop_size=50, p_percent=0.15, R0=5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_percent: float=0.1, R0: float=2.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_percent = self.validator.check_float("p_percent", p_percent, [0, 1.0])
        self.R0 = self.validator.check_float("R0", R0, [-1000, 1000])
        self.set_parameters(["epoch", "pop_size", "p_percent", "R0"])
        self.R = self.R0
        self.sort_flag = True
        self.n_removes = int(self.p_percent * pop_size)

    def evolve(self, epoch: int):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        # Calculate energy (e_i = F_i - F_s)
        e = fit_list - self.g_best.target.fitness
        e_min = np.min(e)
        e_max = np.max(e)

        # Compute the normalized energy of each orca
        if e_max == e_min:
            epsilon = np.zeros(self.pop_size)
        else:
            epsilon = (e - e_min) / (e_max - e_min)
        # Calculate d_i
        d_list = epsilon * self.R

        pop_new = []
        # Move the orcas toward the Seal and ice floe
        for idx in range(1, self.pop_size):  # Keep the best solution intact (elitism)
            if idx < self.pop_size - self.n_removes:
                # Generate random direction (unit vector)
                direction = self.generator.standard_normal(self.problem.n_dims)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                # Random position between R-d_i and R
                r_dist = self.generator.uniform(self.R - d_list[idx], self.R)
                # Update position
                pos_new = self.pop[0].solution + r_dist * direction
            else:
                # Remove the worst orcas (P%) and generate them randomly for the next iteration
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.correct_solution(pos_new)
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        # Evaluate fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new

        # Reduce the radius of the ice floe (simulated linear decay)
        self.R = self.R0 * (1. - epoch / self.epoch)
