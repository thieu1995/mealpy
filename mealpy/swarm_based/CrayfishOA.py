#!/usr/bin/env python
# Created by "Thieu" at 18:31, 12/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCrayfishOA(Optimizer):
    """
    The original version of: Crayfish Optimization Algorithm (COA)

    Hyperparameters
    ----------------
    + epoch (int): maximum number of iterations, default = 10000
    + pop_size (int): number of population size, default = 100

    References
    ----------
    .. [1] Jia, H., Rao, H., Wen, C., & Mirjalili, S. (2023). Crayfish optimization algorithm.
    Artificial Intelligence Review, 56(Suppl 2), 1919-1979. https://doi.org/10.1007/s10462-023-10567-4

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CrayfishOA
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
    >>> model = CrayfishOA.OriginalCrayfishOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def p_obj(self, x: float, c1: float=0.2, sigma:float=3.0, miu:float=25) -> float:
        """
        Calculate the probability object function value (Eq. 4).

        Returns:
            float: Evaluated probability value.
        """
        return c1 * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x - miu) ** 2 / (2 * sigma ** 2))

    def evolve(self, epoch: int):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Define Parameters for the current epoch
        C = 2.0 - (epoch / self.epoch)  # Eq.(7)
        temp = self.generator.random() * 15 + 20  # Eq.(3)
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        xf = (self.g_best.solution + current_best.solution) / 2.0  # Eq.(5)
        Xfood = self.g_best.solution.copy()

        pop_new = []
        for idx in range(self.pop_size):
            if temp > 30:
                # --- Summer resort stage ---
                if self.generator.random() < 0.5:
                    # Eq.(6)
                    pos_new = self.pop[idx].solution + C * self.generator.random(self.problem.n_dims) * (xf - self.pop[idx].solution)
                else:
                    # --- Competition stage ---
                    pos_new = np.zeros(self.problem.n_dims)
                    for jdx in range(self.problem.n_dims):
                        z = self.generator.integers(0, self.pop_size)       # Eq.(9)
                        pos_new[jdx] = self.pop[idx].solution[jdx] - self.pop[z].solution[jdx] + xf[jdx]    # Eq.(8)
            else:
                # --- Foraging stage ---
                # Eq.(4) - Add epsilon to prevent division by zero
                P = 3 * self.generator.random() * self.pop[idx].target.fitness / (self.g_best.target.fitness + self.EPSILON)
                if P > 2:  # The food is too big
                    # Eq.(12) - Update Xfood sequentially
                    Xfood = np.exp(-1 / P) * Xfood
                    rv1 = self.generator.random(self.problem.n_dims)
                    rv2 = self.generator.random(self.problem.n_dims)
                    # Eq.(13)
                    pos_new = self.pop[idx].solution + Xfood * self.p_obj(temp) * (np.cos(2 * np.pi * rv1) - np.sin(2 * np.pi * rv2))
                else:
                    # Eq.(14)
                    pos_new = (self.pop[idx].solution - Xfood + self.generator.random(self.problem.n_dims) * self.pop[idx].solution) * self.p_obj(temp)

            # Boundary conditions handling
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

        # Evaluate the newly generated population
        if self.mode not in self.AVAILABLE_MODES:
            for agent in pop_new:
                agent.target = self.get_target(agent.solution)
        else:
            pop_new = self.update_target_for_population(pop_new)
        # Update the population to a new location (Greedy Selection)
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
