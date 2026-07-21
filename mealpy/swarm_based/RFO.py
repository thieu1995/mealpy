#!/usr/bin/env python
# Created by "Thieu" at 17:31, 12/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRFO(Optimizer):
    """
    The original version of: Red Fox Optimization (RFO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    phi_0 : float
        Fox observation angle set at the beginning. Default is 0.785 (pi/4).
    theta : float
        Weather conditions parameter. Default is 0.5.

    References
    ----------
    1. Połap, Dawid, and Marcin Woźniak. "Red fox optimization algorithm."
       Expert Systems with Applications 166 (2021): 114107. https://doi.org/10.1016/j.eswa.2020.114107

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, RFO
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
    >>> model = RFO.OriginalRFO(epoch=1000, pop_size=50, phi_0=0.785, theta=0.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size: int = 100, phi_0: float = 0.785, theta: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.phi_0 = self.validator.check_float("phi_0", phi_0, [0.0, 3.14])
        self.theta = self.validator.check_float("theta", theta, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "phi_0", "theta"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(self.pop_size):
            # Phase 1: Global Search (In search for food)
            # Calculate euclidean distance to the best individual
            dist = np.sqrt(np.sum((self.pop[idx].solution - self.g_best.solution) ** 2))
            # Define scaling parameter alpha
            alpha = self.generator.uniform(0, dist) if dist > 0 else 0
            # Calculate reallocation according to Eq. (2)
            pos_new = self.pop[idx].solution + alpha * np.sign(self.g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # If reallocation is better, move the fox; else return to previous
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

            # Phase 2: Local Search (Traversing through the local habitat)
            if self.generator.random() > 0.75:  # Fox is not noticed, move closer
                a = self.generator.uniform(0, 0.2)  # Fox approaching change
                # Calculate fox observation radius r according to Eq. (4)
                if self.phi_0 == 0:
                    r = self.theta
                else:
                    r = a * (np.sin(self.phi_0) / self.phi_0)
                # Calculate reallocation according to Eq. (5)
                phi = self.generator.uniform(0, 2 * np.pi, self.problem.n_dims-1)
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                offsets = np.zeros(self.problem.n_dims)
                ar = a * r
                current_sin_sum = 0
                for d in range(self.problem.n_dims - 1):
                    offsets[d] = ar * current_sin_sum + ar * cos_phi[d]
                    current_sin_sum += sin_phi[d]
                offsets[-1] = ar * current_sin_sum
                pos_new = self.pop[idx].solution + offsets
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = agent

        # Phase 3: Reproduction and leaving the herd
        # Sort population again before reproduction phase
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)

        # Select two best individuals to represent the alpha couple
        x_alpha_1 = self.pop[0].solution
        x_alpha_2 = self.pop[1].solution

        # Calculate number of worst foxes to be replaced (5% of population)
        num_worst = max(1, int(0.05 * self.pop_size))

        # Replace the worst foxes (5%)
        for w in range(self.pop_size - num_worst, self.pop_size):
            kappa = self.generator.uniform(0, 1)
            if kappa >= 0.45:
                # New nomadic individual outside habitat
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            else:
                # Reproduction of the alpha couple Eq. (9)
                pos_new = kappa * (x_alpha_1 + x_alpha_2) / 2
            pos_new = self.correct_solution(pos_new)
            self.pop[w] = self.generate_agent(pos_new)
