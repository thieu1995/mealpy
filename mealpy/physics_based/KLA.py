#!/usr/bin/env python
# Created by "Thieu" at 11:07, 10/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalKLA(Optimizer):
    """
    The original version of: Kirchhoff's Law Algorithm (KLA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [10, 10000]. Default is 100.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/181589-kirchhoff-s-law-algorithm-kla
    2. https://doi.org/10.1007/s10462-025-11289-5

    References
    ~~~~~~~~~~
    1. Ghasemi, Mojtaba, Nima Khodadadi, Pavel Trojovský, Li Li, Zulkefli Mansor, Laith Abualigah, Amal H. Alharbi, and El-Sayed M. El-Kenawy.
       "Kirchhoff’s law algorithm (KLA): A novel physics-inspired non-parametric metaheuristic algorithm for optimization problems."
       Artificial Intelligence Review 58, no. 10 (2025): 325.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, KLA
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
    >>> model = KLA.OriginalKLA(epoch=100, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])

    def evolve(self, epoch: int) -> None:
        """
        Args:
            epoch: The current iteration
        """
        # Iterate through each agent in the population
        pop_new = []
        for idx in range(self.pop_size):
            # Select 3 distinct random indices excluding the current one (i)
            a, b, jj = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=3)

            # Get fitness values
            f_i = self.pop[idx].target.fitness
            f_a = self.pop[a].target.fitness
            f_b = self.pop[b].target.fitness
            f_jj = self.pop[jj].target.fitness

            # Calculate movement factors (Q parameters)
            q = (f_i - f_jj) / (np.abs(f_i - f_jj) + self.EPSILON)
            Q = (f_i - f_a) / (np.abs(f_i - f_a) + self.EPSILON)
            Q2 = (f_i - f_b) / (np.abs(f_i - f_b) + self.EPSILON)

            # Calculate random components
            q1 = (f_jj / (f_i + self.EPSILON)) ** (2 * self.generator.random())
            Q1 = (f_a / (f_i + self.EPSILON)) ** (2 * self.generator.random())
            Q21 = (f_b / (f_i + self.EPSILON)) ** (2 * self.generator.random())

            # Calculate steps S1, S2, S3
            s1 = q1 * q * self.generator.random(self.problem.n_dims) * (self.pop[jj].solution - self.pop[idx].solution)
            s2 = Q * Q1 * self.generator.random(self.problem.n_dims) * (self.pop[a].solution - self.pop[idx].solution)
            s3 = Q2 * Q21 * self.generator.random(self.problem.n_dims) * (self.pop[b].solution - self.pop[idx].solution)

            # Sum of steps
            s = (self.generator.random() + self.generator.random()) * s1 + \
                (self.generator.random() + self.generator.random()) * s2 + \
                (self.generator.random() + self.generator.random()) * s3

            # Update position
            pos_new = self.pop[idx].solution + s
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
