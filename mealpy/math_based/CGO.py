#!/usr/bin/env python
# Created by "Thieu" at 22:24, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCGO(Optimizer):
    """
    The original version of: Chaos Game Optimization (CGO)

    Links:
        1. https://doi.org/10.1007/s10462-020-09867-w

    Notes:
        + 4th seed is mutation process, but it is not clear mutation on multiple variables or 1 variable
        + There is no usage of the variable alpha 4th in the paper
        + The replacement of the worst solutions by generated seed are not clear (Lots of grammar errors in this section)

    Examples
    ~~~~~~~~

    >>> import numpy as np
    >>> from mealpy import FloatVar, CGO
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
    >>> model = CGO.OriginalCGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Talatahari, S. and Azizi, M., 2021. Chaos Game Optimization: a novel metaheuristic algorithm.
    Artificial Intelligence Review, 54(2), pp.917-1004.
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
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            s1, s2, s3 = self.generator.choice(range(0, self.pop_size), 3, replace=False)
            MG = (self.pop[s1].solution + self.pop[s2].solution + self.pop[s3].solution) / 3
            ## Calculating alpha based on Eq. 7
            alpha1 = self.generator.random()
            alpha2 = 2 * self.generator.random()
            alpha3 = 1 + self.generator.random() * self.generator.random()
            esp = self.generator.random()
            # There is no usage of this variable in the paper
            alpha4 = esp + esp * self.generator.random()
            beta = self.generator.integers(0, 2, 3)
            gama = self.generator.integers(0, 2, 3)
            ## The seed4 is mutation process, but not sure k is multiple variables or 1 variable.
            ## In the text said, multiple variables, but the defination of k is 1 variable. So confused
            k = self.generator.integers(0, self.problem.n_dims)
            k_idx = self.generator.choice(range(0, self.problem.n_dims), k, replace=False)
            seed1 = self.pop[idx].solution + alpha1 * (beta[0] * self.g_best.solution - gama[0] * MG)  # Eq. 3
            seed2 = self.g_best.solution + alpha2 * (beta[1] * self.pop[idx].solution - gama[1] * MG)  # Eq. 4
            seed3 = MG + alpha3 * (beta[2] * self.pop[idx].solution - gama[2] * self.g_best.solution)  # Eq. 5
            seed4 = self.pop[idx].solution.copy().astype(float)
            seed4[k_idx] += self.generator.uniform(0, 1, k)
            # Check if solutions go outside the search space and bring them back
            seed1 = self.correct_solution(seed1)
            seed2 = self.correct_solution(seed2)
            seed3 = self.correct_solution(seed3)
            seed4 = self.correct_solution(seed4)
            agent1 = self.generate_agent(seed1)
            agent2 = self.generate_agent(seed2)
            agent3 = self.generate_agent(seed3)
            agent4 = self.generate_agent(seed4)
            ## Lots of grammar errors in this section, so confused to understand which strategy they are using
            best_seed = self.get_best_agent([agent1, agent2, agent3, agent4], self.problem.minmax)
            self.pop[idx] = self.get_better_agent(best_seed, self.pop[idx], self.problem.minmax)
