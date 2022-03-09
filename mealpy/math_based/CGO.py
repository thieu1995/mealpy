#!/usr/bin/env python
# Created by "Thieu" at 22:24, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalCGO(Optimizer):
    """
    The original version of: Chaos Game Optimization (CGO)

    Links:
        1. https://doi.org/10.1007/s10462-020-09867-w

    Notes
    ~~~~~
    + 4th seed is mutation process, but it is not clear mutation on multiple variables or 1 variable
    + There is no usage of the variable alpha 4th in the paper
    + The replacement of worst solutions by generated seed are not clear (Lots of grammar errors in this section)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.CGO import OriginalCGO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalCGO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Talatahari, S. and Azizi, M., 2021. Chaos Game Optimization: a novel metaheuristic algorithm.
    Artificial Intelligence Review, 54(2), pp.917-1004.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 4*pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        pop_new = []
        for idx in range(0, self.pop_size):

            s1, s2, s3 = np.random.choice(range(0, self.pop_size), 3, replace=False)
            MG = (self.pop[s1][self.ID_POS] + self.pop[s2][self.ID_POS] + self.pop[s3][self.ID_POS]) / 3

            ## Calculating alpha based on Eq. 7
            alpha1 = np.random.rand()
            alpha2 = 2 * np.random.rand()
            alpha3 = 1 + np.random.random() * np.random.rand()
            esp = np.random.random()
            # There is no usage of this variable in the paper
            alpha4 = esp + esp * np.random.rand()

            beta = np.random.randint(0, 2, 3)
            gama = np.random.randint(0, 2, 3)
            ## The seed4 is mutation process, but not sure k is multiple variables or 1 variable.
            ## In the text said, multiple variables, but the defination of k is 1 variable. So confused
            k = np.random.randint(0, self.problem.n_dims)
            k_idx = np.random.choice(range(0, self.problem.n_dims), k, replace=False)

            seed1 = self.pop[idx][self.ID_POS] + alpha1 * (beta[0] * self.g_best[self.ID_POS] - gama[0] * MG)  # Eq. 3
            seed2 = self.g_best[self.ID_POS] + alpha2 * (beta[1] * self.pop[idx][self.ID_POS] - gama[1] * MG)  # Eq. 4
            seed3 = MG + alpha3 * (beta[2] * self.pop[idx][self.ID_POS] - gama[2] * self.g_best[self.ID_POS])  # Eq. 5
            seed4 = deepcopy(self.pop[idx][self.ID_POS]).astype(float)
            seed4[k_idx] += np.random.uniform(0, 1, k)

            # Check if solutions go outside the search space and bring them back
            seed1 = self.amend_position(seed1)
            seed2 = self.amend_position(seed2)
            seed3 = self.amend_position(seed3)
            seed4 = self.amend_position(seed4)

            sol1 = [seed1, self.get_fitness_position(seed1)]
            sol2 = [seed2, self.get_fitness_position(seed2)]
            sol3 = [seed3, self.get_fitness_position(seed3)]
            sol4 = [seed4, self.get_fitness_position(seed4)]

            ## Lots of grammar errors in this section, so confused to understand which strategy they are using
            _, best_seed = self.get_global_best_solution([sol1, sol2, sol3, sol4])
            pop_new.append(best_seed)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
