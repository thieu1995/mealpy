# !/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BasePFA(Optimizer):
    """
    The original version of: Pathfinder Algorithm (PFA)

    Links:
        1. https://doi.org/10.1016/j.asoc.2019.03.012

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PFA import BasePFA
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
    >>> model = BasePFA(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yapici, H. and Cetinkaya, N., 2019. A new meta-heuristic optimizer: Pathfinder algorithm.
    Applied soft computing, 78, pp.545-568.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        alpha, beta = np.random.uniform(1, 2, 2)
        A = np.random.uniform(self.problem.lb, self.problem.ub) * np.exp(-2 * (epoch + 1) / self.epoch)

        ## Update the position of pathfinder and check the bound
        pos_new = self.pop[0][self.ID_POS] + 2 * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[0][self.ID_POS]) + A
        pos_new = self.amend_position(pos_new)
        fit = self.get_fitness_position(pos_new)
        pop_new = [[pos_new, fit], ]

        ## Update positions of members, check the bound and calculate new fitness
        for idx in range(1, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS]).astype(float)
            for k in range(1, self.pop_size):
                dist = np.sqrt(np.sum((self.pop[k][self.ID_POS] - self.pop[idx][self.ID_POS]) ** 2)) / self.problem.n_dims
                t2 = alpha * np.random.uniform() * (self.pop[k][self.ID_POS] - self.pop[idx][self.ID_POS])
                ## First stabilize the distance
                t3 = np.random.uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.problem.ub - self.problem.lb))
                pos_new += t2 + t3
            ## Second stabilize the population size
            t1 = beta * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (pos_new + t1) / self.pop_size
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
