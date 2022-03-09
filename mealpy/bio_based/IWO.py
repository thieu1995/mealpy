#!/usr/bin/env python
# Created by "Thieu" at 12:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalIWO(Optimizer):
    """
    The original version of: Invasive Weed Optimization (IWO)

    Links:
        1. https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf

    Notes
    ~~~~~
    Better to use normal distribution instead of uniform distribution, updating population by sorting
    both parent population and child population

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + seeds (list): (min_value, max_value) -> ([1, 3], [5, 10]), Number of Seeds
        + exponent (int): [2, 4], Variance Reduction Exponent
        + sigma (list): (initial_value, final_value), ([0.5, 0.9], [0.001, 0.1]), Value of Standard Deviation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.IWO import OriginalIWO
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
    >>> seeds = [3, 9]
    >>> exponent = 3
    >>> sigma = [0.6, 0.01]
    >>> model = OriginalIWO(problem_dict1, epoch, pop_size, seeds, exponent, sigma)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics, 1(4), pp.355-366.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, seeds=(2, 10), exponent=2, sigma=(0.5, 0.001), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seeds (list): (Min, Max) Number of Seeds
            exponent (int): Variance Reduction Exponent
            sigma (list): (Initial, Final) Value of Standard Deviation
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.seeds = seeds
        self.exponent = exponent
        self.sigma = sigma

    def evolve(self, epoch=None):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update Standard Deviation
        sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma[0] - self.sigma[1]) + self.sigma[1]
        pop, best, worst = self.get_special_solutions(self.pop)
        pop_new = []
        for idx in range(0, self.pop_size):
            temp = best[0][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]
            if temp == 0:
                ratio = 0.5
            else:
                ratio = (pop[idx][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]) / temp
            s = int(np.ceil(self.seeds[0] + (self.seeds[1] - self.seeds[0]) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            pop_local = []
            for j in range(s):
                # Initialize Offspring and Generate Random Location
                pos_new = pop[idx][self.ID_POS] + sigma * np.random.normal(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(pos_new)
                pop_local.append([pos_new, None])
            pop_local = self.update_fitness_population(pop_local)
            pop_new += pop_local
        self.pop = self.get_sorted_strim_population(pop_new, self.pop_size)
