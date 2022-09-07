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

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + seed_min (int): [1, 3], Number of Seeds (min)
        + seed_max (int): [4, 10], Number of Seeds (max)
        + exponent (int): [2, 4], Variance Reduction Exponent
        + sigma_start (float): (0.3, 1.0), The initial value of Standard Deviation
        + sigma_end (float): (0, 0.2), The final value of Standard Deviation

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
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> seed_min = 3
    >>> seed_max = 9
    >>> exponent = 3
    >>> sigma_start = 0.6
    >>> sigma_end = 0.01
    >>> model = OriginalIWO(epoch, pop_size, seed_min, seed_max, exponent, sigma_start, sigma_end)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics, 1(4), pp.355-366.
    """

    def __init__(self, epoch=10000, pop_size=100, seed_min=2, seed_max=10, exponent=2, sigma_start=1.0, sigma_end=0.01, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seed_min (int): Number of Seeds (min)
            seed_max (int): Number of seeds (max)
            exponent (int): Variance Reduction Exponent
            sigma_start (float): The initial value of standard deviation
            sigma_end (float): The final value of standard deviation
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.seed_min = self.validator.check_int("seed_min", seed_min, [1, 3])
        self.seed_max = self.validator.check_int("seed_max", seed_max, [4, int(self.pop_size/2)])
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.sigma_start = self.validator.check_float("sigma_start", sigma_start, [0.5, 5.0])
        self.sigma_end = self.validator.check_float("sigma_end", sigma_end, (0, 0.5))
        self.set_parameters(["epoch", "pop_size", "seed_min", "seed_max", "exponent", "sigma_start", "sigma_end"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch=None):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update Standard Deviation
        sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma_start - self.sigma_end) + self.sigma_end
        pop, best, worst = self.get_special_solutions(self.pop)
        pop_new = []
        nfe_epoch = 0
        for idx in range(0, self.pop_size):
            temp = best[0][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]
            if temp == 0:
                ratio = np.random.rand()
            else:
                ratio = (pop[idx][self.ID_TAR][self.ID_FIT] - worst[0][self.ID_TAR][self.ID_FIT]) / temp
            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            pop_local = []
            for j in range(s):
                # Initialize Offspring and Generate Random Location
                pos_new = pop[idx][self.ID_POS] + sigma * np.random.normal(0, 1, self.problem.n_dims)
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_local.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_local[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            if self.mode in self.AVAILABLE_MODES:
                pop_local = self.update_target_wrapper_population(pop_local)
            pop_new += pop_local
            nfe_epoch += s
        self.pop = self.get_sorted_strim_population(pop_new, self.pop_size)
        self.nfe_per_epoch = nfe_epoch
