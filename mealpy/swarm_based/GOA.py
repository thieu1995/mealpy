# !/usr/bin/env python
# Created by "Thieu" at 14:53, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGOA(Optimizer):
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)

    Links:
        1. https://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        2. https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa

    Notes:
        + The matlab version above is not good, therefore
        + I added np.random.normal() component to Eq, 2.7
        + I changed the way to calculate distance between two location

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + c_minmax (list, tuple): (c_min, c_max) -> ([0.00001, 0.01], [0.5, 2.0]), coefficient c, default = (0.00004, 1)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GOA import BaseGOA
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
    >>> c_minmax = [0.00004, 1]
    >>> model = BaseGOA(problem_dict1, epoch, pop_size, c_minmax)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Saremi, S., Mirjalili, S. and Lewis, A., 2017. Grasshopper optimisation algorithm:
    theory and application. Advances in Engineering Software, 105, pp.30-47.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, c_minmax=(0.00004, 1), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_minmax (list, tuple): coefficient c, default = (0.00004, 1)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c_minmax = self.validator.check_tuple_float("c (min, max)", c_minmax, ([0.00001, 0.2], [0.2, 2.0]))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def _s_function__(self, r_vector=None):
        f = 0.5
        l = 1.5
        # Eq.(2.3) in the paper
        return f * np.exp(-r_vector / l) - np.exp(-r_vector)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq.(2.8) in the paper
        c = self.c_minmax[1] - epoch * ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)

        pop_new = []
        for idx in range(0, self.pop_size):
            S_i_total = np.zeros(self.problem.n_dims)
            for j in range(0, self.pop_size):
                dist = np.sqrt(np.sum((self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]) ** 2))
                r_ij_vector = (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]) / (dist + self.EPSILON)  # xj - xi / dij in Eq.(2.7)
                xj_xi = 2 + np.remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
                ## The first part inside the big bracket in Eq. (2.7)   16 955 230 764    212 047 193 643
                ran = (c / 2) * (self.problem.ub - self.problem.lb)
                s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
                S_i_total += s_ij
            x_new = c * np.random.normal() * S_i_total + self.g_best[self.ID_POS]  # Eq. (2.7) in the paper
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
