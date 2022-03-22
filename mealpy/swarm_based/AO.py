# !/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from math import gamma
from mealpy.optimizer import Optimizer


class OriginalAO(Optimizer):
    """
    The original version of: Aquila Optimization (AO)

    Links:
        1. https://doi.org/10.1016/j.cie.2021.107250

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.AO import OriginalAO
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
    >>> model = OriginalAO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
    Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.alpha = 0.1
        self.delta = 0.1
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def get_simple_levy_step(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, self.problem.n_dims) * sigma
        v = np.random.normal(1, self.problem.n_dims)
        step = u / abs(v) ** (1 / beta)
        return step

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        g1 = 2 * np.random.rand() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17

        dim_list = np.array(list(range(1, self.problem.n_dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = (epoch + 1) ** ((2 * np.random.rand() - 1) / (1 - self.epoch) ** 2)  # Eq.(15)        Quality function

        pop_new = []
        for idx in range(0, self.pop_size):
            x_mean = np.mean(np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop]), axis=0)
            if (epoch + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if np.random.rand() < 0.5:
                    pos_new = self.g_best[self.ID_POS] * (1 - (epoch + 1) / self.epoch) + \
                              np.random.rand() * (x_mean - self.g_best[self.ID_POS])
                else:
                    idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = self.g_best[self.ID_POS] * self.get_simple_levy_step() + \
                              self.pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
            else:
                if np.random.rand() < 0.5:
                    pos_new = self.alpha * (self.g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                              (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * self.delta  # Eq. 13
                else:
                    pos_new = QF * self.g_best[self.ID_POS] - (g2 * self.pop[idx][self.ID_POS] * np.random.rand()) - \
                              g2 * self.get_simple_levy_step() + np.random.rand() * g1  # Eq. 14
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
