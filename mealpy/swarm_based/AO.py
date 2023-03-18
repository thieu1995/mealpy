#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
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
    >>> model = OriginalAO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
    Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        alpha = delta = 0.1
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
            levy_step = self.get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)
            if (epoch + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if np.random.rand() < 0.5:
                    pos_new = self.g_best[self.ID_POS] * (1 - (epoch + 1) / self.epoch) + \
                              np.random.rand() * (x_mean - self.g_best[self.ID_POS])
                else:
                    idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = self.g_best[self.ID_POS] * levy_step + self.pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
            else:
                if np.random.rand() < 0.5:
                    pos_new = alpha * (self.g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                              (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * delta  # Eq. 13
                else:
                    pos_new = QF * self.g_best[self.ID_POS] - (g2 * self.pop[idx][self.ID_POS] * np.random.rand()) - \
                              g2 * levy_step + np.random.rand() * g1  # Eq. 14
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
