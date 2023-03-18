#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MRFO import OriginalMRFO
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
    >>> somersault_range = 2.0
    >>> model = OriginalMRFO(epoch, pop_size, somersault_range)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    def __init__(self, epoch=10000, pop_size=100, somersault_range=2.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.somersault_range = self.validator.check_float("somersault_range", somersault_range, [1.0, 5.0])
        self.set_parameters(["epoch", "pop_size", "somersault_range"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if np.random.rand() < 0.5:
                r1 = np.random.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if (epoch + 1) / self.epoch < np.random.rand():
                    x_rand = np.random.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = x_rand + np.random.uniform() * (x_rand - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = x_rand + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                else:
                    if idx == 0:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            # Chain foraging (Eq. 1,2)
            else:
                r = np.random.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(self.pop, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.pop[idx][self.ID_POS] + self.somersault_range * \
                   (np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child)


class WMQIMRFO(Optimizer):
    """
    The original version of: Wavelet Mutation and Quadratic Interpolation MRFO

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.108071

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2
        + pm (float): (0.0, 1.0), probability mutation, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MRFO import WMQIMRFO
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
    >>> somersault_range = 2.0
    >>> model = OriginalMRFO(epoch, pop_size, somersault_range)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] G. Hu, M. Li, X. Wang et al., An enhanced manta ray foraging optimization algorithm for shape optimization of
    complex CCG-Ball curves, Knowledge-Based Systems (2022), doi: https://doi.org/10.1016/j.knosys.2021.108071.
    """

    def __init__(self, epoch=10000, pop_size=100, somersault_range=2.0, pm=0.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
            pm (float): probability mutation, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.somersault_range = self.validator.check_float("somersault_range", somersault_range, [1.0, 5.0])
        self.pm = self.validator.check_float("pm", pm, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "somersault_range", "pm"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            x_t = self.pop[idx][self.ID_POS]
            x_t1 = self.pop[idx-1][self.ID_POS]

            ## Morlet wavelet mutation strategy
            ## Goal is to jump out of local optimum --> Performed in exploration stage
            s_constant = 2.0
            a = s_constant * (1.0 / s_constant) ** (1.0 - (epoch + 1) / self.epoch)
            theta = np.random.uniform(-2.5 * a, 2.5 * a)
            x = theta / a
            w = np.exp(-x ** 2 / 2) * np.cos(5 * x)
            xichma = 1.0 / np.sqrt(a) * w

            if np.random.rand() < 0.5:  # Control parameter adjustment
                coef = np.log(1 + (np.e - 1) * (epoch + 1) / self.epoch)  # Eq. 3.11

                r1 = np.random.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if coef < np.random.rand():     # Cyclone foraging
                    x_rand = self.generate_position(self.problem.lb, self.problem.ub)

                    if np.random.rand() < self.pm:      # Morlet wavelet mutation
                        if idx == 0:
                            pos_new = x_rand + np.random.rand() * (x_rand - x_t) + beta * (x_rand - x_t)
                        else:
                            pos_new = x_rand + np.random.rand() * (x_t1 - x_t) + beta * (x_rand - x_t)
                    else:
                        conditions = np.random.uniform(0, 1, self.problem.n_dims) > 0.5
                        if idx == 0:
                            t1 = x_rand + np.random.rand(self.problem.n_dims) * (x_rand - x_t) + beta * (x_rand - x_t) + xichma * (self.problem.ub - x_t)
                            t2 = x_rand + np.random.rand(self.problem.n_dims) * (x_rand - x_t) + beta * (x_rand - x_t) + xichma * (x_t - self.problem.lb)
                        else:
                            t1 = x_rand + np.random.rand(self.problem.n_dims) * (x_t1 - x_t) + beta * (x_rand - x_t) + xichma * (self.problem.ub - x_t)
                            t2 = x_rand + np.random.rand(self.problem.n_dims) * (x_t1 - x_t) + beta * (x_rand - x_t) + xichma * (x_t - self.problem.lb)
                        pos_new = np.where(conditions, t1, t2)
                else:
                    if idx == 0:
                        pos_new = self.g_best[self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - x_t) + beta * (self.g_best[self.ID_POS] - x_t)
                    else:
                        pos_new = self.g_best[self.ID_POS] + np.random.rand() * (x_t1 - x_t) + beta * (self.g_best[self.ID_POS] - x_t)
            else:   # Chain foraging (Eq. 1,2)
                r = np.random.rand()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    pos_new = x_t + r * (self.g_best[self.ID_POS] - x_t) + alpha * (self.g_best[self.ID_POS] - x_t)
                else:
                    pos_new = x_t + r * (x_t1 - x_t) + alpha * (self.g_best[self.ID_POS] - x_t)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(self.pop, save=False)

        # Somersault foraging   (Eq. 8)
        pop_child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.somersault_range * \
                   (np.random.rand() * g_best[self.ID_POS] - np.random.rand() * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child)
        self.pop, g_best = self.update_global_best_solution(self.pop, save=False)

        # Quadratic Interpolation
        pop_new = []
        for idx in range(0, self.pop_size):
            idx2, idx3 = idx + 1, idx + 2
            if idx == self.pop_size-2:
                idx2, idx3 = idx + 1, 0
            if idx == self.pop_size-1:
                idx2, idx3 = 0, 1
            f1, f2, f3 = self.pop[idx][self.ID_TAR][self.ID_FIT], self.pop[idx2][self.ID_TAR][self.ID_FIT], self.pop[idx3][self.ID_TAR][self.ID_FIT]
            x1, x2, x3 = self.pop[idx][self.ID_POS], self.pop[idx2][self.ID_POS], self.pop[idx3][self.ID_POS]
            a = f1 / ((x1 - x2) * (x1 - x3)) + f2 / ((x2 - x1) * (x2 - x3)) + f3 / ((x3 - x1) * (x3 - x2))
            gx = ((x3 ** 2 - x2 ** 2) * f1 + (x1 ** 2 - x3 ** 2) * f2 + (x2 ** 2 - x1 ** 2) * f3) / (2 * ((x3 - x2) * f1 + (x1 - x3) * f2 + (x2 - x1) * f3))
            pos_new = np.where(a > 0, gx, x1)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(self.pop)
