#!/usr/bin/env python
# Created by "Thieu" at 17:38, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCircleSA(Optimizer):
    """
    The original version of: Circle Search Algorithm (CircleSA)

    Links:
        1. https://doi.org/10.3390/math10101626
        2. https://www.mdpi.com/2227-7390/10/10/1626

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.CircleSA import OriginalCircleSA
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
    >>> model = OriginalCircleSA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Qais, M. H., Hasanien, H. M., Turky, R. A., Alghuwainem, S., Tostado-Véliz, M., & Jurado, F. (2022).
    Circle Search Algorithm: A Geometry-Based Metaheuristic Optimization Algorithm. Mathematics, 10(10), 1626.
    """

    def __init__(self, epoch=10000, pop_size=100, c_factor=0.8, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c_factor = self.validator.check_float("c_factor", c_factor, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c_factor"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = np.pi - np.pi * ((epoch+1)/self.epoch)**2       # Eq. 8
        p = 1 - 0.9 * ((epoch + 1) / self.epoch) ** 0.5
        threshold = self.c_factor * self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            w = a * np.random.rand() - a
            if (epoch+1) > threshold:
                x_new = self.g_best[self.ID_POS] + (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * np.tan(w * np.random.rand())
            else:
                x_new = self.g_best[self.ID_POS] - (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * np.tan(w * p)
            x_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([x_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(x_new)
        self.pop = self.update_target_wrapper_population(pop_new)
