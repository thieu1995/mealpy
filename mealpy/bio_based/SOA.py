#!/usr/bin/env python
# Created by "Thieu" at 17:21, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevSOA(Optimizer):
    """
    The developed version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Notes:
        1. The original one will not work because their operators always make the solution out of bound.
        2. I added the normal random number in Eq. 14 to make its work
        3. Besides, I will check keep the better one and remove the worst

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc: [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SOA import DevSOA
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
    >>> fc = 2
    >>> model = DevSOA(epoch, pop_size, fc)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, fc=2,  **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.fc = self.validator.check_float("fc", fc, [1.0, 10.])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        A = self.fc - (epoch+1)*self.fc / self.epoch    # Eq. 6
        uu = vv = 1

        pop_new = []
        for idx in range(0, self.pop_size):

            B = 2 * A**2 * np.random.random()                                   # Eq. 8
            M = B * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])     # Eq. 7
            C = A * self.pop[idx][self.ID_POS]                                  # Eq. 5
            D = np.abs(C + M)                                                   # Eq. 9

            k = np.random.uniform(0, 2*np.pi)
            r = uu * np.exp(k*vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k

            x_new = xx * yy * zz * D + np.random.normal(0, 1) * self.g_best[self.ID_POS]                 # Eq. 14
            x_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([x_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(x_new)
                self.pop[idx] = self.get_better_solution([x_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalSOA(Optimizer):
    """
    The original version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc: [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SOA import OriginalSOA
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
    >>> model = OriginalSOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory and its applications
    for large-scale industrial engineering problems. Knowledge-based systems, 165, 169-196.
    """

    def __init__(self, epoch=10000, pop_size=100, fc=2,  **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.fc = self.validator.check_float("fc", fc, [1.0, 10.])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        A = self.fc - (epoch+1)*self.fc / self.epoch    # Eq. 6
        uu = vv = 1

        pop_new = []
        for idx in range(0, self.pop_size):
            B = 2 * A**2 * np.random.random()                                   # Eq. 8
            M = B * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])     # Eq. 7
            C = A * self.pop[idx][self.ID_POS]                                  # Eq. 5
            D = np.abs(C + M)                                                   # Eq. 9

            k = np.random.uniform(0, 2*np.pi)
            r = uu * np.exp(k*vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k

            x_new = xx * yy * zz * D + self.g_best[self.ID_POS]                 # Eq. 14
            x_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([x_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(x_new)
        self.pop = self.update_target_wrapper_population(pop_new)
