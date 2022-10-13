#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBES(Optimizer):
    """
    The original version of: Bald Eagle Search (BES)

    Links:
        1. https://doi.org/10.1007/s10462-019-09732-5

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
        + R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
        + alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
        + c1 (float): default: 2, in [1, 2]
        + c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BES import OriginalBES
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
    >>> a_factor = 10
    >>> R_factor = 1.5
    >>> alpha = 2.0
    >>> c1 = 2.0
    >>> c2 = 2.0
    >>> model = OriginalBES(epoch, pop_size, a_factor, R_factor, alpha, c1, c2)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Alsattar, H.A., Zaidan, A.A. and Zaidan, B.B., 2020. Novel meta-heuristic bald eagle
    search optimisation algorithm. Artificial Intelligence Review, 53(3), pp.2237-2264.
    """

    def __init__(self, epoch=10000, pop_size=100, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
            R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
            alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
            c1 (float): default: 2, in [1, 2]
            c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.a_factor = self.validator.check_int("a_factor", a_factor, [2, 20])
        self.R_factor = self.validator.check_float("R_factor", R_factor, [0.1, 3.0])
        self.alpha = self.validator.check_float("alpha", alpha, [0.5, 3.0])
        self.c1 = self.validator.check_float("c1", c1, (0, 4.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 4.0))
        self.set_parameters(["epoch", "pop_size", "a_factor", "R_factor", "alpha", "c1", "c2"])
        self.nfe_per_epoch = 3 * self.pop_size
        self.sort_flag = False

    def create_x_y_x1_y1__(self):
        """ Using numpy vector for faster computational time """
        ## Eq. 2
        phi = self.a_factor * np.pi * np.random.uniform(0, 1, self.pop_size)
        r = phi + self.R_factor * np.random.uniform(0, 1, self.pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)

        ## Eq. 3
        r1 = phi1 = self.a_factor * np.pi * np.random.uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)

        x_list = xr / np.max(xr)
        y_list = yr / np.max(yr)
        x1_list = xr1 / np.max(xr1)
        y1_list = yr1 / np.max(yr1)
        return x_list, y_list, x1_list, y1_list

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## 0. Pre-definded
        x_list, y_list, x1_list, y1_list = self.create_x_y_x1_y1__()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([individual[self.ID_POS] for individual in self.pop])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.g_best[self.ID_POS] + self.alpha * np.random.uniform() * (pos_mean - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## 2. Search in space
        pos_list = np.array([individual[self.ID_POS] for individual in self.pop])
        pos_mean = np.mean(pos_list, axis=0)

        pop_child = []
        for idx in range(0, self.pop_size):
            idx_rand = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos_new = self.pop[idx][self.ID_POS] + y_list[idx] * (self.pop[idx][self.ID_POS] - self.pop[idx_rand][self.ID_POS]) + \
                      x_list[idx] * (self.pop[idx][self.ID_POS] - pos_mean)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child)

        ## 3. Swoop
        pos_list = np.array([individual[self.ID_POS] for individual in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.uniform() * self.g_best[self.ID_POS] + x1_list[idx] * (self.pop[idx][self.ID_POS] - self.c1 * pos_mean) \
                      + y1_list[idx] * (self.pop[idx][self.ID_POS] - self.c2 * self.g_best[self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
