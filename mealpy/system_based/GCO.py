# !/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseGCO(Optimizer):
    """
    My changed version of: Germinal Center Optimization (GCO)

    Notes
    ~~~~~
    + The global best solution and 2 random solutions are used instead of randomizing 3 solutions

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.GCO import BaseGCO
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
    >>> cr = 0.7
    >>> wf = 1.25
    >>> model = BaseGCO(problem_dict1, epoch, pop_size, cr, wf)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, cr=0.7, wf=1.25, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.wf = self.validator.check_float("wf", wf, (0, 3.0))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        ## Dynamic variables
        self.dyn_list_cell_counter = np.ones(self.pop_size)  # CEll Counter
        self.dyn_list_life_signal = 70 * np.ones(self.pop_size)  # 70% to duplicate, and 30% to die  # LIfe-Signal

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Dark-zone process    (can be parallelization)
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1

            # Mutate process
            r1, r2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = self.g_best[self.ID_POS] + self.wf * (self.pop[r2][self.ID_POS] - self.pop[r1][self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.dyn_list_cell_counter[idx] += 10
                self.pop[idx] = deepcopy(pop_new[idx])

        ## Light-zone process   (no needs parallelization)
        for i in range(0, self.pop_size):
            self.dyn_list_cell_counter[i] = 10
            fit_list = np.array([item[self.ID_TAR][self.ID_FIT] for item in pop_new])
            fit_max = max(fit_list)
            fit_min = min(fit_list)
            self.dyn_list_cell_counter[i] += 10 * (self.pop[i][self.ID_TAR][self.ID_FIT] - fit_max) / (fit_min - fit_max + self.EPSILON)


class OriginalGCO(BaseGCO):
    """
    The original version of: Germinal Center Optimization (GCO)

    Links:
        1. https://doi.org/10.2991/ijcis.2018.25905179
        2. https://www.atlantis-press.com/journals/ijcis/25905179/view

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.GCO import OriginalGCO
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
    >>> cr = 0.7
    >>> wf = 1.25
    >>> model = OriginalGCO(problem_dict1, epoch, pop_size, cr, wf)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Villaseñor, C., Arana-Daniel, N., Alanis, A.Y., López-Franco, C. and Hernandez-Vargas, E.A., 2018.
    Germinal center optimization algorithm. International Journal of Computational Intelligence Systems, 12(1), p.13.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, cr=0.7, wf=1.25, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(problem, epoch, pop_size, cr, wf, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Dark-zone process    (can be parallelization)
        for idx in range(0, self.pop_size):
            if np.random.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            elif self.dyn_list_cell_counter[idx] > 1:
                self.dyn_list_cell_counter[idx] -= 1

            # Mutate process
            p = self.dyn_list_cell_counter / np.sum(self.dyn_list_cell_counter)
            r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size))), 3, replace=False, p=p)
            pos_new = self.pop[r1][self.ID_POS] + self.wf * (self.pop[r2][self.ID_POS] - self.pop[r3][self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, pos_new,
                               self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new)
            # for each pos_new, generate the fitness_population
            pop_new = self.update_fitness_population([[pos_new, None]])
            if self.compare_agent(pop_new[0], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[0])
                self.dyn_list_life_signal[idx] += 10
                if self.compare_agent(pop_new[0], self.g_best):
                    self.g_best = deepcopy(pop_new[0])

        ## Light-zone process   (no needs parallelization)
        self.dyn_list_life_signal -= 10
        fit_list = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        fit_max = max(fit_list)
        fit_min = min(fit_list)
        fit = (fit_list - fit_max) / (fit_min - fit_max)
        if self.problem.minmax != 'min':
            fit = 1 - fit
        self.dyn_list_life_signal += 10 * fit
