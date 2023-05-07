#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseVCS(Optimizer):
    """
    The developed version: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes
    ~~~~~
    + In Immune response process, updates the whole position instead of updating each variable in position

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.VCS import BaseVCS
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
    >>> lamda = 0.5
    >>> sigma = 0.3
    >>> model = BaseVCS(epoch, pop_size, lamda, sigma)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, lamda=0.5, sigma=1.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Percentage of the number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.sigma = self.validator.check_float("sigma", sigma, (0, 5.0))
        self.n_best = int(self.lamda * self.pop_size)
        self.set_parameters(["epoch", "pop_size", "lamda", "sigma"])
        self.sort_flag = True

    def calculate_xmean__(self, pop):
        """
        Calculate the mean position of list of solutions (population)

        Args:
            pop (list): List of solutions (population)

        Returns:
            list: Mean position
        """
        ## Calculate the weighted mean of the λ best individuals by
        pop, local_best = self.get_global_best_solution(pop)
        pos_list = [agent[self.ID_POS] for agent in pop[:self.n_best]]
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(np.prod(range(1, self.n_best + 1)))
        weight = np.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        x_mean = weight * np.sum(pos_list, axis=0)
        return x_mean

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Viruses diffusion
        pop = []
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch + 1) / self.epoch) * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            gauss = np.random.normal(np.random.normal(self.g_best[self.ID_POS], np.abs(sigma)))
            pos_new = gauss + np.random.uniform() * self.g_best[self.ID_POS] - np.random.uniform() * self.pop[idx][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)

        ## Host cells infection
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - (epoch + 1) / self.epoch)
        pop = []
        for idx in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * np.random.normal(0, 1, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)

        ## Calculate the weighted mean of the λ best individuals by
        self.pop, g_best = self.get_global_best_solution(self.pop)

        ## Immune response
        pop = []
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            temp = self.pop[id1][self.ID_POS] - (self.pop[id2][self.ID_POS] - self.pop[idx][self.ID_POS]) * np.random.uniform()
            condition = np.random.random(self.problem.n_dims) < pr
            pos_new = np.where(condition, self.pop[idx][self.ID_POS], temp)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)


class OriginalVCS(BaseVCS):
    """
    The original version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.VCS import OriginalVCS
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
    >>> lamda = 0.5
    >>> sigma = 0.3
    >>> model = OriginalVCS(epoch, pop_size, lamda, sigma)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm
    for optimization: Virus colony search. Advances in Engineering Software, 92, pp.65-88.
    """

    def __init__(self, epoch=10000, pop_size=100, lamda=0.5, sigma=1.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(epoch, pop_size, lamda, sigma, **kwargs)

    def bounded_position(self, position=None, lb=None, ub=None):
        condition = np.logical_and(lb <= position, position <= ub)
        random_pos = np.random.uniform(lb, ub)
        return np.where(condition, position, random_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Viruses diffusion
        pop = []
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch + 1) / self.epoch) * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            gauss = np.array([np.random.normal(self.g_best[self.ID_POS][j], np.abs(sigma[j])) for j in range(0, self.problem.n_dims)])
            pos_new = gauss + np.random.uniform() * self.g_best[self.ID_POS] - np.random.uniform() * self.pop[idx][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)

        ## Host cells infection
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - (epoch + 1) / self.epoch)
        pop = []
        for idx in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * np.random.normal(0, 1, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)

        ## Immune response
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            pos_new = pop[idx][self.ID_POS]
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() > pr:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new[j] = pop[id1][self.ID_POS][j] - (pop[id2][self.ID_POS][j] - pop[idx][self.ID_POS][j]) * np.random.uniform()
            pop[idx][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(pop, self.pop)
