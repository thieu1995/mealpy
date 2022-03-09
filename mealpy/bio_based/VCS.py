#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseVCS(Optimizer):
    """
    My changed version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes
    ~~~~~
    + Removes all third loop, makes algrithm 10 times faster than original
    + In Immune response process, updates the whole position instead of updating each variable in position
    + Drops batch-size idea to 3 main process of this algorithm, makes it more robust

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + lamda (float): [0.2, 0.5], Number of the best will keep
        + xichma (float): [0.1, 0.5], Weight factor

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
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> lamda = 0.5
    >>> xichma = 0.3
    >>> model = BaseVCS(problem_dict1, epoch, pop_size, lamda, xichma)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            xichma (float): Weight factor, default = 0.3
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.xichma = xichma
        self.lamda = lamda
        if lamda < 1:
            self.n_best = int(lamda * self.pop_size)
        else:
            self.n_best = int(lamda)

    def _calculate_xmean(self, pop):
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
        for i in range(0, self.pop_size):
            xichma = (np.log1p(epoch + 1) / self.epoch) * (self.pop[i][self.ID_POS] - self.g_best[self.ID_POS])
            gauss = np.random.normal(np.random.normal(self.g_best[self.ID_POS], np.abs(xichma)))
            pos_new = gauss + np.random.uniform() * self.g_best[self.ID_POS] - np.random.uniform() * self.pop[i][self.ID_POS]
            self.pop[i][self.ID_POS] = self.amend_position(pos_new)
        self.pop = self.update_fitness_population(self.pop)

        ## Host cells infection
        x_mean = self._calculate_xmean(self.pop)
        xichma = self.xichma * (1 - (epoch + 1) / self.epoch)
        for i in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + xichma * np.random.normal(0, 1, self.problem.n_dims)
            self.pop[i][self.ID_POS] = self.amend_position(pos_new)
        self.pop = self.update_fitness_population(self.pop)

        ## Calculate the weighted mean of the λ best individuals by
        self.pop, g_best = self.get_global_best_solution(self.pop)

        ## Immune response
        for i in range(0, self.pop_size):
            pr = (self.problem.n_dims - i + 1) / self.problem.n_dims
            id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
            temp = self.pop[id1][self.ID_POS] - (self.pop[id2][self.ID_POS] - self.pop[i][self.ID_POS]) * np.random.uniform()
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < pr, self.pop[i][self.ID_POS], temp)
            self.pop[i][self.ID_POS] = self.amend_position(pos_new)
        self.pop = self.update_fitness_population(self.pop)


class OriginalVCS(BaseVCS):
    """
    The original version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes
    ~~~~~
    This is basic version, not the full version of the paper

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + lamda (float): [0.2, 0.5], Number of the best will keep
        + xichma (float): [0.1, 0.5], Weight factor

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
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> lamda = 0.5
    >>> xichma = 0.3
    >>> model = OriginalVCS(problem_dict1, epoch, pop_size, lamda, xichma)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm
    for optimization: Virus colony search. Advances in Engineering Software, 92, pp.65-88.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            xichma (float): Weight factor, default = 0.3
        """
        super().__init__(problem, epoch, pop_size, lamda, xichma, **kwargs)

    def amend_position(self, position):
        """
        If solution out of bound at dimension x, then it will re-arrange to random location in the range of domain

        Args:
            position: vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.where(np.logical_and(self.problem.lb <= position, position <= self.problem.ub),
                        position, np.random.uniform(self.problem.lb, self.problem.ub))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = deepcopy(self.pop)
        ## Viruses diffusion
        for i in range(0, self.pop_size):
            xichma = (np.log1p(epoch + 1) / self.epoch) * (pop[i][self.ID_POS] - self.g_best[self.ID_POS])
            gauss = np.array([np.random.normal(self.g_best[self.ID_POS][idx], np.abs(xichma[idx])) for idx in range(0, self.problem.n_dims)])
            pos_new = gauss + np.random.uniform() * self.g_best[self.ID_POS] - np.random.uniform() * pop[i][self.ID_POS]
            pop[i][self.ID_POS] = self.amend_position(pos_new)
        pop = self.update_fitness_population(pop)

        ## Host cells infection
        x_mean = self._calculate_xmean(pop)
        xichma = self.xichma * (1 - (epoch + 1) / self.epoch)
        for i in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + xichma * np.random.normal(0, 1, self.problem.n_dims)
            pop[i][self.ID_POS] = self.amend_position(pos_new)
        pop = self.update_fitness_population(pop)

        ## Immune response
        for i in range(0, self.pop_size):
            pr = (self.problem.n_dims - i + 1) / self.problem.n_dims
            pos_new = pop[i][self.ID_POS]
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() > pr:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    pos_new[j] = pop[id1][self.ID_POS][j] - (pop[id2][self.ID_POS][j] - pop[i][self.ID_POS][j]) * np.random.uniform()
            pop[i][self.ID_POS] = self.amend_position(pos_new)
        pop = self.update_fitness_population(pop)

        ## Greedy selection
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop[idx])
