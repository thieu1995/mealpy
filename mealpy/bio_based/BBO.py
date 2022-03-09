# !/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalBBO(Optimizer):
    """
    The original version of: Biogeography-Based Optimization (BBO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + p_m: [0.01, 0.2], Mutation probability
        + elites: [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.BBO import OriginalBBO
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
    >>> p_m = 0.01
    >>> elites = 2
    >>> model = OriginalBBO(problem_dict1, epoch, pop_size, p_m, elites)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Simon, D., 2008. Biogeography-based optimization. IEEE transactions on evolutionary computation, 12(6), pp.702-713.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_m=0.01, elites=2, **kwargs):
        """
        Initialize the algorithm components.

        Args:
            problem (dict): The problem dictionary
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size, default = 100
            p_m (float): Mutation probability, default=0.01
            elites (int): Number of elites will be keep for next generation, default=2
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.p_m = p_m
        self.elites = elites

        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        self.mr = 1 - self.mu

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, pop_elites, _ = self.get_special_solutions(self.pop, best=self.elites)
        pop = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(self.problem.n_dims):
                if np.random.uniform() < self.mr[idx]:  # Should we immigrate?
                    # Pick a position from which to emigrate (roulette wheel selection)
                    random_number = np.random.uniform() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (select_index < self.pop_size - 1):
                        select_index += 1
                        select += self.mu[select_index]
                    # this is the migration step
                    pos_new[j] = self.pop[select_index][self.ID_POS][j]

            noise = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, noise, pos_new)
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])

        pop = self.update_fitness_population(pop)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.pop = self.get_sorted_strim_population(pop + pop_elites, self.pop_size)


class BaseBBO(OriginalBBO):
    """
    My changed version of: Biogeography-Based Optimization (BBO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + p_m: [0.01, 0.2], Mutation probability
        + elites: [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.BBO import BaseBBO
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
    >>> p_m = 0.01
    >>> elites = 2
    >>> model = BaseBBO(problem_dict1, epoch, pop_size, p_m, elites)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_m=0.01, elites=2, **kwargs):
        """
        Initialize the algorithm components.

        Args:
            problem (dict): The problem dictionary
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size, default = 100
            p_m (float): Mutation probability, default=0.01
            elites (int): Number of elites will be keep for next generation, default=2
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, p_m, elites, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, pop_elites, _ = self.get_special_solutions(self.pop, best=self.elites)
        list_fitness = [agent[self.ID_TAR][self.ID_FIT] for agent in self.pop]
        pop = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            # Pick a position from which to emigrate (roulette wheel selection)
            idx_selected = self.get_index_roulette_wheel_selection(list_fitness)
            # this is the migration step
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.mr[idx], self.pop[idx_selected][self.ID_POS], self.pop[idx][self.ID_POS])
            # Mutation
            temp = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, temp, pos_new)
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)
        # Replace the solutions with their new migrated and mutated versions then merge populations
        self.pop = self.get_sorted_strim_population(pop + pop_elites, self.pop_size)
