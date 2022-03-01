# !/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseABC(Optimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Notes
    ~~~~~
    + This version is based on ABC in the book Clever Algorithms
    + Improved the function _search_neigh__

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + couple_bees (list): n bees for (good locations, other locations) -> ([10, 20], [3, 8])
        + patch_variables (list): (patch_var, patch_reduce_factor) -> ([3, 6], [0.85, 0,.99])
        + patch_variables = patch_variables * patch_factor (0.985)
        + sites (list): 3 bees (employed bees, onlookers and scouts), 1 good partition -> (3, 1), fixed parameter

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ABC import BaseABC
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
    >>> couple_bees = [16, 4]
    >>> patch_variables = [5, 0.98]
    >>> sites = [3, 1]
    >>> model = BaseABC(problem_dict1, epoch, pop_size, couple_bees, patch_variables, sites)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Karaboga, D. and Basturk, B., 2008. On the performance of artificial bee colony (ABC)
    algorithm. Applied soft computing, 8(1), pp.687-697.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, couple_bees=(16, 4),
                 patch_variables=(5.0, 0.985), sites=(3, 1), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            couple_bees (list): number of bees which provided for good location and other location
            patch_variables (list): patch_variables = patch_variables * patch_factor (0.985)
            sites (list): 3 bees (employed bees, onlookers and scouts), 1 good partition
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.e_bees = couple_bees[0]
        self.o_bees = couple_bees[1]
        self.patch_size = patch_variables[0]
        self.patch_factor = patch_variables[1]
        self.num_sites = sites[0]
        self.elite_sites = sites[1]

        self.nfe_per_epoch = self.e_bees * self.elite_sites + self.o_bees * self.num_sites + (self.pop_size - self.num_sites)
        self.sort_flag = True

    def _search_neigh__(self, parent=None, neigh_size=None):
        """
        Search 1 best position in neigh_size position
        """
        pop_neigh = []
        for idx in range(0, neigh_size):
            t1 = np.random.randint(0, len(parent[self.ID_POS]) - 1)
            new_bee = deepcopy(parent[self.ID_POS])
            new_bee[t1] = (parent[self.ID_POS][t1] + np.random.uniform() * self.patch_size) if np.random.uniform() < 0.5 \
                else (parent[self.ID_POS][t1] - np.random.uniform() * self.patch_size)
            new_bee[t1] = np.maximum(self.problem.lb[t1], np.minimum(self.problem.ub[t1], new_bee[t1]))
            pop_neigh.append([new_bee, None])
        pop_neigh = self.update_fitness_population(pop_neigh)
        _, current_best = self.get_global_best_solution(pop_neigh)
        return current_best

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.num_sites:
                if idx < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                agent = self._search_neigh__(self.pop[idx], neigh_size)
            else:
                agent = self.create_solution()
            pop_new.append(agent)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
