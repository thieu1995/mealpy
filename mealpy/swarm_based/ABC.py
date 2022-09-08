# !/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalABC(Optimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Notes
    ~~~~~
    + This version is based on ABC in the book Clever Algorithms
    + Improved the function search_neighborhood__

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_elites (int): number of employed bees which provided for good location
        + n_others (int): number of employed bees which provided for other location
        + patch_size (float): patch_variables = patch_variables * patch_reduction
        + patch_reduction (float): the reduction factor
        + n_sites (int): 3 bees (employed bees, onlookers and scouts),
        + n_elite_sites (int): 1 good partition

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ABC import OriginalABC
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
    >>> n_elites = 16
    >>> n_others = 4
    >>> patch_size = 5.0
    >>> patch_reduction = 0.985
    >>> n_sites = 3
    >>> n_elite_sites = 1
    >>> model = OriginalABC(epoch, pop_size, n_elites, n_others, patch_size, patch_reduction, n_sites, n_elite_sites)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Karaboga, D. and Basturk, B., 2008. On the performance of artificial bee colony (ABC)
    algorithm. Applied soft computing, 8(1), pp.687-697.
    """

    def __init__(self, epoch=10000, pop_size=100, n_elites=16, n_others=4, patch_size=5.0, patch_reduction=0.985, n_sites=3, n_elite_sites=1, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_elites (int): number of employed bees which provided for good location
            n_others (int): number of employed bees which provided for other location
            patch_size (float): patch_variables = patch_variables * patch_reduction
            patch_reduction (float): the reduction factor
            n_sites (int): 3 bees (employed bees, onlookers and scouts),
            n_elite_sites (int): 1 good partition
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_elites = self.validator.check_int("n_elites", n_elites, [4, 20])
        self.n_others = self.validator.check_int("n_others", n_others, [2, 5])
        self.patch_size = self.validator.check_float("patch_size", patch_size, [2, 10])
        self.patch_reduction = self.validator.check_float("patch_reduction", patch_reduction, (0, 1.0))
        self.n_sites = self.validator.check_int("n_sites", n_sites, [2, 5])
        self.n_elite_sites = self.validator.check_int("n_elite_sites", n_elite_sites, [1, 3])
        self.set_parameters(["epoch", "pop_size", "n_elites", "n_others", "patch_size", "patch_reduction", "n_sites", "n_elite_sites"])

        self.nfe_per_epoch = self.n_elites * self.n_elite_sites + self.n_others * self.n_sites + (self.pop_size - self.n_sites)
        self.sort_flag = True

    def search_neighborhood__(self, parent=None, neigh_size=None):
        """
        Search 1 best position in neigh_size position
        """
        pop_neigh = []
        for idx in range(0, neigh_size):
            t1 = np.random.randint(0, len(parent[self.ID_POS]) - 1)
            new_bee = deepcopy(parent[self.ID_POS])
            new_bee[t1] = (parent[self.ID_POS][t1] + np.random.uniform() * self.patch_size) if np.random.uniform() < 0.5 \
                else (parent[self.ID_POS][t1] - np.random.uniform() * self.patch_size)
            pos_new = self.amend_position(new_bee, self.problem.lb, self.problem.ub)
            pop_neigh.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_neigh[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_neigh = self.update_target_wrapper_population(pop_neigh)
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
            if idx < self.n_sites:
                if idx < self.n_elite_sites:
                    neigh_size = self.n_elites
                else:
                    neigh_size = self.n_others
                agent = self.search_neighborhood__(self.pop[idx], neigh_size)
            else:
                agent = self.create_solution(self.problem.lb, self.problem.ub)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self.get_better_solution(agent, self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.greedy_selection_population(self.pop, pop_new)
