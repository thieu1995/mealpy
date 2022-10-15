#!/usr/bin/env python
# Created by "Thieu" at 14:20, 15/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSOS(Optimizer):
    """
    The original version: Symbiotic Organisms Search (SOS)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2014.03.007

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SOS import OriginalSOS
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
    >>> model = OriginalSOS(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Cheng, M. Y., & Prayogo, D. (2014). Symbiotic organisms search: a new metaheuristic
    optimization algorithm. Computers & Structures, 139, 98-112.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.support_parallel_modes = False
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        self.nfe_per_epoch = 4 * self.pop_size
        for idx in range(0, self.pop_size):
            ## Mutualism Phase
            jdx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            mutual_vector = (self.pop[idx][self.ID_POS] + self.pop[jdx][self.ID_POS]) / 2
            bf1, bf2 = np.random.randint(1, 3, 2)
            xi_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - bf1 * mutual_vector)
            xj_new = self.pop[jdx][self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - bf2 * mutual_vector)
            xi_new = self.amend_position(xi_new, self.problem.lb, self.problem.ub)
            xj_new = self.amend_position(xj_new, self.problem.lb, self.problem.ub)
            xi_tar = self.get_target_wrapper(xi_new)
            xj_tar = self.get_target_wrapper(xj_new)
            if self.compare_agent([xi_new, xi_tar], self.pop[idx]):
                self.pop[idx] = [xi_new, xi_tar]
            if self.compare_agent([xj_new, xj_tar], self.pop[jdx]):
                self.pop[jdx] = [xj_new, xj_tar]

            ## Commensalism phase
            jdx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            xi_new = self.pop[idx][self.ID_POS] + np.random.uniform(-1, 1) * (self.g_best[self.ID_POS] - self.pop[jdx][self.ID_POS])
            xi_new = self.amend_position(xi_new, self.problem.lb, self.problem.ub)
            xi_tar = self.get_target_wrapper(xi_new)
            if self.compare_agent([xi_new, xi_tar], self.pop[idx]):
                self.pop[idx] = [xi_new, xi_tar]

            ## Parasitism phase
            jdx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            temp_idx = np.random.randint(0, self.problem.n_dims)
            xi_new = self.pop[jdx][self.ID_POS].copy()
            xi_new[temp_idx] = self.generate_position(self.problem.lb, self.problem.ub)[temp_idx]
            xi_tar = self.get_target_wrapper(xi_new)
            if self.compare_agent([xi_new, xi_tar], self.pop[idx]):
                self.pop[idx] = [xi_new, xi_tar]
