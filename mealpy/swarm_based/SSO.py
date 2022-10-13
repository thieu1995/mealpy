#!/usr/bin/env python
# Created by "Thieu" at 11:38, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSSO(Optimizer):
    """
    The original version of: Salp Swarm Optimization (SSO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.07.002

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SSO import OriginalSSO
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
    >>> model = OriginalSSO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. and Mirjalili, S.M., 2017.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in
    Engineering Software, 114, pp.163-191.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * (epoch + 1) / self.epoch) ** 2))
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.pop_size / 2:
                c2_list = np.random.random(self.problem.n_dims)
                c3_list = np.random.random(self.problem.n_dims)
                pos_new_1 = self.g_best[self.ID_POS] + c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new_2 = self.g_best[self.ID_POS] - c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
            else:
                # Eq. (3.4) in the paper
                pos_new = (self.pop[idx][self.ID_POS] + self.pop[idx - 1][self.ID_POS]) / 2

            # Check if salps go out of the search space and bring it back then re-calculate its fitness value
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
