#!/usr/bin/env python
# Created by "Thieu" at 18:37, 28/05/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCSA(Optimizer):
    """
    The original version of: Cuckoo Search Algorithm (CSA)

    Links:
        1. https://doi.org/10.1109/NABIC.2009.5393690

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_a (float): [0.1, 0.7], probability a, default=0.3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.CSA import OriginalCSA
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
    >>> p_a = 0.3
    >>> model = OriginalCSA(epoch, pop_size, p_a)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S. and Deb, S., 2009, December. Cuckoo search via Lévy flights. In 2009 World
    congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). Ieee.
    """

    def __init__(self, epoch=10000, pop_size=100, p_a=0.3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_a (float): probability a, default=0.3
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p_a = self.validator.check_float("p_a", p_a, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_a"])
        self.n_cut = int(self.p_a * self.pop_size)
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for i in range(0, self.pop_size):
            ## Generate levy-flight solution
            levy_step = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = self.pop[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                      levy_step * (self.pop[i][self.ID_POS] - self.g_best[self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop[i]):
                pop_new[i] = [pos_new, target]

        ## Abandoned some worst nests
        pop = self.get_sorted_strim_population(pop_new, self.pop_size)
        pop_new = []
        for i in range(0, self.n_cut):
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop[:(self.pop_size - self.n_cut)] + pop_new
