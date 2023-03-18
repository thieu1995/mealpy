#!/usr/bin/env python
# Created by "Thieu" at 17:41, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWarSO(Optimizer):
    """
    The original version of: War Strategy Optimization (WarSO) algorithm

    Links:
       1. https://www.researchgate.net/publication/358806739_War_Strategy_Optimization_Algorithm_A_New_Effective_Metaheuristic_Algorithm_for_Global_Optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + rr (float): [0.1, 0.9], the probability of switching position updating, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.WarSO import OriginalWarSO
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
    >>> model = OriginalWarSO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
    war strategy optimization algorithm." International Journal of Energy Research (2022).
    """

    def __init__(self, epoch=10000, pop_size=100, rr=0.1, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            rr (float): the probability of switching position updating, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.rr = self.validator.check_float("rr", rr, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size"])
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.wl = 2 * np.ones(self.pop_size)
        self.wg = np.zeros(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_sorted, _ = self.get_global_best_solution(self.pop)
        com = np.random.permutation(self.pop_size)
        for idx in range(0, self.pop_size):
            r1 = np.random.rand()
            if r1 < self.rr:
                pos_new = 2*r1*(self.g_best[self.ID_POS] - self.pop[com[idx]][self.ID_POS]) + \
                          self.wl[idx]*np.random.rand()*(pop_sorted[idx][self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                pos_new = 2*r1*(pop_sorted[idx][self.ID_POS] - self.g_best[self.ID_POS]) + \
                          np.random.rand() * (self.wl[idx] * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
                self.wg[idx] += 1
                self.wl[idx] = 1 * self.wl[idx] * (1 - self.wg[idx] / self.epoch)**2
