#!/usr/bin/env python
# Created by "Thieu" at 08:57, 12/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalHCO(Optimizer):
    """
    The original version of: Human Conception Optimizer (HCO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/124200-human-conception-optimizer-hco
        2. https://www.nature.com/articles/s41598-022-25031-6

    Notes:
        1. Kinda similar to PSO algorithm. Just the concepts of nature-inspired animals are difference
        2. Matlab code different to the paper

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + w (float): (0, 1.) - weight factor for probability of fitness selection, default=0.65
        + w1 (float): (0, 1.0) - weight factor for velocity update stage, default=0.05
        + c1 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4
        + c2 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.HCO import OriginalHCO
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
    >>> model = OriginalHCO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Acharya, D., & Das, D. K. (2022). A novel Human Conception Optimizer for solving optimization problems. Scientific Reports, 12(1), 21631.
    """

    def __init__(self, epoch=10000, pop_size=100, w=0.65, w1=0.05, c1=1.4, c2=1.4, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            w (float): weight factor for probability of fitness selection, default=0.65
            w1 (float): weight factor for velocity update stage, default=0.05
            c1 (float): acceleration coefficient, same as PSO, default=1.4
            c2 (float): acceleration coefficient, same as PSO, default=1.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.w = self.validator.check_float("w", w, [0, 1.0])
        self.w1 = self.validator.check_float("w1", w1, [0, 1.0])
        self.c1 = self.validator.check_float("c1", c1, [0., 100.])
        self.c2 = self.validator.check_float("c2", c2, [1., 100.])
        self.set_parameters(["epoch", "pop_size", "w", "w1", "c1", "c2"])
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        pop_op = []
        for idx in range(0, self.pop_size):
            pos_new = self.problem.ub + self.problem.lb - self.pop[idx][self.ID_POS]
            pop_op.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_op = self.update_target_wrapper_population(pop_op)
            self.pop = self.greedy_selection_population(self.pop, pop_op)
        _, (best,), (worst,) = self.get_special_solutions(self.pop, best=1, worst=1)
        pfit = (worst[self.ID_TAR][self.ID_FIT] - best[self.ID_TAR][self.ID_FIT]) * self.w + best[self.ID_TAR][self.ID_FIT]
        for idx in range(0, self.pop_size):
            if self.compare_agent([None, [pfit, None]], self.pop[idx]):
                while True:
                    sol = self.create_solution(self.problem.lb, self.problem.ub)
                    if self.compare_agent(sol, [None, [pfit, None]]):
                        self.pop[idx] = sol
                        break
        self.vec = np.random.rand(self.pop_size, self.problem.n_dims)
        self.pop_p = deepcopy(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        lamda = np.random.rand()
        neu = 2
        fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        fit_mean = np.mean(fits)
        RR = (self.g_best[self.ID_TAR][self.ID_FIT] - fits) ** 2
        rr = (fit_mean - fits) ** 2
        ll = RR - rr
        LL = (self.g_best[self.ID_TAR][self.ID_FIT] - fit_mean)
        VV = lamda * (ll / (4 * neu * LL))
        pop_new = []
        for idx in range(0, self.pop_size):
            a1 = self.pop_p[idx][self.ID_POS] - self.pop[idx][self.ID_POS]
            a2 = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
            self.vec[idx] = self.w1 * (VV[idx] + self.vec[idx]) + self.c1 * a1*np.sin(2*np.pi*(epoch+1)/self.epoch) + self.c2*a2*np.sin(2*np.pi*(epoch+1)/self.epoch)
            pos_new = self.pop[idx][self.ID_POS] + self.vec[idx]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = pop_new[idx]
                if self.compare_agent(pop_new[idx], self.pop_p[idx]):
                    self.pop_p[idx] = deepcopy(pop_new[idx])
