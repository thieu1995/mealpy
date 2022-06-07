# !/usr/bin/env python
# Created by "Thieu" at 19:34, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseFPA(Optimizer):
    """
    The original version of: Flower Pollination Algorithm (FPA)

    Links:
        1. https://doi.org/10.1007/978-3-642-32894-7_27

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_s (float): [0.5, 0.95], switch probability, default = 0.8
        + levy_multiplier: [0.0001, 1000], mutiplier factor of Levy-flight trajectory, depends on the problem

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.FPA import BaseFPA
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
    >>> p_s = 0.8
    >>> levy_multiplier = 0.2
    >>> model = BaseFPA(problem_dict1, epoch, pop_size, p_s, levy_multiplier)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
    conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_s=0.8, levy_multiplier=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
            levy_multiplier (float): mutiplier factor of Levy-flight trajectory, default = 0.2
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p_s = self.validator.check_float("p_s", p_s, (0, 1.0))
        self.levy_multiplier = self.validator.check_float("levy_multiplier", levy_multiplier, (0, 1000))
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        condition = np.logical_and(lb <= position, position <= ub)
        random_pos = np.random.uniform(lb, ub)
        return np.where(condition, position, random_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            if np.random.uniform() < self.p_s:
                levy = self.get_levy_flight_step(multiplier=self.levy_multiplier, size=self.problem.n_dims, case=-1)
                pos_new = self.pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * \
                          levy * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            else:
                id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.pop[id1][self.ID_POS] - self.pop[id2][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop)
