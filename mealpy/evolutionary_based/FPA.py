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

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + p_s (float): [0.5, 0.95], switch probability, default = 0.8

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
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> p_s = 0.8
    >>> model = BaseFPA(problem_dict1, epoch, pop_size, p_s)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
    conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_s=0.8, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.p_s = p_s

    def amend_position(self, position=None):
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
        pop = []
        for idx in range(0, self.pop_size):
            if np.random.uniform() < self.p_s:
                levy = self.get_levy_flight_step(multiplier=0.001, case=-1)
                pos_new = self.pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                          levy * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            else:
                id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.pop[id1][self.ID_POS] - self.pop[id2][self.ID_POS])
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])
        self.pop = self.update_fitness_population(pop)
