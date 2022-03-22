# !/usr/bin/env python
# Created by "Thieu" at 18:08, 19/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseCEM(Optimizer):
    """
    The original version of: Cross-Entropy Method (CEM)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms
        2. https://doi.org/10.1007/s10479-005-5724-z

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + n_best (int): N selected solutions as a samples for next evolution
        + alpha (float): weight factor for means and stdevs (normal distribution)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.probabilistic_based.CEM import BaseCEM
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
    >>> n_best = 30
    >>> alpha = 0.7
    >>> model = BaseCEM(problem_dict1, epoch, pop_size, n_best, alpha)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005. A tutorial on the
    cross-entropy method. Annals of operations research, 134(1), pp.19-67.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_best=30, alpha=0.7, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): N selected solutions as a samples for next evolution
            alpha (float): weight factor for means and stdevs (normal distribution)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size/2)])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.means = np.random.uniform(self.problem.lb, self.problem.ub)
        self.stdevs = np.abs(self.problem.ub - self.problem.lb)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Selected the best samples and update means and stdevs
        pop_best = self.pop[:self.n_best]
        pos_list = np.array([item[self.ID_POS] for item in pop_best])

        means_new = np.mean(pos_list, axis=0)
        means_new_repeat = np.repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
        stdevs_new = np.mean((pos_list - means_new_repeat) ** 2, axis=0)
        self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
        self.stdevs = np.abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

        ## Create new population for next generation
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = np.random.normal(self.means, self.stdevs)
            pop_new.append([self.amend_position(pos_new, self.problem.lb, self.problem.ub), None])
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
