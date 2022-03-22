# !/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseES(Optimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. http://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.ES import BaseES
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
    >>> lamda = 0.75
    >>> model = BaseES(problem_dict1, epoch, pop_size, lamda)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_STR = 2  # strategy

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.75, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.n_child = int(self.lamda * self.pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)
        self.nfe_per_epoch = self.n_child
        self.sort_flag = True

    def create_solution(self, lb=None, ub=None):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: solution with format [position, target, strategy]
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        strategy = np.random.uniform(0, self.distance)
        return [position, target, strategy]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem.n_dims) + tau * np.random.normal(0, 1.0, self.problem.n_dims))
            child.append([pos_new, None, strategy])
        child = self.update_target_wrapper_population(child)
        self.pop = self.get_sorted_strim_population(child + self.pop, self.pop_size)


class LevyES(BaseES):
    """
    My Levy-flight version of: Evolution Strategies (ES)

    Links:
        1. http://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Notes
    ~~~~~
    I implement Levy-flight and change the flow of original version.

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.ES import BaseES
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
    >>> lamda = 0.75
    >>> model = BaseES(problem_dict1, epoch, pop_size, lamda)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, lamda=0.75, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(problem, epoch, pop_size, lamda, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        self.nfe_per_epoch = 2 * self.n_child
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem.n_dims) + tau * np.random.normal(0, 1.0, self.problem.n_dims))
            child.append([pos_new, None, strategy])
        child = self.update_target_wrapper_population(child)

        child_levy = []
        for idx in range(0, self.n_child):
            levy = self.get_levy_flight_step(multiplier=0.01, case=-1)
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * \
                      levy * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** -1.0
            stdevs = np.array([np.exp(tau_p * np.random.normal(0, 1.0) + tau * np.random.normal(0, 1.0)) for _ in range(self.problem.n_dims)])
            child_levy.append([pos_new, None, stdevs])
        child_levy = self.update_target_wrapper_population(child_levy)
        self.pop = self.get_sorted_strim_population(child + child_levy + self.pop, self.pop_size)
