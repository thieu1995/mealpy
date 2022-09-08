# !/usr/bin/env python
# Created by "Thieu" at 12:01, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalALO(Optimizer):
    """
    The original version of: Ant Lion Optimizer (ALO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/49920-ant-lion-optimizer-alo
        2. https://dx.doi.org/10.1016/j.advengsoft.2015.01.010

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ALO import OriginalALO
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
    >>> model = OriginalALO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. The ant lion optimizer. Advances in engineering software, 83, pp.80-98.
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

    def random_walk_antlion__(self, solution, current_epoch):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)

        # Decrease boundaries to converge towards antlion
        lb = self.problem.lb / I  # Equation (2.10) in the paper
        ub = self.problem.ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        if np.random.rand() < 0.5:
            lb = lb + solution  # Equation(2.8) in the paper
        else:
            lb = -lb + solution
        if np.random.rand() < 0.5:
            ub = ub + solution  # Equation(2.9) in the paper
        else:
            ub = -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        temp = []
        for k in range(0, self.problem.n_dims):
            X = np.cumsum(2 * (np.random.rand(self.epoch, 1) > 0.5) - 1)
            a = np.min(X)
            b = np.max(X)
            c = lb[k]  # [a b] - -->[c d]
            d = ub[k]
            X_norm = ((X - a) * (d - c)) / (b - a) + c  # Equation(2.7) in the paper
            temp.append(X_norm)
        return np.array(temp)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_fitness = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        # This for loop simulate random walks

        pop_new = []
        for idx in range(0, self.pop_size):
            # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
            rolette_index = self.get_index_roulette_wheel_selection(list_fitness)

            # RA is the random walk around the selected antlion by rolette wheel
            RA = self.random_walk_antlion__(self.pop[rolette_index][self.ID_POS], epoch)

            # RE is the random walk around the elite (the best antlion so far)
            RE = self.random_walk_antlion__(self.g_best[self.ID_POS], epoch)

            temp = (RA[:, epoch] + RE[:, epoch]) / 2  # Equation(2.13) in the paper

            # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update antlion positions and fitnesses based on the ants (if an ant becomes fitter than an antlion
        # we assume it was caught by the antlion and the antlion update goes to its position to build the trap)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)

        # Keep the elite in the population
        self.pop[-1] = deepcopy(self.g_best)


class BaseALO(OriginalALO):
    """
    The developed version: Ant Lion Optimizer (ALO)

    Notes
    ~~~~~
    + Improved performance by using matrix multiplication
    + The flow of updating a new position is updated

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ALO import BaseALO
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
    >>> model = BaseALO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def random_walk_antlion__(self, solution, current_epoch):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)

        # Decrease boundaries to converge towards antlion
        lb = self.problem.lb / I  # Equation (2.10) in the paper
        ub = self.problem.ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]. Eq 2.8, 2.9
        lb = lb + solution if np.random.rand() < 0.5 else -lb + solution
        ub = ub + solution if np.random.rand() < 0.5 else -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        ## Using matrix and vector for better performance
        X = np.array([np.cumsum(2 * (np.random.rand(self.epoch, 1) > 0.5) - 1) for _ in range(0, self.problem.n_dims)])
        a = np.min(X, axis=1)
        b = np.max(X, axis=1)
        temp1 = np.reshape((ub - lb) / (b - a), (self.problem.n_dims, 1))
        temp0 = X - np.reshape(a, (self.problem.n_dims, 1))
        X_norm = temp0 * temp1 + np.reshape(lb, (self.problem.n_dims, 1))
        return X_norm
