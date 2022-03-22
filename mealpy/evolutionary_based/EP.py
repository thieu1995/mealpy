# !/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseEP(Optimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. http://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.EP import BaseEP
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
    >>> bout_size = 0.05
    >>> model = BaseEP(problem_dict1, epoch, pop_size, bout_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, problem, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            n_child (float): percentage of child agents implement tournament selection
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.bout_size = self.validator.check_float("bout_size", bout_size, (0, 1.0))
        self.n_bout_size = int(self.bout_size * pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def create_solution(self, lb=None, ub=None):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, target, strategy, times_win]
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        strategy = np.random.uniform(0, self.distance, len(lb))
        times_win = 0
        return [position, target, strategy, times_win]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
        child = self.update_target_wrapper_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1
        pop = sorted(pop, key=lambda item: item[self.ID_WIN], reverse=True)
        self.pop = pop[:self.pop_size]


class LevyEP(BaseEP):
    """
    My Levy-flight version of: Evolutionary Programming (LevyEP)

    Notes
    ~~~~~
    I try to apply Levy-flight to EP, change flow and add some equations.

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.EP import LevyEP
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
    >>> bout_size = 0.05
    >>> model = LevyEP(problem_dict1, epoch, pop_size, bout_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_POS = 0
    ID_TAR = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, problem, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(problem, epoch, pop_size, bout_size, **kwargs)
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
        child = self.update_target_wrapper_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1

        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        pop = sorted(pop, key=lambda agent: agent[self.ID_WIN], reverse=True)
        pop = deepcopy(pop[:self.pop_size])
        pop_left = deepcopy(pop[self.pop_size:])

        ## Choice random 50% of population left
        pop_comeback = []
        idx_list = np.random.choice(range(0, len(pop_left)), int(0.5 * len(pop_left)), replace=False)
        for idx in idx_list:
            levy = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = pop_left[idx][self.ID_POS] + 0.01 * levy
            strategy = self.distance = 0.05 * (self.problem.ub - self.problem.lb)
            pop_comeback.append([pos_new, None, strategy, 0])
        pop_comeback = self.update_target_wrapper_population(pop_comeback)
        self.nfe_per_epoch = self.pop_size + int(0.5 * len(pop_left))
        self.pop = self.get_sorted_strim_population(pop + pop_comeback, self.pop_size)
