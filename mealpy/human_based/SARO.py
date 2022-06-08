# !/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSARO(Optimizer):
    """
    My changed version of: Search And Rescue Optimization (SARO)

    Notes
    ~~~~~
    All third loop is removed

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): [10, 20], maximum unsuccessful search number, default = 15

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.SARO import BaseSARO
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
    >>> se = 0.5
    >>> mu = 50
    >>> model = BaseSARO(problem_dict1, epoch, pop_size, se, mu)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=15, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.se = self.validator.check_float("se", se, (0, 1.0))
        self.mu = self.validator.check_int("mu", mu, [2, 2+int(self.pop_size/2)])

        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True
        ## Dynamic variable
        self.dyn_USN = np.zeros(self.pop_size)

    def after_initialization(self):
        pop = self.pop + self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(pop)

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        condition = np.logical_and(lb <= position, position <= ub)
        rand_pos = np.random.uniform(lb, ub)
        return np.where(condition, position, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_x = deepcopy(self.pop[:self.pop_size])
        pop_m = deepcopy(self.pop[self.pop_size:])

        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - self.pop[k][self.ID_POS]

            #### Remove third loop here, also using random flight back when out of bound
            pos_new_1 = self.pop[k][self.ID_POS] + np.random.uniform() * sd
            pos_new_2 = pop_x[idx][self.ID_POS] + np.random.uniform() * sd
            pos_new = np.where(np.logical_and(np.random.uniform(0, 1, self.problem.n_dims) < self.se,
                                              self.pop[k][self.ID_TAR] < pop_x[idx][self.ID_TAR]), pos_new_1, pos_new_2)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        pop = deepcopy(pop_x) + deepcopy(pop_m)
        pop_new = []
        for idx in range(self.pop_size):
            ## Individual phase
            k1, k2 = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            #### Remove third loop here, and flight back strategy now be a random
            pos_new = self.g_best[self.ID_POS] + np.random.uniform() * (pop[k1][self.ID_POS] - pop[k2][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution(self.problem.lb, self.problem.ub)
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m


class OriginalSARO(BaseSARO):
    """
    The original version of: Search And Rescue Optimization (SARO)

    Links:
       1. https://doi.org/10.1155/2019/2482543

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): [10, 20], maximum unsuccessful search number, default = 15

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.SARO import OriginalSARO
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
    >>> se = 0.5
    >>> mu = 50
    >>> model = OriginalSARO(problem_dict1, epoch, pop_size, se, mu)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Shabani, A., Asgarian, B., Gharebaghi, S.A., Salido, M.A. and Giret, A., 2019. A new optimization
    algorithm based on search and rescue operations. Mathematical Problems in Engineering, 2019.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=15, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        super().__init__(problem, epoch, pop_size, se, mu, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_x = deepcopy(self.pop[:self.pop_size])
        pop_m = deepcopy(self.pop[self.pop_size:])

        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - self.pop[k][self.ID_POS]
            j_rand = np.random.randint(0, self.problem.n_dims)
            r1 = np.random.uniform(-1, 1)

            pos_new = deepcopy(pop_x[idx][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() < self.se or j == j_rand:
                    if self.compare_agent(self.pop[k], pop_x[idx]):
                        pos_new[j] = self.pop[k][self.ID_POS][j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[idx][self.ID_POS][j] + r1 * sd[j]
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = deepcopy(pop_x[idx])
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        ## Individual phase
        pop = deepcopy(pop_x) + deepcopy(pop_m)
        pop_new = []
        for idx in range(0, self.pop_size):
            k, m = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            pos_new = pop_x[idx][self.ID_POS] + np.random.uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx]
                pop_x[idx] = deepcopy(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution(self.problem.lb, self.problem.ub)
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m
