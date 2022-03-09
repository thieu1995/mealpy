#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalLCO(Optimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)

    Links:
        1. https://doi.org/10.1007/s00500-019-04443-z

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import OriginalLCO
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
    >>> r1 = 2.35
    >>> model = OriginalLCO(problem_dict1, epoch, pop_size, r1)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.r1 = r1
        self.n_agents = int(np.ceil(np.sqrt(self.pop_size)))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for i in range(0, self.pop_size):
            rand_number = np.random.random()

            if rand_number > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.random() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif rand_number < 0.7:  # Update using Eq. 2-6
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                if i == 0:
                    pop_new.append(deepcopy(self.g_best))
                    continue
                else:
                    best_diff = f1 * self.r1 * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                    better_diff = f2 * self.r1 * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                temp = self.pop[i][self.ID_POS] + np.random.random() * better_diff + np.random.random() * best_diff
            else:
                temp = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.random()
            pos_new = self.amend_position(temp)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class BaseLCO(OriginalLCO):
    """
    My changed version of: Life Choice-based Optimization (LCO)

    Notes
    ~~~~~
    I only change the flow with simpler if else statement than the original

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import BaseLCO
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
    >>> r1 = 2.35
    >>> model = BaseLCO(problem_dict1, epoch, pop_size, r1)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(problem, epoch, pop_size, r1, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for i in range(0, self.pop_size):
            rand = np.random.random()

            if rand > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.random() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if i != 0:
                    better_diff = f * self.r1 * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                else:
                    better_diff = f * self.r1 * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                best_diff = (1 - f) * self.r1 * (self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS])
                temp = self.pop[i][self.ID_POS] + np.random.uniform() * better_diff + np.random.uniform() * best_diff
            else:
                temp = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(temp)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class ImprovedLCO(Optimizer):
    """
    My improved version of: Life Choice-based Optimization (ILCO)

    Notes
    ~~~~~
    + The flow of the original LCO is kept.
    + Add gaussian distribution and mutation mechanism
    + Remove the hyper-parameter r1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import BaseLCO
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
    >>> model = BaseLCO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

        self.pop_len = int(self.pop_size / 2)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for i in range(0, self.pop_size):
            rand = np.random.random()
            if rand > 0.875:  # Update using Eq. 1, update from n best position
                n = int(np.ceil(np.sqrt(self.pop_size)))
                pos_new = np.array([np.random.uniform() * self.pop[j][self.ID_POS] for j in range(0, n)])
                pos_new = np.mean(pos_new, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if i != 0:
                    better_diff = f * np.random.uniform() * (self.pop[i - 1][self.ID_POS] - self.pop[i][self.ID_POS])
                else:
                    better_diff = f * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                best_diff = (1 - f) * np.random.uniform() * (self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS])
                pos_new = self.pop[i][self.ID_POS] + better_diff + best_diff
            else:
                pos_new = self.problem.ub - (self.pop[i][self.ID_POS] - self.problem.lb) * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)

        ## Sort the updated population based on fitness
        pop, local_best = self.get_global_best_solution(pop_new)
        pop_s1, pop_s2 = pop[:self.pop_len], pop[self.pop_len:]

        ## Mutation scheme
        for i in range(0, self.pop_len):
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))
            pop_s1[i][self.ID_POS] = self.amend_position(pos_new)
        pop_s1 = self.update_fitness_population(pop_s1)

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        for i in range(0, self.pop_len):
            pos_new = (local_best[self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            pop_s2[i][self.ID_POS] = self.amend_position(pos_new)
        pop_s2 = self.update_fitness_population(pop_s2)

        ## Construct a new population
        self.pop = pop_s1 + pop_s2
