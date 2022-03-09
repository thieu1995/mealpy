# !/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseNMRA(Optimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (NMRA)

    Links:
        1. https://www.doi.org10.1007/s00521-019-04464-7

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.NMRA import BaseNMRA
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
    >>> pb = 0.75
    >>> model = BaseNMRA(problem_dict1, epoch, pop_size, pb)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm.
    Neural Computing and Applications, 31(12), pp.8837-8857.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.75, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): probability of breeding, default = 0.75
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.pb = pb

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            if idx < self.size_b:  # breeding operators
                if np.random.uniform() < self.pb:
                    alpha = np.random.uniform()
                    pos_new = (1 - alpha) * self.pop[idx][self.ID_POS] + alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            else:  # working operators
                t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.pop[t1][self.ID_POS] - self.pop[t2][self.ID_POS])
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class ImprovedNMRA(Optimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (I-NMRA)

    Notes:
    + Use mutation probability idea
    + Use crossover operator
    + Use Levy-flight technique

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75
        + pm (float): [0.01, 0.1], probability of mutation, default = 0.01

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.NMRA import ImprovedNMRA
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
    >>> pb = 0.75
    >>> pm = 0.01
    >>> model = ImprovedNMRA(problem_dict1, epoch, pop_size, pb, pm)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm.
    Neural Computing and Applications, 31(12), pp.8837-8857.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.75, pm=0.01, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): breeding probability, default = 0.75
            pm (float): probability of mutation, default = 0.01
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.pb = pb
        self.pm = pm

    def _crossover_random(self, pop, g_best):
        start_point = np.random.randint(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[np.random.randint(0, self.pop_size)][self.ID_POS]
        new_temp = deepcopy(g_best[self.ID_POS])
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            if idx < self.size_b:  # breeding operators
                if np.random.uniform() < self.pb:
                    pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                              (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    levy_step = self.get_levy_flight_step(beta=1, multiplier=0.001, case=-1)
                    pos_new = self.pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                              levy_step * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            # Exploitation
            else:  # working operators
                if np.random.uniform() < 0.5:
                    t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                    pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                              (self.pop[t1][self.ID_POS] - self.pop[t2][self.ID_POS])
                else:
                    pos_new = self._crossover_random(self.pop, self.g_best)
            # Mutation
            temp = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, temp, pos_new)
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
