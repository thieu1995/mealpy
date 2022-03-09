# !/usr/bin/env python
# Created by "Thieu" at 16:10, 08/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalArchOA(Optimizer):
    """
    The original version of: Archimedes Optimization Algorithm (ArchOA)

    Links:
        1. https://doi.org/10.1007/s10489-020-01893-z

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + c1 (int): factor, default belongs to [1, 2]
        + c2 (int): factor, Default belongs to [2, 4, 6]
        + c3 (int): factor, Default belongs to [1, 2]
        + c4 (float): factor, Default belongs to [0.5, 1]
        + acc_max (float): acceleration max, Default 0.9
        + acc_min (float): acceleration min, Default 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.ArchOA import OriginalArchOA
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
    >>> c1 = 2
    >>> c2 = 5
    >>> c3 = 2
    >>> c4 = 0.5
    >>> acc_max = 0.9
    >>> acc_min = 0.1
    >>> model = OriginalArchOA(problem_dict1, epoch, pop_size, c1, c2, c3, c4, acc_max, acc_min)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Hussain, K., Houssein, E.H., Mabrouk, M.S. and Al-Atabany, W., 2021. Archimedes optimization
    algorithm: a new metaheuristic algorithm for solving optimization problems. Applied Intelligence, 51(3), pp.1531-1551.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_DEN = 2  # Density
    ID_VOL = 3  # Volume
    ID_ACC = 4  # Acceleration

    def __init__(self, problem, epoch=10000, pop_size=100, c1=2, c2=6, c3=2, c4=0.5, acc_max=0.9, acc_min=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (int): factor, default belongs [1, 2]
            c2 (int): factor, Default belongs [2, 4, 6]
            c3 (int): factor, Default belongs [1, 2]
            c4 (float): factor, Default belongs [0.5, 1]
            acc_max (float): acceleration max, Default 0.9
            acc_min (float): acceleration min, Default 0.1
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.acc_max = acc_max
        self.acc_min = acc_min

    def create_solution(self):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], density, volume, acceleration]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        position = self.amend_position(position)
        fitness = self.get_fitness_position(position=position)
        den = np.random.uniform(self.problem.lb, self.problem.ub)
        vol = np.random.uniform(self.problem.lb, self.problem.ub)
        acc = self.problem.lb + np.random.uniform(self.problem.lb, self.problem.ub) * (self.problem.ub - self.problem.lb)
        return [position, fitness, den, vol, acc]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Transfer operator Eq. 8
        tf = np.exp((epoch + 1) / self.epoch - 1)
        ## Density decreasing factor Eq. 9
        ddf = np.exp(1 - (epoch + 1) / self.epoch) - (epoch + 1) / self.epoch

        list_acc = []
        ## Calculate new density, volume and acceleration
        for i in range(0, self.pop_size):
            # Update density and volume of each object using Eq. 7
            new_den = self.pop[i][self.ID_DEN] + np.random.uniform() * (self.g_best[self.ID_DEN] - self.pop[i][self.ID_DEN])
            new_vol = self.pop[i][self.ID_VOL] + np.random.uniform() * (self.g_best[self.ID_VOL] - self.pop[i][self.ID_VOL])

            # Exploration phase
            if tf <= 0.5:
                # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                id_rand = np.random.choice(list(set(range(0, self.pop_size)) - {i}))
                new_acc = (self.pop[id_rand][self.ID_DEN] + self.pop[id_rand][self.ID_VOL] * self.pop[id_rand][self.ID_ACC]) / (new_den * new_vol)
            else:
                new_acc = (self.g_best[self.ID_DEN] + self.g_best[self.ID_VOL] * self.g_best[self.ID_ACC]) / (new_den * new_vol)
            list_acc.append(new_acc)
            self.pop[i][self.ID_DEN] = new_den
            self.pop[i][self.ID_VOL] = new_vol
        min_acc = np.min(list_acc)
        max_acc = np.max(list_acc)
        ## Normalize acceleration using Eq. 12
        for i in range(0, self.pop_size):
            self.pop[i][self.ID_ACC] = self.acc_max * (self.pop[i][self.ID_ACC] - min_acc) / (max_acc - min_acc) + self.acc_min

        pop_new = []
        for idx in range(0, self.pop_size):
            solution = deepcopy(self.pop[idx])
            if tf <= 0.5:  # update position using Eq. 13
                id_rand = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                pos_new = self.pop[idx][self.ID_POS] + self.c1 * np.random.uniform() * \
                          self.pop[idx][self.ID_ACC] * ddf * (self.pop[id_rand][self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                p = 2 * np.random.rand() - self.c4
                f = 1 if p <= 0.5 else -1
                t = self.c3 * tf
                pos_new = self.g_best[self.ID_POS] + f * self.c2 * np.random.rand() * self.pop[idx][self.ID_ACC] * \
                          ddf * (t * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            solution[self.ID_POS] = self.amend_position(pos_new)
            pop_new.append(solution)
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
