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

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
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
    >>> model = OriginalArchOA(epoch, pop_size, c1, c2, c3, c4, acc_max, acc_min)
    >>> best_position, best_fitness = model.solve(problem_dict1)
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

    def __init__(self, epoch=10000, pop_size=100, c1=2, c2=6, c3=2, c4=0.5, acc_max=0.9, acc_min=0.1, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (int): factor, default belongs [1, 2]
            c2 (int): factor, Default belongs [2, 4, 6]
            c3 (int): factor, Default belongs [1, 2]
            c4 (float): factor, Default belongs [0.5, 1]
            acc_max (float): acceleration max, Default 0.9
            acc_min (float): acceleration min, Default 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_int("c1", c1, [1, 3])
        self.c2 = self.validator.check_int("c2", c2, [2, 6])
        self.c3 = self.validator.check_int("c3", c3, [1, 3])
        self.c4 = self.validator.check_float("c4", c4, (0, 1.0))
        self.acc_max = self.validator.check_float("acc_max", acc_max, (0.3, 1.0))
        self.acc_min = self.validator.check_float("acc_min", acc_min, (0, 0.3))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "c3", "c4", "acc_max", "acc_min"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, density, volume, acceleration]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        den = np.random.uniform(lb, ub)
        vol = np.random.uniform(lb, ub)
        acc = lb + np.random.uniform(lb, ub) * (ub - lb)
        return [position, target, den, vol, acc]

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
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            solution[self.ID_POS] = pos_new
            pop_new.append(solution)
            if self.mode not in self.AVAILABLE_MODES:
                solution[self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(solution, self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
