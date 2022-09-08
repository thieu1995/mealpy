# !/usr/bin/env python
# Created by "Thieu" at 07:02, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import math
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalNRO(Optimizer):
    """
    The original version of: Nuclear Reaction Optimization (NRO)

    Links:
        1. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8720256

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.NRO import OriginalNRO
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
    >>> model = OriginalNRO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wei, Z., Huang, C., Wang, X., Han, T. and Li, Y., 2019. Nuclear reaction optimization: A novel and
    powerful physics-based algorithm for global optimization. IEEE Access, 7, pp.66084-66109.
    [2] Wei, Z.L., Zhang, Z.R., Huang, C.Q., Han, B., Tang, S.Q. and Wang, L., 2019, June. An Approach
    Inspired from Nuclear Reaction Processes for Numerical Optimization. In Journal of Physics:
    Conference Series (Vol. 1213, No. 3, p. 032009). IOP Publishing.
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

        self.nfe_per_epoch = 3 * self.pop_size
        self.sort_flag = False

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
        rand_pos = np.random.uniform(lb, ub)
        condition = np.logical_and(lb <= position, position <= ub)
        return np.where(condition, position, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        xichma_v = 1
        xichma_u = ((math.gamma(1 + 1.5) * math.sin(math.pi * 1.5 / 2)) / (math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (np.random.normal(0, xichma_u ** 2)) / (np.sqrt(np.abs(np.random.normal(0, xichma_v ** 2))) ** (1.0 / 1.5))

        # NFi phase
        Pb = np.random.uniform()
        Pfi = np.random.uniform()
        freq = 0.05
        alpha = 0.01

        pop_new = []
        for i in range(self.pop_size):
            ## Calculate neutron vector Nei by Eq. (2)
            ## Random 1 more index to select neutron
            temp1 = list(set(range(0, self.pop_size)) - {i})
            i1 = np.random.choice(temp1, replace=False)
            Nei = (self.pop[i][self.ID_POS] + self.pop[i1][self.ID_POS]) / 2
            ## Update population of fission products according to Eq.(3), (6) or (9);
            if np.random.uniform() <= Pfi:
                ### Update based on Eq. 3
                if np.random.uniform() <= Pb:
                    xichma1 = (np.log(epoch + 1) * 1.0 / (epoch + 1)) * np.abs(np.subtract(self.pop[i][self.ID_POS], self.g_best[self.ID_POS]))
                    gauss = np.array([np.random.normal(self.g_best[self.ID_POS][j], xichma1[j]) for j in range(self.problem.n_dims)])
                    Xi = gauss + np.random.uniform() * self.g_best[self.ID_POS] - round(np.random.rand() + 1) * Nei
                ### Update based on Eq. 6
                else:
                    i2 = np.random.choice(temp1, replace=False)
                    xichma2 = (np.log(epoch + 1) * 1.0 / (epoch + 1)) * np.abs(np.subtract(self.pop[i2][self.ID_POS], self.g_best[self.ID_POS]))
                    gauss = np.array([np.random.normal(self.pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem.n_dims)])
                    Xi = gauss + np.random.uniform() * self.g_best[self.ID_POS] - round(np.random.rand() + 2) * Nei
            ## Update based on Eq. 9
            else:
                i3 = np.random.choice(temp1, replace=False)
                xichma2 = (np.log(epoch + 1) * 1.0 / (epoch + 1)) * np.abs(np.subtract(self.pop[i3][self.ID_POS], self.g_best[self.ID_POS]))
                Xi = np.array([np.random.normal(self.pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem.n_dims)])

            ## Check the boundary and evaluate the fitness function
            pos_new = self.amend_position(Xi, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[i] = self.get_better_solution([pos_new, target], self.pop[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        # NFu phase

        ## Ionization stage
        ## Calculate the Pa through Eq. (10)
        pop_child = []
        ranked_pop = np.argsort([self.pop[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            X_ion = deepcopy(self.pop[i][self.ID_POS])
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.random():
                i1, i2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                for j in range(self.problem.n_dims):
                    #### Levy flight strategy is described as Eq. 18
                    if self.pop[i2][self.ID_POS][j] == self.pop[i][self.ID_POS][j]:
                        X_ion[j] = self.pop[i][self.ID_POS][j] + alpha * levy_b * (self.pop[i][self.ID_POS][j] - self.g_best[self.ID_POS][j])
                    #### If not, based on Eq. 11, 12
                    else:
                        if np.random.uniform() <= 0.5:
                            X_ion[j] = self.pop[i1][self.ID_POS][j] + np.random.uniform() * (self.pop[i2][self.ID_POS][j] - self.pop[i][self.ID_POS][j])
                        else:
                            X_ion[j] = self.pop[i1][self.ID_POS][j] - np.random.uniform() * (self.pop[i2][self.ID_POS][j] - self.pop[i][self.ID_POS][j])

            else:  #### Levy flight strategy is described as Eq. 21
                _, _, worst = self.get_special_solutions(self.pop, worst=1)
                X_worst = worst[0]
                for j in range(self.problem.n_dims):
                    ##### Based on Eq. 21
                    if X_worst[self.ID_POS][j] == self.g_best[self.ID_POS][j]:
                        X_ion[j] = self.pop[i][self.ID_POS][j] + alpha * levy_b * (self.problem.ub[j] - self.problem.lb[j])
                    ##### Based on Eq. 13
                    else:
                        X_ion[j] = self.pop[i][self.ID_POS][j] + round(np.random.uniform()) * np.random.uniform() * \
                                   (X_worst[self.ID_POS][j] - self.g_best[self.ID_POS][j])

            ## Check the boundary and evaluate the fitness function for X_ion
            pos_new = self.amend_position(X_ion, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[i] = self.get_better_solution([pos_new, target], self.pop[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)

        ## Fusion Stage

        ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
        pop_new = []
        ranked_pop = np.argsort([self.pop[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            i1, i2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

            #### Generate fusion nucleus
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.random():
                t1 = np.random.uniform() * (self.pop[i1][self.ID_POS] - self.g_best[self.ID_POS])
                t2 = np.random.uniform() * (self.pop[i2][self.ID_POS] - self.g_best[self.ID_POS])
                temp2 = self.pop[i1][self.ID_POS] - self.pop[i2][self.ID_POS]
                X_fu = self.pop[i][self.ID_POS] + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
            #### Else
            else:
                ##### Based on Eq. 22
                check_equal = (self.pop[i1][self.ID_POS] == self.pop[i2][self.ID_POS])
                if check_equal.all():
                    X_fu = self.pop[i][self.ID_POS] + alpha * levy_b * (self.pop[i][self.ID_POS] - self.g_best[self.ID_POS])
                ##### Based on Eq. 16, 17
                else:
                    if np.random.uniform() > 0.5:
                        X_fu = self.pop[i][self.ID_POS] - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) *
                            (self.epoch - epoch) / self.epoch + 1) * (self.pop[i1][self.ID_POS] - self.pop[i2][self.ID_POS])
                    else:
                        X_fu = self.pop[i][self.ID_POS] - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * epoch / self.epoch + 1) * \
                               (self.pop[i1][self.ID_POS] - self.pop[i2][self.ID_POS])
            pos_new = self.amend_position(X_fu, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[i] = self.get_better_solution([pos_new, target], self.pop[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
