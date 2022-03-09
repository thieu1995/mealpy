# !/usr/bin/env python
# Created by "Thieu" at 07:02, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import math
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseNRO(Optimizer):
    """
    The original version of: Nuclear Reaction Optimization (NRO)

    Links:
        1. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8720256

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.NRO import BaseNRO
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
    >>> model = BaseNRO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wei, Z., Huang, C., Wang, X., Han, T. and Li, Y., 2019. Nuclear reaction optimization: A novel and
    powerful physics-based algorithm for global optimization. IEEE Access, 7, pp.66084-66109.
    [2] Wei, Z.L., Zhang, Z.R., Huang, C.Q., Han, B., Tang, S.Q. and Wang, L., 2019, June. An Approach
    Inspired from Nuclear Reaction Processes for Numerical Optimization. In Journal of Physics:
    Conference Series (Vol. 1213, No. 3, p. 032009). IOP Publishing.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def amend_position(self, position=None):
        """
        If solution out of bound at dimension x, then it will re-arrange to random location in the range of domain

        Args:
            position: vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.where(np.logical_and(self.problem.lb <= position, position <= self.problem.ub),
                        position, np.random.uniform(self.problem.lb, self.problem.ub))

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
            Xi = self.amend_position(Xi)
            pop_new.append([Xi, None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)

        # NFu phase

        ## Ionization stage
        ## Calculate the Pa through Eq. (10)
        pop_child = []
        ranked_pop = np.argsort([pop_new[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            X_ion = deepcopy(pop_new[i][self.ID_POS])
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.random():
                i1, i2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                for j in range(self.problem.n_dims):
                    #### Levy flight strategy is described as Eq. 18
                    if pop_new[i2][self.ID_POS][j] == pop_new[i][self.ID_POS][j]:
                        X_ion[j] = pop_new[i][self.ID_POS][j] + alpha * levy_b * (pop_new[i][self.ID_POS][j] - self.g_best[self.ID_POS][j])
                    #### If not, based on Eq. 11, 12
                    else:
                        if np.random.uniform() <= 0.5:
                            X_ion[j] = pop_new[i1][self.ID_POS][j] + np.random.uniform() * (pop_new[i2][self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        else:
                            X_ion[j] = pop_new[i1][self.ID_POS][j] - np.random.uniform() * (pop_new[i2][self.ID_POS][j] - pop_new[i][self.ID_POS][j])

            else:  #### Levy flight strategy is described as Eq. 21
                _, _, worst = self.get_special_solutions(pop_new, worst=1)
                X_worst = worst[0]
                for j in range(self.problem.n_dims):
                    ##### Based on Eq. 21
                    if X_worst[self.ID_POS][j] == self.g_best[self.ID_POS][j]:
                        X_ion[j] = pop_new[i][self.ID_POS][j] + alpha * levy_b * (self.problem.ub[j] - self.problem.lb[j])
                    ##### Based on Eq. 13
                    else:
                        X_ion[j] = pop_new[i][self.ID_POS][j] + round(np.random.uniform()) * np.random.uniform() * \
                                   (X_worst[self.ID_POS][j] - self.g_best[self.ID_POS][j])

            ## Check the boundary and evaluate the fitness function for X_ion
            X_ion = self.amend_position(X_ion)
            pop_child.append([X_ion, None])
        pop_child = self.update_fitness_population(pop_child)
        pop_child = self.greedy_selection_population(pop_new, pop_child)

        ## Fusion Stage

        ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
        pop_new = []
        ranked_pop = np.argsort([pop_child[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            i1, i2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

            #### Generate fusion nucleus
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.random():
                t1 = np.random.uniform() * (pop_child[i1][self.ID_POS] - self.g_best[self.ID_POS])
                t2 = np.random.uniform() * (pop_child[i2][self.ID_POS] - self.g_best[self.ID_POS])
                temp2 = pop_child[i1][self.ID_POS] - pop_child[i2][self.ID_POS]
                X_fu = pop_child[i][self.ID_POS] + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
            #### Else
            else:
                ##### Based on Eq. 22
                check_equal = (pop_child[i1][self.ID_POS] == pop_child[i2][self.ID_POS])
                if check_equal.all():
                    X_fu = pop_child[i][self.ID_POS] + alpha * levy_b * (pop_child[i][self.ID_POS] - self.g_best[self.ID_POS])
                ##### Based on Eq. 16, 17
                else:
                    if np.random.uniform() > 0.5:
                        X_fu = pop_child[i][self.ID_POS] - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) *
                                                                  (self.epoch - epoch) / self.epoch + 1) * (
                                           pop_child[i1][self.ID_POS] - pop_child[i2][self.ID_POS])
                    else:
                        X_fu = pop_child[i][self.ID_POS] - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * epoch / self.epoch + 1) * \
                               (pop_child[i1][self.ID_POS] - pop_child[i2][self.ID_POS])
            X_fu = self.amend_position(X_fu)
            pop_new.append([X_fu, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(pop_child, pop_new)
