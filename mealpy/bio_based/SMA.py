#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:22, 12/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSMA(Optimizer):
    """
        My modified version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Link:
             https://doi.org/10.1016/j.future.2020.03.055
             https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization
        Notes:
            + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
            + Check bound and update fitness after each individual move instead of after the whole population move in the original version
            + My version not only faster but also better
    """

    ID_WEI = 2

    def __init__(self, problem, epoch=10000, pop_size=100, pr=0.03, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pr (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pr = pr

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], weight]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        weight = np.zeros(self.problem.n_dims)
        return [position, fitness, weight]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # plus eps to avoid denominator zero
        s = self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[-1][self.ID_FIT][self.ID_TAR] + self.EPSILON

        # calculate the fitness weight of each slime mold
        for i in range(0, self.pop_size):
            # Eq.(2.5)
            if i <= int(self.pop_size / 2):
                self.pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.problem.n_dims) * \
                                      np.log10((self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[i][self.ID_FIT][self.ID_TAR]) / s + 1)
            else:
                self.pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.problem.n_dims) * \
                                      np.log10((self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[i][self.ID_FIT][self.ID_TAR]) / s + 1)

        a = np.arctanh(-((epoch + 1) / self.epoch) + 1)         # Eq.(2.4)
        b = 1 - (epoch + 1) / self.epoch

        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            if np.random.uniform() < self.pr:  # Eq.(2.7)
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            else:
                p = np.tanh(np.abs(self.pop[idx][self.ID_FIT][self.ID_TAR] - self.g_best[self.ID_FIT][self.ID_TAR]))  # Eq.(2.2)
                vb = np.random.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = np.random.uniform(-b, b, self.problem.n_dims)

                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)

                pos_1 = self.g_best[self.ID_POS] + vb * (self.pop[idx][self.ID_WEI] * self.pop[id_a][self.ID_POS] - self.pop[id_b][self.ID_POS])
                pos_2 = vc * self.pop[idx][self.ID_POS]
                pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < p, pos_1, pos_2)

            # Check bound and re-calculate fitness after each individual move
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None, np.zeros(self.problem.n_dims)])
        self.pop = self.update_fitness_population(pop_new)


class OriginalSMA(BaseSMA):
    """
        This version developed by one on my student: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Link:
            https://doi.org/10.1016/j.future.2020.03.055
    """

    ID_WEI = 2

    def __init__(self, problem, epoch=10000, pop_size=100, pr=0.03, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            pr (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(problem, epoch, pop_size, pr, **kwargs)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """

        s = self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[-1][self.ID_FIT][self.ID_TAR] + self.EPSILON  # plus eps to avoid denominator zero

        # calculate the fitness weight of each slime mold
        for i in range(0, self.pop_size):
            # Eq.(2.5)
            if i <= int(self.pop_size / 2):
                self.pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.problem.n_dims) * \
                                      np.log10((self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[i][self.ID_FIT][self.ID_TAR]) / s + 1)
            else:
                self.pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.problem.n_dims) * \
                                      np.log10((self.g_best[self.ID_FIT][self.ID_TAR] - self.pop[i][self.ID_FIT][self.ID_TAR]) / s + 1)

        a = np.arctanh(-((epoch + 1) / self.epoch) + 1)  # Eq.(2.4)
        b = 1 - (epoch + 1) / self.epoch

        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            current_agent = deepcopy(self.pop[idx])
            if np.random.uniform() < self.pr:  # Eq.(2.7)
                current_agent[self.ID_POS] = np.random.uniform(self.problem.lb, self.problem.ub)
            else:
                p = np.tanh(np.abs(current_agent[self.ID_FIT][self.ID_TAR] - self.g_best[self.ID_FIT][self.ID_TAR]))  # Eq.(2.2)
                vb = np.random.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = np.random.uniform(-b, b, self.problem.n_dims)
                for j in range(0, self.problem.n_dims):
                    # two positions randomly selected from population
                    id_a, id_b = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    if np.random.uniform() < p:  # Eq.(2.1)
                        current_agent[self.ID_POS][j] = self.g_best[self.ID_POS][j] + vb[j] * (
                                current_agent[self.ID_WEI][j] * self.pop[id_a][self.ID_POS][j] - self.pop[id_b][self.ID_POS][j])
                    else:
                        current_agent[self.ID_POS][j] = vc[j] * current_agent[self.ID_POS][j]
            pos_new = self.amend_position_faster(current_agent[self.ID_POS])
            pop_new.append([pos_new, None, np.zeros(self.problem.n_dims)])
        self.pop = self.update_fitness_population(pop_new)
