#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:14, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseACOR(Optimizer):
    """
        The original version of: Ant Colony Optimization Continuous (ACOR)
            Ant Colony Optimization for Continuous Domains (ACOR)
        Link:
            https://doi.org/10.1016/j.ejor.2006.06.046
        My improvements:
            + Using Gaussian Distribution instead of random number (np.random.normal() function)      (1)
            + Amend solution when they went out of space    (2)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, sample_count=50, q=0.5, zeta=1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sample_count (int): Number of Newly Generated Samples, default = 50
            q (float): Intensification Factor (Selection Pressure), default = 0.5
            zeta (int): Deviation-Distance Ratio, default = 1
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.sample_count = sample_count
        self.q = q
        self.zeta = zeta

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Calculate Selection Probabilities
        pop = self.pop[:self.pop_size]
        pop_rank = np.array([i for i in range(1, self.pop_size + 1)])
        qn = self.q * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)  # Normalize to find the probability.

        # Means and Standard Deviations
        matrix_pos = np.array([solution[self.ID_POS] for solution in pop])
        matrix_sigma = []
        for i in range(0, self.pop_size):
            matrix_i = np.repeat(pop[i][self.ID_POS].reshape((1, -1)), self.pop_size, axis=0)
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (self.pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)

        # Generate Samples
        pop_new = []
        for idx in range(0, self.sample_count):
            # Generate Samples
            child = np.zeros(self.problem.n_dims)
            for j in range(0, self.problem.n_dims):
                idx = self.get_index_roulette_wheel_selection(matrix_p)
                child[j] = pop[idx][self.ID_POS][j] + np.random.normal() * matrix_sigma[idx, j]  # (1)
            pos_new = self.amend_position_faster(child)  # (2)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = pop + pop_new
