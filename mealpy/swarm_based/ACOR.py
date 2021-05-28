#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:14, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import normal
from numpy import sqrt, pi, exp, array, sum, zeros, repeat, abs
from mealpy.root import Root


class BaseACOR(Root):
    """
        The original version of: Ant Colony Optimization Continuous (ACOR)
            Ant Colony Optimization for Continuous Domains (ACOR)
        Link:
            https://doi.org/10.1016/j.ejor.2006.06.046
        My improvements:
            + Using Gaussian Distribution instead of random number (normal() function)      (1)
            + Amend solution when they went out of space    (2)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 sample_count=50, q=0.5, zeta=1, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.sample_count = sample_count # Number of Newly Generated Samples
        self.q = q          # Intensification Factor (Selection Pressure)
        self.zeta = zeta    # Deviation-Distance Ratio

    def train(self):
        # Create Initial Population (Sorted)
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Calculate Selection Probabilities
            pop_rank = array([i for i in range(1, self.pop_size + 1)])
            qn = self.q * self.pop_size
            w = 1 / (sqrt(2 * pi) * qn) * exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
            p = w / sum(w)  # Normalize to find the probability.

            # Means and Standard Deviations
            matrix_pos = array([solution[self.ID_POS] for solution in pop])
            matrix_sigma = []
            for i in range(0, self.pop_size):
                matrix_i = repeat(pop[i][self.ID_POS].reshape((1, -1)), self.pop_size, axis=0)
                D = sum(abs(matrix_pos - matrix_i), axis=0)
                temp = self.zeta * D / (self.pop_size - 1)
                matrix_sigma.append(temp)
            matrix_sigma = array(matrix_sigma)

            # Generate Samples
            pop_child = []
            for i in range(0, self.sample_count):
                child = zeros(self.problem_size)
                for j in range(0, self.problem_size):
                    idx = self.get_index_roulette_wheel_selection(p)
                    child[j] = pop[idx][self.ID_POS][j] + normal() * matrix_sigma[idx, j]       # (1)
                child = self.amend_position_faster(child)                                       # (2)
                fit = self.get_fitness_position(child)
                pop_child.append([child, fit])

            # Merge, Sort and Selection, update global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop + pop_child, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
