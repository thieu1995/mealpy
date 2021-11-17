#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:30, 16/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseJA(Optimizer):
    """
        My original version of: Jaya Algorithm (JA)
            (A simple and new optimization algorithm for solving constrained and unconstrained optimization problems)
        Link:
            https://www.researchgate.net/publication/282532308_Jaya_A_simple_and_new_optimization_algorithm_for_solving_constrained_and_unconstrained_optimization_problems
        Notes:
            + Remove all third loop in algorithm
            + Change the second random variable r2 to Gaussian instead of np.random.uniform
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) + \
                      np.random.normal() * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class OriginalJA(BaseJA):
    """
        The original version of: Jaya Algorithm (JA)
            (A simple and new optimization algorithm for solving constrained and unconstrained optimization problems)
        Link:
            http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                      (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) - \
                      np.random.uniform(0, 1, self.problem.n_dims) * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class LevyJA(BaseJA):
    """
        The original version of: Levy-flight Jaya Algorithm (LJA)
            (An improved Jaya optimization algorithm with Levy flight)
        Link:
            + https://doi.org/10.1016/j.eswa.2020.113902
        Note:
            + This version I still remove all third loop in algorithm
            + The beta value of Levy-flight equal to 1.8 as the best value in the paper.
    """
    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            L1 = self.get_levy_flight_step(multiplier=1.0, beta=1.0, case=-1)
            L2 = self.get_levy_flight_step(multiplier=1.0, beta=1.0, case=-1)
            pos_new = self.pop[idx][self.ID_POS] + np.abs(L1) * (g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) - \
                      np.abs(L2) * (g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
