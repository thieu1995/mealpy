#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:30, 16/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
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

    def create_child(self, idx, pop, g_best, g_worst):
        pos_new = pop[idx][self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - np.abs(pop[idx][self.ID_POS])) + \
                  np.random.normal() * (g_worst[self.ID_POS] - np.abs(pop[idx][self.ID_POS]))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        _, best, worst = self.get_special_solutions(pop, best=1, worst=1)
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=best[0], g_worst=worst[0]), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=best[0], g_worst=worst[0]), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, best[0], worst[0]) for idx in pop_idx]
        return child


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
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def create_child(self, idx, pop, g_best, g_worst):
        pos_new = pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (g_best[self.ID_POS] - np.abs(pop[idx][self.ID_POS])) - \
            np.random.uniform(0, 1, self.problem.n_dims) * (g_worst[self.ID_POS] - np.abs(pop[idx][self.ID_POS]))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]


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

    def create_child(self, idx, pop, g_best, g_worst):
        L1 = self.get_levy_flight_step(multiplier=1.0, beta=1.0, case=-1)
        L2 = self.get_levy_flight_step(multiplier=1.0, beta=1.0, case=-1)
        pos_new = pop[idx][self.ID_POS] + np.abs(L1) * (g_best[self.ID_POS] - np.abs(pop[idx][self.ID_POS])) - \
                  np.abs(L2) * (g_worst[self.ID_POS] - np.abs(pop[idx][self.ID_POS]))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

