#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:01, 16/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFOA(Optimizer):
    """
        The original version of: Fruit-fly Optimization Algorithm (FOA)
            (A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example)
        Link:
            DOI: https://doi.org/10.1016/j.knosys.2011.07.001
        Notes:
            + This optimization can't apply to complicated objective function in this library.
            + So I changed the implementation Original FOA in BaseFOA version
            + This algorithm is the weakest algorithm in MHAs, that's why so many researchers produce paper based
            on this algorithm (Easy to improve, and easy to implement).
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

    def norm_consecutive_adjacent(self, position=None):
        return np.array([np.linalg.norm([position[x], position[x + 1]]) for x in range(0, self.problem.n_dims - 1)] + \
                        [np.linalg.norm([position[-1], position[0]])])

    def create_solution(self):
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        s = self.norm_consecutive_adjacent(position)
        pos = self.amend_position_faster(s)
        fit = self.get_fitness_position(pos)
        return [position, fit]

    def create_child(self, idx, pop, g_best):
        pos_new = pop[idx][self.ID_POS] + np.random.normal(self.problem.lb, self.problem.ub)
        pos_new = self.norm_consecutive_adjacent(pos_new)
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
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best=g_best) for idx in pop_idx]
        return child


class BaseFOA(OriginalFOA):
    """
        My version of: Fruit-fly Optimization Algorithm (FOA)
            (A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example)
        Notes:
            + 1) I changed the fitness function (smell function) by taking the distance each 2 adjacent dimensions
            + 2) Update the position if only it find the better fitness value.
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

    def create_child(self, idx, pop, g_best):
        if np.random.rand() < 0.5:
            pos_new = pop[idx][self.ID_POS] + 0.1 * np.random.normal(self.problem.lb, self.problem.ub)
        else:
            pos_new = g_best[self.ID_POS] + 0.1 * np.random.normal(self.problem.lb, self.problem.ub)
        pos_new = self.norm_consecutive_adjacent(pos_new)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()


class WFOA(OriginalFOA):
    """
        The original version of: Whale Fruit-fly Optimization Algorithm (WFOA)
            (Boosted Hunting-based Fruit Fly Optimization and Advances in Real-world Problems)
        Link:
            https://doi.org/10.1016/j.eswa.2020.113502
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

    def create_child2(self, idx, pop, g_best, a):
        r = np.random.rand()
        A = 2 * a * r - a
        C = 2 * r
        l = np.random.uniform(-1, 1)
        p = 0.5
        b = 1
        if np.random.rand() < p:
            if np.abs(A) < 1:
                D = np.abs(C * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = g_best[self.ID_POS] - A * D
            else:
                x_rand = pop[np.random.randint(self.pop_size)]  # select random 1 position in pop
                D = np.abs(C * x_rand[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = (x_rand[self.ID_POS] - A * D)
        else:
            D1 = np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS])
            pos_new = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best[self.ID_POS]

        pos_new = self.amend_position_faster(pos_new)
        smell = self.norm_consecutive_adjacent(pos_new)
        fit_new = self.get_fitness_position(smell)
        return [pos_new ,fit_new]

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
        a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child2(idx, pop, g_best=g_best, a=a) for idx in pop_idx]
        return child
