#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseNMR(Optimizer):
    """
    The original version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Link:
        https://www.doi.org10.1007/s00521-019-04464-7
    """

    def __init__(self, problem, epoch=10000, pop_size=100, bp=0.75, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            bp (float): breeding probability (0.75)
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = bp

    def create_child(self, idx, pop, g_best):
        pos_new = pop[idx][self.ID_POS].copy()
        if idx < self.size_b:  # breeding operators
            if np.random.uniform() < self.bp:
                alpha = np.random.uniform()
                pos_new = (1 - alpha) * pop[idx][self.ID_POS] + alpha * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        else:  # working operators
            t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
            pos_new = pop[idx][self.ID_POS] + np.random.uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

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
            child = [self.create_child(idx, pop, g_best) for idx in pop_idx]
        return child


class ImprovedNMR(BaseNMR):
    """
    My improved version of: Naked Mole-rat Algorithm (NMRA)
        (The naked mole-rat algorithm)
    Notes:
        + Using mutation probability
        + Using levy-flight
        + Using crossover operator
    """

    def __init__(self, problem, epoch=10000, pop_size=100, bp=0.75, pm=0.01, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            bp (float): breeding probability (0.75)
            pm ():
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, bp, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        self.pm = pm

    def create_child2(self, idx, pop, g_best, epoch):
        # Exploration
        if idx < self.size_b:  # breeding operators
            if np.random.uniform() < self.bp:
                pos_new = pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
            else:
                levy_step = self.get_levy_flight_step(beta=1, multiplier=0.001, case=-1)
                pos_new = pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                            levy_step * (pop[idx][self.ID_POS] - g_best[self.ID_POS])
        # Exploitation
        else:  # working operators
            if np.random.uniform() < 0.5:
                t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                pos_new = pop[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
            else:
                pos_new = self._crossover_random__(pop, g_best)

        # Mutation
        temp = np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, temp, pos_new)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

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
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child2(idx, pop, g_best, epoch) for idx in pop_idx]
        return child

    def _crossover_random__(self, pop, g_best):
        start_point = np.random.randint(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[np.random.randint(0, self.pop_size)][self.ID_POS]
        new_temp = g_best[self.ID_POS].copy()
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp
