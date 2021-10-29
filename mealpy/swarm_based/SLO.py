#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:05, 03/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from math import gamma
import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseSLO(Optimizer):
    """
        The original version of: Sea Lion Optimization Algorithm (SLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
        Notes:
            + The original paper is unclear in some equations and parameters
            + This version is based on my expertise
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

    def create_child(self, idx, pop, g_best, SP_leader, c):
        if SP_leader < 0.25:
            if c < 1:
                pos_new = g_best[self.ID_POS] - c * np.abs(2 * np.random.rand() * g_best[self.ID_POS] - pop[idx][self.ID_POS])
            else:
                ri = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))  # random index
                pos_new = pop[ri][self.ID_POS] - c * np.abs(2 * np.random.rand() * pop[ri][self.ID_POS] - pop[idx][self.ID_POS])
        else:
            pos_new = np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS]) * np.cos(2 * np.pi * np.random.uniform(-1, 1)) + g_best[self.ID_POS]
        # In the paper doesn't check also doesn't update old solution at this point
        pos_new = self.amend_position_random(pos_new)
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
        c = 2 - 2 * epoch / self.epoch
        t0 = np.random.rand()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)  # In the paper this is not clear how to calculate

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, SP_leader=SP_leader, c=c), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, SP_leader=SP_leader, c=c), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, SP_leader, c) for idx in pop_idx]
        return child


class ModifiedSLO(Optimizer):
    """
        My modified version of: Sea Lion Optimization (ISLO)
            (Sea Lion Optimization Algorithm)
        Noted:
            + Using the idea of shrink encircling combine with levy flight techniques
            + Also using the idea of local best in PSO
    """
    ID_LOC_POS = 2
    ID_LOC_FIT = 3

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

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        ## Increase exploration at the first initial population using opposition-based learning.
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        local_pos = self.problem.lb + self.problem.ub - position
        local_fit = self.get_fitness_position(local_pos)
        if fitness < local_fit:
            return [local_pos, local_fit, position, fitness]
        else:
            return [position, fitness, local_pos, local_fit]

    def _shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2))
        xich_ma_1 = np.power(up / down, 1 / beta)
        xich_ma_2 = 1
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (np.power(np.abs(b), 1 / beta)) * dist * c
        D = np.random.uniform(self.problem.lb, self.problem.ub)
        levy = LB * D
        return (current_pos - np.sqrt(epoch + 1) * np.sign(np.random.random(1) - 0.5)) * levy

    def create_child(self, idx, pop, epoch, g_best, SP_leader, c, pa):
        if SP_leader >= 0.6:
            new_pos = np.cos(2 * np.pi * np.random.normal(0, 1)) * np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS]) + g_best[self.ID_POS]
        else:
            if np.random.uniform() < pa:
                dist1 = np.random.uniform() * np.abs(2 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                new_pos = self._shrink_encircling_levy__(pop[idx][self.ID_POS], epoch, dist1, c)
            else:
                rand_SL = pop[np.random.randint(0, self.pop_size)][self.ID_LOC_POS]
                rand_SL = 2 * g_best[self.ID_POS] - rand_SL
                new_pos = rand_SL - c * np.abs(np.random.uniform() * rand_SL - pop[idx][self.ID_POS])

        new_pos = self.amend_position_random(new_pos)
        new_fit = self.get_fitness_position(new_pos)
        if self.compare_agent([new_pos, new_fit], pop[idx]):
            pop[idx][self.ID_LOC_POS] = new_pos.copy()
            pop[idx][self.ID_LOC_FIT] = new_fit
            return [new_pos, new_fit, pop[idx][self.ID_LOC_POS], pop[idx][self.ID_LOC_FIT]]
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

        c = 2 - 2 * epoch / self.epoch
        if c > 1:
            pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
        else:
            pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
        SP_leader = np.random.uniform(0, 1)

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best, SP_leader=SP_leader, c=c, pa=pa), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, epoch=epoch, g_best=g_best, SP_leader=SP_leader, c=c, pa=pa), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop=pop, epoch=epoch, g_best=g_best, SP_leader=SP_leader, c=c, pa=pa) for idx in pop_idx]
        return child



class ISLO(ModifiedSLO):
    """
        My improved version of: Improved Sea Lion Optimization Algorithm (ISLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
    """

    def __init__(self, problem, epoch=10000, pop_size=100, c1=1.2, c2=1.2, **kwargs):
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

        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2

    def create_child2(self, idx, pop, g_best, SP_leader, c):
        if SP_leader < 0.5:
            if c < 1:  # Exploitation improved by historical movement + global best affect
                # pos_new = g_best[self.ID_POS] - c * np.abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                dif1 = np.abs(2 * np.random.rand() * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                dif2 = np.abs(2 * np.random.rand() * pop[idx][self.ID_LOC_POS] - pop[idx][self.ID_POS])
                pos_new = self.c1 * np.random.rand() * (pop[idx][self.ID_POS] - c * dif1) + self.c2 * np.random.rand() * (pop[idx][self.ID_POS] - c * dif2)
            else:  # Exploration improved by opposition-based learning
                # Create a new solution by equation below
                # Then create an opposition solution of above solution
                # Compare both of them and keep the good one (Searching at both direction)
                pos_new = g_best[self.ID_POS] + c * np.random.normal(0, 1, self.problem.n_dims) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pos_new_oppo = self.problem.lb + self.problem.ub - g_best[self.ID_POS] + np.random.rand() * (g_best[self.ID_POS] - pos_new)
                fit_new_oppo = self.get_fitness_position(pos_new_oppo)
                if self.compare_agent([pos_new_oppo, fit_new_oppo], [pos_new, fit_new]):
                    pos_new = pos_new_oppo
        else:  # Exploitation
            pos_new = g_best[self.ID_POS] + np.cos(2 * np.pi * np.random.uniform(-1, 1)) * np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new, pos_new.copy(), fit_new]
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

        c = 2 - 2 * epoch / self.epoch
        t0 = np.random.rand()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, SP_leader=SP_leader, c=c), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, SP_leader=SP_leader, c=c), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child2(idx, pop, g_best, SP_leader, c) for idx in pop_idx]
        return child

