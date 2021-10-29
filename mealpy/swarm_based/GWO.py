#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseGWO(Optimizer):
    """
        The original version of: Grey Wolf Optimizer (GWO)
        Link:
            https://ieeexplore.ieee.org/document/7867415
        Notes:
            In this algorithms: Prey means the best position
            https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav
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

    def create_child(self, idx, pop, list_best, a):
        A1, A2, A3 = a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1)
        C1, C2, C3 = 2 * np.random.uniform(), 2 * np.random.uniform(), 2 * np.random.uniform()
        X1 = list_best[0][self.ID_POS] - A1 * np.abs(C1 * list_best[0][self.ID_POS] - pop[idx][self.ID_POS])
        X2 = list_best[1][self.ID_POS] - A2 * np.abs(C2 * list_best[1][self.ID_POS] - pop[idx][self.ID_POS])
        X3 = list_best[2][self.ID_POS] - A3 * np.abs(C3 * list_best[2][self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = (X1 + X2 + X3) / 3.0
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
        # linearly decreased from 2 to 0
        a = 2 - 2 * epoch / (self.epoch - 1)
        _, list_best, _ = self.get_special_solutions(pop, best=3)
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, list_best=list_best, a=a), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, list_best=list_best, a=a), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, list_best=list_best, a=a) for idx in pop_idx]
        return child


class RW_GWO(Optimizer):
    """
        The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)
        Link:
            A novel Random Walk Grey Wolf Optimizer
        Note:
            What a trash paper, the random walk GWO always perform worst than original GWO
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

    def create_child(self, idx, pop, leaders, g_best, b):
        # Eq. 3
        miu1, miu2, miu3 = b * (2 * np.random.uniform() - 1), b * (2 * np.random.uniform() - 1), b * (2 * np.random.uniform() - 1)
        # Eq. 4
        c1, c2, c3 = 2 * np.random.uniform(), 2 * np.random.uniform(), 2 * np.random.uniform()
        X1 = leaders[0][self.ID_POS] - miu1 * np.abs(c1 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
        X2 = leaders[1][self.ID_POS] - miu2 * np.abs(c2 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
        X3 = leaders[2][self.ID_POS] - miu3 * np.abs(c3 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = (X1 + X2 + X3) / 3.0

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
        # linearly decreased from 2 to 0, Eq. 5
        b = 2 - 2 * epoch / (self.epoch - 1)
        # linearly decreased from 2 to 0
        a = 2 - 2 * epoch / (self.epoch - 1)

        _, leaders, _ = self.get_special_solutions(pop, best=3)

        ## Random walk here
        for i in range(0, len(leaders)):
            pos_new = leaders[i][self.ID_POS] + a * np.random.standard_cauchy(self.problem.n_dims)
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], leaders[i]):
                leaders[i] = [pos_new, fit_new]
        pop_idx = np.array(range(0, self.pop_size))
        ## Update other wolfs
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, leaders=leaders, g_best=g_best, b=b), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, leaders=leaders, g_best=g_best, b=b), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, leaders=leaders, g_best=g_best, b=b) for idx in pop_idx]
        pop = self.get_sorted_strim_population(child + leaders, self.pop_size)
        return pop
