#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseWOA(Optimizer):
    """
        The original version of: Whale Optimization Algorithm (WOA)
            - In this algorithms: Prey means the best position
        Link:
            https://doi.org/10.1016/j.advengsoft.2016.01.008
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
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best, a):
        r = np.random.rand()
        A = 2 * a * r - a
        C = 2 * r
        l = np.random.uniform(-1, 1)
        p = 0.5
        b = 1
        if np.random.uniform() < p:
            if np.abs(A) < 1:
                D = np.abs(C * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = g_best[self.ID_POS] - A * D
            else:
                # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                x_rand = self.create_solution()
                D = np.abs(C * x_rand[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = x_rand[self.ID_POS] - A * D
        else:
            D1 = np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS])
            pos_new = g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

        ## batch size idea
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) % self.pop_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

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
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, a) for idx in pop_idx]
        return child


class HI_WOA(Optimizer):
    """
        The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)
            A hybrid improved whale optimization algorithm
        Link:
            https://ieenp.explore.ieee.org/document/8900003
    """

    def __init__(self, problem, epoch=10000, pop_size=100, feedback_max=10, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.feedback_max = feedback_max
        # The maximum of times g_best doesn't change -> need to change half of population
        self.n_changes = int(pop_size/2)

        ## Dynamic variable
        self.dyn_feedback_count = 0

    def create_child(self, idx, pop, g_best, a):
        r = np.random.rand()
        A = 2 * a * r - a
        C = 2 * r
        l = np.random.uniform(-1, 1)
        p = 0.5
        b = 1
        if np.random.uniform() < p:
            if np.abs(A) < 1:
                D = np.abs(C * g_best[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = g_best[self.ID_POS] - A * D
            else:
                # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                x_rand = self.create_solution()
                D = np.abs(C * x_rand[self.ID_POS] - pop[idx][self.ID_POS])
                pos_new = x_rand[self.ID_POS] - A * D
        else:
            D1 = np.abs(g_best[self.ID_POS] - pop[idx][self.ID_POS])
            pos_new = g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

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
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))    # Eq. 8
        w = 0.5 + 0.5 * (epoch / self.epoch) ** 2                   # Eq. 9
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, a=a), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, a) for idx in pop_idx]

        ## Feedback Mechanism
        _, current_best = self.get_global_best_solution(child)
        if current_best[self.ID_FIT][self.ID_TAR] == g_best[self.ID_FIT][self.ID_TAR]:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = np.random.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_new = [self.create_solution() for _ in range(0, self.n_changes)]
            for idx_counter, idx in enumerate(idx_list):
                child[idx] = pop_new[idx_counter]
        return child
