#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:34, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseFPA(Optimizer):
    """
        The original version of: Flower Pollination Algorithm (FPA)
            (Flower Pollination Algorithm for Global Optimization)
    Link:
        https://doi.org/10.1007/978-3-642-32894-7_27
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_s=0.8, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.p_s = p_s

    def create_child(self, idx, pop_copy, g_best, epoch):
        if np.random.uniform() < self.p_s:
            levy = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = pop_copy[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                      levy * (pop_copy[idx][self.ID_POS] - g_best[self.ID_POS])
        else:
            id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = pop_copy[idx][self.ID_POS] + np.random.uniform() * (pop_copy[id1][self.ID_POS] - pop_copy[id2][self.ID_POS])
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

        # batch size idea to update the global best
        # if self.batch_idea:
        #     if (idx + 1) % self.batch_size == 0:
        #         self.update_global_best_solution(pop_copy)
        # else:
        #     if (idx + 1) % self.pop_size == 0:
        #         self.update_global_best_solution(pop_copy)

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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, g_best=g_best, epoch=epoch), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, g_best=g_best, epoch=epoch), pop_idx)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child(idx, pop_copy, g_best, epoch) for idx in pop_idx]
        return pop
