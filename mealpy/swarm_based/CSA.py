#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 18:37, 28/05/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseCSA(Optimizer):
    """
        The original version of: Cuckoo Search Algorithm (CSA)
            (Cuckoo search via Levy flights)
        Link:
            https://doi.org/10.1109/NABIC.2009.5393690
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_a=0.3, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_a (float): probability a
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_a = p_a
        self.n_cut = int(self.p_a * self.pop_size)
        self.nfe_per_epoch = self.pop_size + self.n_cut
        self.sort_flag = False

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
        if mode != "sequential":
            print("CSA is only support sequential mode!")
            exit(0)

        for i in range(0, self.pop_size):
            ## Generate levy-flight solution
            # pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS], step=0.001, case=2)
            pos_new = pop[i][self.ID_POS] + self.get_levy_flight_step(multiplier=0.001, case=-1)
            levy_step = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = pop[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                      levy_step * (pop[i][self.ID_POS] - g_best[self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            j_idx = np.random.choice(list(set(range(0, self.pop_size)) - {i}))
            if self.compare_agent([pos_new, fit_new], pop[j_idx]):
                pop[j_idx] = [pos_new, fit_new]

        ## Abandoned some worst nests
        pop = self.get_sorted_strim_population(pop, self.pop_size, reverse=True)
        for i in range(0, self.n_cut):
            # pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
            # levy_step = self.get_levy_flight_step(multiplier=0.001, case=-1)
            pos_new = pop[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                      levy_step * (pop[i][self.ID_POS] - g_best[self.ID_POS])
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            pop[i] = [pos_new ,fit_new]
        return pop
