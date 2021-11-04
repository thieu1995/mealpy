#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalIWO(Optimizer):
    """
    Original version of: Invasive Weed Optimization (IWO)
        A novel numerical optimization algorithm inspired from weed colonization
        (https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf)
    Noted:
        + Better to use normal distribution instead of uniform distribution
        + Updating population by sorting both parent population and child population
    """

    def __init__(self, problem, epoch=10000, pop_size=100, seeds=(2, 10), exponent=2, sigma=(0.5, 0.001), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seeds (list): (Min, Max) Number of Seeds
            exponent (int): Variance Reduction Exponent
            sigma (list): (Initial, Final) Value of Standard Deviation
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.seeds = seeds
        self.exponent = exponent
        self.sigma = sigma

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
        # Update Standard Deviation
        sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma[0] - self.sigma[1]) + self.sigma[1]
        pop, best, worst = self.get_special_solutions(pop)
        pop_new = pop.copy()
        for idx in range(0, self.pop_size):
            ratio = (pop[idx][self.ID_FIT][self.ID_TAR] - worst[0][self.ID_FIT][self.ID_TAR]) / \
                    (best[0][self.ID_FIT][self.ID_TAR] - worst[0][self.ID_FIT][self.ID_TAR] + self.EPSILON)
            s = int(np.ceil(self.seeds[0] + (self.seeds[1] - self.seeds[0]) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            pop_local = []
            for j in range(s):
                # Initialize Offspring and Generate Random Location
                pos_new = pop[idx][self.ID_POS] + sigma * np.random.normal(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position_faster(pos_new)
                pop_local.append([pos_new, None])
            pop_local = self.update_fitness_population(mode, pop_local)
            pop_new += pop_local
        pop = self.get_sorted_strim_population(pop_new, self.pop_size)
        return pop




