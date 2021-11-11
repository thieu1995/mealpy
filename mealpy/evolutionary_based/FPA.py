#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:34, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            if np.random.uniform() < self.p_s:
                levy = self.get_levy_flight_step(multiplier=0.001, case=-1)
                pos_new = self.pop[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * \
                          levy * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            else:
                id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.pop[id1][self.ID_POS] - self.pop[id2][self.ID_POS])
            pos_new = self.amend_position_random(pos_new)
            pop.append([pos_new, None])
        self.pop = self.update_fitness_population(pop)

