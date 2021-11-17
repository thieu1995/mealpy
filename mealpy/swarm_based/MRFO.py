#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)
        (Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications)
    Link:
        https://doi.org/10.1016/j.engappai.2019.103300
    """

    def __init__(self, problem, epoch=10000, pop_size=100, somersault_range=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (): somersault factor that decides the somersault range of manta rays, default=2
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.somersault_range = somersault_range

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if np.random.rand() < 0.5:
                r1 = np.random.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if (epoch + 1) / self.epoch < np.random.rand():
                    x_rand = np.random.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = x_rand + np.random.uniform() * (x_rand - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = x_rand + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (x_rand - self.pop[idx][self.ID_POS])
                else:
                    if idx == 0:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        x_t1 = self.g_best[self.ID_POS] + np.random.uniform() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                               beta * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            # Chain foraging (Eq. 1,2)
            else:
                r = np.random.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + r * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                           alpha * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position_faster(x_t1)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)
        _, g_best = self.update_global_best_solution(pop_new, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = pop_new[idx][self.ID_POS] + self.somersault_range * \
                   (np.random.uniform() * g_best[self.ID_POS] - np.random.uniform() * pop_new[idx][self.ID_POS])
            pos_new = self.amend_position_faster(x_t1)
            pop_child.append([pos_new, None])
        pop_child = self.update_fitness_population(pop_child)
        self.pop = self.greedy_selection_population(pop_new, pop_child)
