#!/usr/bin/env python                                                                                   #
# ------------------------------------------------------------------------------------------------------#
# Created by "Thieu Nguyen" at 10:55, 02/12/2019                                                        #
#                                                                                                       #
#       Email:      nguyenthieu2102@gmail.com                                                           #
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  #
#       Github:     https://github.com/thieu1995                                                        #
#-------------------------------------------------------------------------------------------------------#

import numpy as np
from mealpy.optimizer import Optimizer


class BaseSHO(Optimizer):
    """
    My modified version: Spotted Hyena Optimizer (SHO)
        (Spotted hyena optimizer: A novel bio-inspired based metaheuristic technique for engineering applications)
    Link:
        https://doi.org/10.1016/j.advengsoft.2017.05.014
    """

    def __init__(self, problem, epoch=10000, pop_size=100, h=5, M=(0.5, 1), N_tried=10, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h (float): default = 5, coefficient linearly decreased from 5 to 0
            M (list): default = [0.5, 1], random vector in [0.5, 1]
            N_tried (int): default = 10,
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.h = h
        self.M = M
        self.N_tried = N_tried

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = []
        for idx in range(0, self.pop_size):
            h = 5 - (epoch + 1.0) * (5 / self.epoch)
            rd1 = np.random.uniform(0, 1, self.problem.n_dims)
            rd2 = np.random.uniform(0, 1, self.problem.n_dims)
            B = 2 * rd1
            E = 2 * h * rd2 - h

            if np.random.rand() < 0.5:
                D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] - np.dot(E, D_h)
            else:
                N = 0
                for i in range(0, self.N_tried):
                    pos_new = self.g_best[self.ID_POS] + np.random.uniform(self.M[0], self.M[1]) * \
                              np.random.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if self.compare_agent(self.g_best, [pos_new, fit_new]):
                        N += 1
                        nfe_epoch += 1
                        break
                    N += 1
                circle_list = []
                idx_list = np.random.choice(range(0, self.pop_size), N, replace=False)
                for j in range(0, N):
                    D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx_list[j]][self.ID_POS])
                    p_k = self.g_best[self.ID_POS] - np.dot(E, D_h)
                    circle_list.append(p_k)
                pos_new = np.mean(np.array(circle_list))
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        nfe_epoch += self.pop_size
        self.nfe_per_epoch = nfe_epoch
