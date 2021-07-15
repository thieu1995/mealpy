#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:34, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseFPA(Optimizer):
    """
        The original version of: Flower Pollination Algorithm (FPA)
            (Flower Pollination Algorithm for Global Optimization)
    Link:
        https://doi.org/10.1007/978-3-642-32894-7_27
    """

    def __init__(self, problem: dict, epoch=750, pop_size=100, p_s=0.8):
        """
        Args:
            problem (dict): a dictionary of your problem
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
        """
        super().__init__(problem)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_s = p_s

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        _, g_best = self.get_global_best_solution(pop)  # We don't sort the population
        self.history_list_g_best = [g_best]
        self.history_list_c_best = self.history_list_g_best.copy()

        for epoch in range(self.epoch):
            time_start = time.time()

            for i in range(0, self.pop_size):
                if np.random.uniform() < self.p_s:
                    levy = self.get_step_size_levy_flight(multiplier=0.001, case=-1)
                    pos_new = pop[i][self.ID_POS] + levy * (pop[i][self.ID_POS] - self.history_list_g_best[-1][self.ID_POS])
                else:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    pos_new = pop[i][self.ID_POS] + np.random.uniform() * (pop[id1][self.ID_POS] - pop[id2][self.ID_POS])
                pos_new = self.amend_position_random(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[i] = self.get_better_solution([pos_new, fit_new], pop[i])

                # batch size idea to update the global best
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        self.update_global_best_solution(pop)
                else:
                    if (i + 1) % self.pop_size == 0:
                        self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_start = time.time() - time_start
            self.history_list_epoch_time.append(time_start)
            self.print_epoch(epoch + 1, time_start)
            self.history_list_pop.append(pop.copy())

        ## Additional information for the framework
        self.solution = self.history_list_g_best[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]