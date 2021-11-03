#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:41, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseEHO(Optimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)
        (Elephant Herding Optimization )
    Link:
        https://doi.org/10.1109/ISCBI.2015.8
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.5, beta=0.5, n_clans=5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan
            beta (float): a factor that determines the influence of the x_center
            n_clans (int): the number of clans
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size + n_clans
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.n_clans = n_clans
        self.n_individuals = int(self.pop_size / self.n_clans)

    def _creat_population(self, mode=None):
        pop = []
        for i in range(0, self.n_clans):
            group = self.create_population(mode, self.n_individuals)
            pop.append(group)
        return pop

    def _sort_clan_find_center_pop_best(self, pop=None):
        centers = []
        pop_best = []
        for i in range(0, self.n_clans):
            pop[i] = self.get_sorted_strim_population(pop[i], self.n_individuals)
            center = np.mean(np.array([item[self.ID_POS] for item in pop[i]]), axis=0)
            centers.append(center)
            pop_best.append(pop[i][0].copy())
        return pop, centers, pop_best

    def _composite_population(self, pop):
        pop_new = []
        for i in range(0, self.n_clans):
            pop_new += pop[i]
        return pop_new

    def solve(self, mode='sequential'):
        self.termination_start()
        pop = self._creat_population()
        pop, centers, pop_best = self._sort_clan_find_center_pop_best(pop)
        _, g_best = self.get_global_best_solution(pop_best)
        self.history.save_initial_best(g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            # Clan updating operator
            for i in range(0, self.pop_size):
                clan_idx = int(i / self.n_individuals)
                pos_clan_idx = int(i % self.n_individuals)

                if pos_clan_idx == 0:  # The best in clan, because all clans are sorted based on fitness
                    pos_new = self.beta * centers[clan_idx]
                else:
                    pos_new = pop[clan_idx][pos_clan_idx][self.ID_POS] + self.alpha * np.random.uniform() * \
                              (pop[clan_idx][0][self.ID_POS] - pop[clan_idx][pos_clan_idx][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                pop[clan_idx][pos_clan_idx][self.ID_POS] = pos_new

            # Update fitness value
            for i in range(0, self.n_clans):
                pop[i] = self.update_fitness_population(mode, pop[i])

            # Separating operator
            for i in range(0, self.n_clans):
                pop[i] = self.get_sorted_strim_population(pop[i], self.n_individuals)
                pop[i][-1] = self.create_solution()

            ## Update the global best
            pop, centers, pop_best = self._sort_clan_find_center_pop_best(pop)
            _, g_best = self.update_global_best_solution(pop_best)

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(self._composite_population(pop))
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

