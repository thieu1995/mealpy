#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:55, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
import time


class BaseWCA(Optimizer):
    """
    The original version of: Water Cycle Algorithm (WCA)
    Link:
        https://doi.org/10.1016/j.compstruc.2012.07.010
    Noted: The idea are:
        + 1 sea is global best solution
        + a few river which are second, third, ...
        + other left are stream (will flow directed to sea or river)
        + The idea is almost the same as ICO algorithm
    """

    def __init__(self, problem, epoch=10000, pop_size=100, nsr=4, C=2, dmax=1e-6, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            nsr (int): Number of rivers + sea (sea = 1), default = 4
            C (int): Coefficient, default = 2
            dmax (float): Evaporation condition constant, default=1e-6
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size

        self.epoch = epoch
        self.pop_size = pop_size
        self.nsr = nsr
        self.C = C
        self.dmax = dmax

    def solve(self, mode='sequential'):
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
            print("WCA is not support parallel process!")
            exit(0)
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history.save_initial_best(g_best)

        dmax = self.dmax
        n_stream = self.pop_size - self.nsr
        g_best = pop[0].copy()  # Global best solution (sea)
        pop_best = pop[:self.nsr].copy()  # Including sea and river (1st solution is sea)
        pop_stream = pop[self.nsr:].copy()  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in pop_best])
        num_child_in_river_list = np.round(abs(cost_river_list / sum(cost_river_list)) * n_stream).astype(int)
        if sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr - 1):
            streams[i] = []
            idx_list = np.random.choice(list(set(range(0, n_stream)) - set(idx_already_selected)), num_child_in_river_list[i], replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                streams[i].append(pop_stream[idx])
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        streams[self.nsr - 1] = []
        for idx in idx_last:
            streams[self.nsr - 1].append(pop_stream[idx])

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Evolve method will be called in child class
            # pop = self.evolve(mode, epoch, pop, g_best)

            # Update stream and river
            for idx, stream_list in streams.items():
                # Update stream
                for idx_stream, stream in enumerate(stream_list):
                    pos_new = stream[self.ID_POS] + np.random.uniform() * self.C * (pop_best[idx][self.ID_POS] - stream[self.ID_POS])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    streams[idx][idx_stream] = [pos_new, fit_new]
                    if self.compare_agent([pos_new, fit_new], pop_best[idx]):
                        pop_best[idx] = [pos_new, fit_new]
                    # if fit_new < pop_best[idx][self.ID_FIT]:
                    #     pop_best[idx] = [pos_new, fit_new]
                        # if fit_new < g_best[self.ID_FIT]:
                        #     g_best = [pos_new, fit_new]
                # Update river
                pos_new = pop_best[idx][self.ID_POS] + np.random.uniform() * self.C * (g_best[self.ID_POS] - pop_best[idx][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop_best[idx] = [pos_new, fit_new]
                # if fit_new < g_best[self.ID_FIT]:
                #     g_best = [pos_new, fit_new]

            # Evaporation
            for i in range(1, self.nsr):
                distance = np.sqrt(sum((g_best[self.ID_POS] - pop_best[i][self.ID_POS]) ** 2))
                if distance < dmax or np.random.rand() < 0.1:
                    child = self.create_solution()
                    pop_current_best = sorted(streams[i] + [child], key=lambda item: item[self.ID_FIT][self.ID_TAR])
                    pop_best[i] = pop_current_best.pop(0)
                    streams[i] = pop_current_best

            # Reduce the dmax
            dmax = dmax - dmax / self.epoch

            # update global best position
            pop, g_best = self.update_global_best_solution(pop)  # We sort the population

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
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


