#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:55, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


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
            pop_size (int): number of population size, default = 100
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

        self.streams, self.pop_bset, self.pop_stream = None, None, None

    def initialization(self):
        pop = self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(pop)  # We sort the population

        self.ecc = self.dmax    # Evaporation condition constant - variable
        n_stream = self.pop_size - self.nsr
        g_best = deepcopy(pop[0])  # Global best solution (sea)
        self.pop_best = deepcopy(pop[:self.nsr])  # Including sea and river (1st solution is sea)
        self.pop_stream = deepcopy(pop[self.nsr:])  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in self.pop_best])
        num_child_in_river_list = np.round(np.abs(cost_river_list / np.sum(cost_river_list)) * n_stream).astype(int)
        if np.sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - np.sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr - 1):
            streams[i] = []
            idx_list = np.random.choice(list(set(range(0, n_stream)) - set(idx_already_selected)), num_child_in_river_list[i], replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                streams[i].append(self.pop_stream[idx])
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        streams[self.nsr - 1] = []
        for idx in idx_last:
            streams[self.nsr - 1].append(self.pop_stream[idx])
        self.streams = streams

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Update stream and river
        for idx, stream_list in self.streams.items():
            # Update stream
            stream_new = []
            for idx_stream, stream in enumerate(stream_list):
                pos_new = stream[self.ID_POS] + np.random.uniform() * self.C * (self.pop_best[idx][self.ID_POS] - stream[self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                stream_new.append([pos_new, None])
            stream_new = self.update_fitness_population(stream_new)
            stream_new, stream_best = self.get_global_best_solution(stream_new)
            self.streams[idx] = stream_new
            if self.compare_agent(stream_best, self.pop_best[idx]):
                self.pop_best[idx] = deepcopy(stream_best)

            # Update river
            pos_new = self.pop_best[idx][self.ID_POS] + np.random.uniform() * self.C * (self.g_best[self.ID_POS] - self.pop_best[idx][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], self.pop_best[idx]):
                self.pop_best[idx] = [pos_new, fit_new]

        # Evaporation
        for i in range(1, self.nsr):
            distance = np.sqrt(np.sum((self.g_best[self.ID_POS] - self.pop_best[i][self.ID_POS]) ** 2))
            if distance < self.ecc or np.random.rand() < 0.1:
                child = self.create_solution()
                pop_current_best, _ = self.get_global_best_solution(self.streams[i] + [child])
                self.pop_best[i] = pop_current_best.pop(0)
                self.streams[i] = pop_current_best

        self.pop = deepcopy(self.pop_best)
        for idx, stream_list in self.streams.items():
            self.pop += stream_list

        # Reduce the ecc
        self.ecc = self.ecc - self.ecc / self.epoch
