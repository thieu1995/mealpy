#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:55, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, rand
from numpy import array,  sqrt, abs, sum, round
from mealpy.root import Root
from copy import deepcopy


class BaseWCA(Root):
    """
    The original version of: Water Cycle Algorithm (WCA)

    Noted: The idea are:
        + 1 sea is global best solution
        + a few river which are second, third, ...
        + other left are stream (will flow directed to sea or river)
        + The idea is almost the same as ICO algorithm
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 nsr=4, C=2, dmax=1e-6, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.nsr = nsr      # Number of rivers + sea (sea = 1)
        self.C = C
        self.dmax = dmax    # Evaporation condition constant

    def train(self):
        # Initial population
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])

        dmax = self.dmax
        n_stream = self.pop_size - self.nsr
        g_best = deepcopy(pop[0])                           # Global best solution (sea)
        pop_best = deepcopy(pop[:self.nsr])                 # Including sea and river (1st solution is sea)
        pop_stream = deepcopy(pop[self.nsr:])               # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = array([solution[self.ID_FIT] for solution in pop_best])
        num_child_in_river_list = round(abs(cost_river_list / sum(cost_river_list)) * n_stream).astype(int)
        if sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr-1):
            streams[i] = []
            idx_list = choice(list(set(range(0, n_stream)) - set(idx_already_selected)), num_child_in_river_list[i], replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                streams[i].append(pop_stream[idx])
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        streams[self.nsr - 1] = []
        for idx in idx_last:
            streams[self.nsr - 1].append(pop_stream[idx])

        # Main Loop
        for epoch in range(self.epoch):

            # Update stream and river
            for idx, stream_list in streams.items():
                # Update stream
                for idx_stream, stream in enumerate(stream_list):
                    pos_new = stream[self.ID_POS] + uniform() * self.C * (pop_best[idx][self.ID_POS] - stream[self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    streams[idx][idx_stream] = [pos_new, fit_new]
                    if fit_new < pop_best[idx][self.ID_FIT]:
                        pop_best[idx] = [pos_new, fit_new]
                        if fit_new < g_best[self.ID_FIT]:
                            g_best = [pos_new, fit_new]
                # Update river
                pos_new = pop_best[idx][self.ID_POS] + uniform() * self.C * (g_best[self.ID_POS] - pop_best[idx][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pop_best[idx] = [pos_new, fit_new]
                if fit_new < g_best[self.ID_FIT]:
                    g_best = [pos_new, fit_new]

            # Evaporation
            for i in range(1, self.nsr):
                distance = sqrt(sum((g_best[self.ID_POS] - pop_best[i][self.ID_POS]) ** 2))
                if distance < dmax or rand() < 0.1:
                    child = self.create_solution()
                    pop_current_best = sorted(streams[i] + [child], key=lambda item: item[self.ID_FIT])
                    pop_best[i] = pop_current_best.pop(0)
                    streams[i] = pop_current_best

            # Reduce the dmax
            dmax = dmax - dmax / self.epoch

            ## Update the global best
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
