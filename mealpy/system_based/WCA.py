# !/usr/bin/env python
# Created by "Thieu" at 09:55, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseWCA(Optimizer):
    """
    The original version of: Water Cycle Algorithm (WCA)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2012.07.010

    Notes
    ~~~~~
    The ideas are (almost the same as ICO algorithm):
        + 1 sea is global best solution
        + a few river which are second, third, ...
        + other left are stream (will flow directed to sea or river)

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + nsr (int): [4, 10], Number of rivers + sea (sea = 1), default = 4
        + wc (float): [1.0, 3.0], Weighting coefficient (C in the paper), default = 2
        + dmax (float): [1e-6], fixed parameter, Evaporation condition constant, default=1e-6

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.WCA import BaseWCA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> nsr = 4
    >>> wc = 2.0
    >>> dmax = 1e-6
    >>> model = BaseWCA(problem_dict1, epoch, pop_size, nsr, wc, dmax)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Eskandar, H., Sadollah, A., Bahreininejad, A. and Hamdi, M., 2012. Water cycle algorithm–A novel metaheuristic
    optimization method for solving constrained engineering optimization problems. Computers & Structures, 110, pp.151-166.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, nsr=4, wc=2.0, dmax=1e-6, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            nsr (int): Number of rivers + sea (sea = 1), default = 4
            wc (float): Weighting coefficient (C in the paper), default = 2.0
            dmax (float): Evaporation condition constant, default=1e-6
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.nsr = self.validator.check_int("nsr", nsr, [2, int(self.pop_size/2)])
        self.wc = self.validator.check_float("wc", wc, (1.0, 3.0))
        self.dmax = self.validator.check_float("dmax", dmax, (0, 1.0))
        self.streams, self.pop_bset, self.pop_stream = None, None, None
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialization(self):
        pop = self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(pop)  # We sort the population

        self.ecc = self.dmax  # Evaporation condition constant - variable
        n_stream = self.pop_size - self.nsr
        g_best = deepcopy(pop[0])  # Global best solution (sea)
        self.pop_best = deepcopy(pop[:self.nsr])  # Including sea and river (1st solution is sea)
        self.pop_stream = deepcopy(pop[self.nsr:])  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([solution[self.ID_TAR][self.ID_FIT] for solution in self.pop_best])
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
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update stream and river
        for idx, stream_list in self.streams.items():
            # Update stream
            stream_new = []
            for idx_stream, stream in enumerate(stream_list):
                pos_new = stream[self.ID_POS] + np.random.uniform() * self.wc * (self.pop_best[idx][self.ID_POS] - stream[self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                stream_new.append([pos_new, None])
            stream_new = self.update_target_wrapper_population(stream_new)
            stream_new, stream_best = self.get_global_best_solution(stream_new)
            self.streams[idx] = stream_new
            if self.compare_agent(stream_best, self.pop_best[idx]):
                self.pop_best[idx] = deepcopy(stream_best)

            # Update river
            pos_new = self.pop_best[idx][self.ID_POS] + np.random.uniform() * self.wc * (self.g_best[self.ID_POS] - self.pop_best[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop_best[idx]):
                self.pop_best[idx] = [pos_new, target]

        # Evaporation
        for i in range(1, self.nsr):
            distance = np.sqrt(np.sum((self.g_best[self.ID_POS] - self.pop_best[i][self.ID_POS]) ** 2))
            if distance < self.ecc or np.random.rand() < 0.1:
                child = self.create_solution(self.problem.lb, self.problem.ub)
                pop_current_best, _ = self.get_global_best_solution(self.streams[i] + [child])
                self.pop_best[i] = pop_current_best.pop(0)
                self.streams[i] = pop_current_best

        self.pop = deepcopy(self.pop_best)
        for idx, stream_list in self.streams.items():
            self.pop += stream_list

        # Reduce the ecc
        self.ecc = self.ecc - self.ecc / self.epoch
