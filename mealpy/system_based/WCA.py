#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWCA(Optimizer):
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

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + nsr (int): [4, 10], Number of rivers + sea (sea = 1), default = 4
        + wc (float): [1.0, 3.0], Weighting coefficient (C in the paper), default = 2
        + dmax (float): [1e-6], fixed parameter, Evaporation condition constant, default=1e-6

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WCA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = WCA.OriginalWCA(epoch=1000, pop_size=50, nsr = 4, wc = 2.0, dmax = 1e-6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Eskandar, H., Sadollah, A., Bahreininejad, A. and Hamdi, M., 2012. Water cycle algorithm–A novel metaheuristic
    optimization method for solving constrained engineering optimization problems. Computers & Structures, 110, pp.151-166.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, nsr: int = 4, wc: float = 2.0, dmax: float = 1e-6, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            nsr (int): Number of rivers + sea (sea = 1), default = 4
            wc (float): Weighting coefficient (C in the paper), default = 2.0
            dmax (float): Evaporation condition constant, default=1e-6
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.nsr = self.validator.check_int("nsr", nsr, [2, int(self.pop_size/2)])
        self.wc = self.validator.check_float("wc", wc, (1.0, 3.0))
        self.dmax = self.validator.check_float("dmax", dmax, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "nsr", "wc", "dmax"])
        self.sort_flag = True

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
        self.g_best = self.pop[0]
        self.ecc = self.dmax  # Evaporation condition constant - variable
        n_stream = self.pop_size - self.nsr
        g_best = self.pop[0].copy()  # Global best solution (sea)
        self.pop_best = self.pop[:self.nsr]  # Including sea and river (1st solution is sea)
        self.pop_stream = self.pop[self.nsr:] # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([agent.target.fitness for agent in self.pop_best])
        num_child_in_river_list = np.round(np.abs(cost_river_list / np.sum(cost_river_list)) * n_stream).astype(int)
        if np.sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - np.sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr - 1):
            streams[i] = []
            idx_list = self.generator.choice(list(set(range(0, n_stream)) - set(idx_already_selected)), num_child_in_river_list[i], replace=False).tolist()
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
                pos_new = stream.solution + self.generator.uniform() * self.wc * (self.pop_best[idx].solution - stream.solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                stream_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    stream_new[-1].target = self.get_target(pos_new)
            stream_new = self.update_target_for_population(stream_new)
            self.streams[idx] = stream_new
            stream_best = self.get_best_agent(stream_new, self.problem.minmax)
            if self.compare_target(stream_best.target, self.pop_best[idx].target, self.problem.minmax):
                self.pop_best[idx] = stream_best.copy()
            # Update river
            pos_new = self.pop_best[idx].solution + self.generator.uniform() * self.wc * (self.g_best.solution - self.pop_best[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop_best[idx].target, self.problem.minmax):
                self.pop_best[idx] = agent
        # Evaporation
        for idx in range(1, self.nsr):
            distance = np.sqrt(np.sum((self.g_best.solution - self.pop_best[idx].solution) ** 2))
            if distance < self.ecc or self.generator.random() < 0.1:
                child = self.generate_agent()
                pop_current_best = self.get_sorted_population(self.streams[idx] + [child], self.problem.minmax)
                self.pop_best[idx] = pop_current_best.pop(0)
                self.streams[idx] = pop_current_best
        self.pop = self.pop_best.copy()
        for idx, stream_list in self.streams.items():
            self.pop += stream_list
        # Reduce the ecc
        self.ecc = self.ecc - self.ecc / self.epoch
