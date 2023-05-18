#!/usr/bin/env python
# Created by "Thieu" at 00:47, 16/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHBO(Optimizer):
    """
    The original version of: Heap-based optimizer (HBO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420305261#!
        2. https://github.com/qamar-askari/HBO/blob/master/HBO.m

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + degree (int): [2, 4], the degree level in Corporate Rank Hierarchy (CRH), default=2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.HBO import OriginalHBO
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
    >>> degree = 3
    >>> model = OriginalHBO(epoch, pop_size, degree)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Askari, Q., Saeed, M., & Younas, I. (2020). Heap-based optimizer inspired by corporate rank hierarchy
    for global optimization. Expert Systems with Applications, 161, 113702.
    """

    def __init__(self, epoch=10000, pop_size=100, degree=2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            degree (int): the degree level in Corporate Rank Hierarchy (CRH), default=2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.degree = self.validator.check_int("degree", degree, [2, 10])
        self.set_parameters(["epoch", "pop_size", "degree"])
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.cycles = np.floor(self.epoch / 25)
        self.it_per_cycle = self.epoch / self.cycles
        self.qtr_cycle = self.it_per_cycle / 4

    def colleagues_limits_generator__(self, pop_size, degree=3):
        friend_limits = np.zeros((pop_size, 2))
        for c in range(pop_size - 1, -1, -1):
            hi = int(np.ceil((np.log10(c * degree - c + 1) / np.log10(degree)))) - 1
            lowerLim = ((degree * degree ** (hi - 1) - 1) / (degree - 1) + 1)
            upperLim = (degree * degree ** hi - 1) / (degree - 1)
            friend_limits[c, 0] = lowerLim if lowerLim <= pop_size else pop_size
            friend_limits[c, 1] = upperLim if upperLim <= pop_size else pop_size
        return friend_limits.astype(int)

    def heapifying__(self, pop, degree=3):
        pop_size = len(pop)
        heap = []
        for c in range(pop_size):
            heap.append([pop[c][self.ID_TAR], c])
            # Heapifying
            t = c
            while t > 0:
                parent_id = int(np.floor((t + 1)/degree) - 1)
                if self.compare_agent(pop[parent_id], pop[t]):
                    break
                else:
                    heap[t], heap[parent_id] = heap[parent_id], heap[t]
                t = parent_id
        return heap

    def before_main_loop(self):
        self.heap = self.heapifying__(self.pop, self.degree)
        self.friend_limits = self.colleagues_limits_generator__(self.pop_size, self.degree)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        gama = (np.mod(epoch+1, self.it_per_cycle) +1) / self.qtr_cycle
        gama = np.abs(2 - gama)
        p1 = 1 - (epoch+1) / self.epoch
        p2 = p1 + (1 - p1) / 2
        for c in range(self.pop_size-1, 0, -1):

            if c == 0: # Dealing with root
                continue
            else:
                parent_id = int(np.floor((c+1)/self.degree) - 1)
                cur_agent = self.pop[self.heap[c][1]].copy()         #Sol to be updated
                par_agent = self.pop[self.heap[parent_id][1]]           #Sol to be updated with reference to

                # Sol to be updated with reference to
                if self.friend_limits[c, 0] < self.friend_limits[c, 1]+1:
                    friend_idx = self.friend_limits[c, 0]
                else:
                    friend_idx = np.random.choice(list(set(range(self.friend_limits[c, 0], self.friend_limits[c, 1])) - {c}))
                fri_agent = self.pop[self.heap[friend_idx][1]]

                #Position Updating
                rr = np.random.rand(self.problem.n_dims)
                rn = (2 * np.random.rand(self.problem.n_dims) - 1)

                for jdx in range(self.problem.n_dims):
                    if rr[jdx] < p1:
                        continue
                    elif rr[jdx] < p2:
                        cur_agent[self.ID_POS][jdx] = par_agent[self.ID_POS][jdx] + rn[jdx] * gama * \
                                                      np.abs(par_agent[self.ID_POS][jdx] - cur_agent[self.ID_POS][jdx])
                    else:
                        if self.compare_agent([None, self.heap[friend_idx][0]], [None, self.heap[c][0]]):
                            cur_agent[self.ID_POS][jdx] = fri_agent[self.ID_POS][jdx] + rn[jdx] * \
                                                          gama * np.abs(fri_agent[self.ID_POS][jdx] - cur_agent[self.ID_POS][jdx])
                        else:
                            cur_agent[self.ID_POS][jdx] += rn[jdx] * gama * np.abs(fri_agent[self.ID_POS][jdx] - cur_agent[self.ID_POS][jdx])
                cur_agent[self.ID_POS] = self.amend_position(cur_agent[self.ID_POS], self.problem.lb, self.problem.ub)
                cur_agent[self.ID_TAR] = self.get_target_wrapper(cur_agent[self.ID_POS])

                if self.compare_agent(cur_agent, [None, self.heap[c][0]]):
                    self.pop[self.heap[c][1]] = cur_agent
                    self.heap[c][0] = cur_agent[self.ID_TAR].copy()

            # Heapifying
            t = c
            while t > 1:
                parent_id = int((t + 1) / self.degree)
                if self.compare_agent([None, self.heap[parent_id][0]], [None, self.heap[t][0]]):
                    break
                else:
                    self.heap[t], self.heap[parent_id] = self.heap[parent_id], self.heap[t]
                t = parent_id
