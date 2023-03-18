#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFOX(Optimizer):
    """
    The original version of: Fox Optimizer (FOX)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03533-0
        2. https://www.mathworks.com/matlabcentral/fileexchange/121592-fox-a-fox-inspired-optimization-algorithm

    Notes (parameters):
        1. c1 (float): the probability of jumping (c1 in the paper), default = 0.18
        2. c2 (float): the probability of jumping (c2 in the paper), default = 0.82

    Notes (Algorithm's design):
        1. I don't know how this algorithm get accepted in Applied Intelligence journal
        2. The equation to calculate distance_S_travel value in matlab code is meaningless
        3. The whole point of if else conditions with p > 0.18 is meaningless. The authors just choice the best value
        based on his experiment without explaining it.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FOX import OriginalFOX
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
    >>> model = OriginalFOX(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mohammed, H., & Rashid, T. (2023). FOX: a FOX-inspired optimization algorithm. Applied Intelligence, 53(1), 1030-1050.
    """
    def __init__(self, epoch=10000, pop_size=100, c1=0.18, c2=0.82, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): the probability of jumping (c1 in the paper), default = 0.18
            c2 (float): the probability of jumping (c2 in the paper), default = 0.82
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_float("c1", c1, (-100., 100.))      # c1 in the paper
        self.c2 = self.validator.check_float("c2", c2, (-100., 100.))      # c2 in the paper
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.mint = 10000000

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        aa = 2 * (1 - (1.0 / self.epoch))
        pop_new = []
        for idx in range(0, self.pop_size):

            if np.random.rand() >= 0.5:
                t1 = np.random.rand(self.problem.n_dims)
                sps = self.g_best[self.ID_POS] / t1
                dis = 0.5 * sps * t1
                tt = np.mean(t1)
                t = tt / 2
                jump = 0.5 * 9.81 * t ** 2
                if np.random.rand() > 0.18:
                    pos_new = dis * jump * self.c1
                else:
                    pos_new = dis * jump * self.c2
                if self.mint > tt:
                    self.mint = tt
            else:
                pos_new = self.g_best[self.ID_POS] + np.random.randn(self.problem.n_dims) * (self.mint * aa)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = [pos_new, target]
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_wrapper_population(pop_new)
