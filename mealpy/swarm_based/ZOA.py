#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalZOA(Optimizer):
    """
    The original version of: Zebra Optimization Algorithm (ZOA)

    Links:
        1. https://ieeexplore.ieee.org/document/9768820
        2. https://www.mathworks.com/matlabcentral/fileexchange/122942-zebra-optimization-algorithm-zoa

    Notes:
		1. It's concerning that the author seems to be reusing the same algorithms with minor variations.
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO).
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ZOA import OriginalZOA
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
    >>> model = OriginalZOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra optimization algorithm: A new bio-inspired
    optimization algorithm for solving optimization algorithm. IEEE Access, 10, 49445-49473.
    """
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # PHASE1: Foraging Behaviour
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = np.round(1 + np.random.rand())
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.g_best[self.ID_POS] - r1 * self.pop[idx][self.ID_POS])   # Eq. 3
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        # PHASE2: defense strategies against predators
        kk = np.random.permutation(self.pop_size)[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.rand() < 0.5:
                # S1: the lion attacks the zebra and thus the zebra chooses an escape strategy
                r2 = 0.1
                pos_new = self.pop[idx][self.ID_POS] + r2 * (2 + np.random.rand(self.problem.n_dims) - 1) * (1 - (epoch+1)/self.epoch)*self.pop[idx][self.ID_POS]
            else:
                # S2: other predators attack the zebra and the zebra will choose the offensive strategy
                r2 = np.random.randint(1, 3)
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.pop[kk][self.ID_POS] - r2 * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
