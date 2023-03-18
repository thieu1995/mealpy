#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalOOA(Optimizer):
    """
    The original version of: Osprey Optimization Algorithm (OOA)

    Links:
        1. https://www.frontiersin.org/articles/10.3389/fmech.2022.1126450/full
        2. https://www.mathworks.com/matlabcentral/fileexchange/124555-osprey-optimization-algorithm

    Notes (Plagiarism):
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.OOA import OriginalOOA
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
    >>> model = OriginalOOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. Osprey Optimization Algorithm: A new bio-inspired metaheuristic algorithm
    for solving engineering optimization problems. Frontiers in Mechanical Engineering, 8, 136.
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
        self.support_parallel_modes = False
        self.sort_flag = False

    def get_indexes_better__(self, pop, idx):
        fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        if self.problem.minmax == "min":
            idxs = np.where(fits < pop[idx][self.ID_TAR][self.ID_FIT])
        else:
            idxs = np.where(fits > pop[idx][self.ID_TAR][self.ID_FIT])
        return idxs[0]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            # Phase 1: : POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            idxs = self.get_indexes_better__(self.pop, idx)
            if len(idxs) == 0:
                sf = self.g_best
            else:
                if np.random.rand() < 0.5:
                    sf = self.g_best
                else:
                    kk = np.random.permutation(idxs)[0]
                    sf = self.pop[kk]
            r1 = np.random.randint(1, 3)
            pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, 1) * (sf[self.ID_POS] - r1 * self.pop[idx][self.ID_POS])     # Eq. 5
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            pos_new = self.pop[idx][self.ID_POS] + self.problem.lb + np.random.rand() * (self.problem.ub - self.problem.lb)     # Eq. 7
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
