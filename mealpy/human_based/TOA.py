#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTOA(Optimizer):
    """
    The original version of: Teamwork Optimization Algorithm (TOA)

    Links:
        1. https://www.mdpi.com/1424-8220/21/13/4567

    Notes (Plagiarism):
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Pelican Optimization Algorithm (POA), Northern goshawk optimization (NGO),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.TOA import OriginalTOA
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
    >>> model = OriginalTOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., & Trojovský, P. (2021). Teamwork optimization algorithm: A new optimization
    approach for function minimization/maximization. Sensors, 21(13), 4567.
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
            # Stage 1: Supervisor guidance
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - np.random.randint(1, 3) * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # Stage 2: Information sharing
            idxs = self.get_indexes_better__(self.pop, idx)
            if len(idxs) == 0:
                sf = self.g_best
            else:
                sf_pos = np.array([self.pop[jdx][self.ID_POS] for jdx in idxs])
                sf_pos = self.amend_position(np.mean(sf_pos, axis=0), self.problem.lb, self.problem.ub)
                sf_tar = self.get_target_wrapper(sf_pos)
                sf = [sf_pos, sf_tar]
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (sf[self.ID_POS] - np.random.randint(1, 3) * self.pop[idx][self.ID_POS]) * \
                        np.sign(self.pop[idx][self.ID_TAR][self.ID_FIT] - sf[self.ID_TAR][self.ID_FIT])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # Stage 3: Individual activity
            pos_new = self.pop[idx][self.ID_POS] + (-0.01 + np.random.rand() * 0.02) * self.pop[idx][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
