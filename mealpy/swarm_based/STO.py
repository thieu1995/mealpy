#!/usr/bin/env python
# Created by "Thieu" at 22:00, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSTO(Optimizer):
    """
    The original version of: Siberian Tiger Optimization (STO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/9989374
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9989374

    Notes (Plagiarism):
        0. This is really disgusting, because the source code for this algorithm is exact the same as the source code for Osprey Optimization Algorithm (OOA)
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
        Northern Goshawk Optimization (NGO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.STO import OriginalSTO
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
    >>> model = OriginalSTO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., Dehghani, M., & Hanuš, P. (2022). Siberian Tiger Optimization: A New Bio-Inspired
    Metaheuristic Algorithm for Solving Engineering Optimization Problems. IEEE Access, 10, 132396-132431.
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
            # PHASE 1: PREY HUNTING
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
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (sf[self.ID_POS] - r1 * self.pop[idx][self.ID_POS])     # Eq. 5
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.problem.ub - self.problem.lb) / (epoch+1)     # Eq. 7
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
