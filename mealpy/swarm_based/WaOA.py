#!/usr/bin/env python
# Created by "Thieu" at 23:18, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWaOA(Optimizer):
    """
    The original version of: Walrus Optimization Algorithm (WaOA)

    Links:
        1. https://www.researchgate.net/publication/364684780_Walrus_Optimization_Algorithm_A_New_Bio-Inspired_Metaheuristic_Algorithm

    Notes (Plagiarism):
        0. This is really disgusting, because the source code for this algorithm is exactly the same as the source code of Northern Goshawk Optimization (NGO)
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Northern Goshawk Optimization (NGO),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.WaOA import OriginalWaOA
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
    >>> model = OriginalWaOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. (2022). Walrus Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            # Phase 1: Feeding strategy (exploration)
            kk = np.random.permutation(self.pop_size)[0]
            if self.compare_agent(self.pop[kk], self.pop[idx]):     # Eq. 4
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[kk][self.ID_POS] - np.random.randint(1, 3) * self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[idx][self.ID_POS] - self.pop[kk][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # PHASE 2 Exploitation
            LB, UB = self.problem.lb / (epoch+1), self.problem.ub / (epoch + 1)
            pos_new = self.pop[idx][self.ID_POS] + LB + (UB - np.random.rand() * LB)     # Eq. 7
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
