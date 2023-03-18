#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalPOA(Optimizer):
    """
    The original version of: Pelican Optimization Algorithm (POA)

    Links:
        1. https://www.mdpi.com/1424-8220/22/3/855
        2. https://www.mathworks.com/matlabcentral/fileexchange/106680-pelican-optimization-algorithm-a-novel-nature-inspired

    Notes (Plagiarism):
        0. This is really disgusting, because the source code for this algorithm is exactly the same as the source code for Northern Goshawk Optimization (NGO)
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.POA import OriginalPOA
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
    >>> model = OriginalPOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. (2022). Pelican optimization algorithm: A novel nature-inspired
    algorithm for engineering applications. Sensors, 22(3), 855.
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
        ## UPDATE location of food
        kk = np.random.permutation(self.pop_size)[0]
        for idx in range(0, self.pop_size):
            # PHASE 1: Moving towards prey (exploration phase)
            if self.compare_agent(self.pop[kk], self.pop[idx]):     # Eq. 4
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[kk][self.ID_POS] - np.random.randint(1, 3) * self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[idx][self.ID_POS] - self.pop[kk][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            # PHASE 2: Winging on the water surface (exploitation phase)        # Eq. 6
            pos_new = self.pop[idx][self.ID_POS] + 0.2 * (1 - (epoch+1)/self.epoch) *(2*np.random.rand(self.problem.n_dims) - 1) * self.pop[idx][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
