#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTDO(Optimizer):
    """
    The original version of: Tasmanian Devil Optimization (TDO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/111380-tasmanian-devil-optimization-tdo
        2. https://ieeexplore.ieee.org/abstract/document/9714388

    Notes (Plagiarism):
        0. This is really disgusting, because the source code for this algorithm is almost exactly the same as the source code of Osprey Optimization Algorithm
        1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
        Osprey Optimization Algorithm (OOA), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. Check the matlab code of all above algorithms
        2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
        4. Check the results of benchmark functions in the papers, they are mostly make up results

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.TDO import OriginalTDO
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
    >>> model = OriginalTDO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., Hubálovský, Š., & Trojovský, P. (2022). Tasmanian devil optimization: a new bio-inspired
    optimization algorithm for solving optimization algorithm. IEEE Access, 10, 19599-19620.
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
            # PHASE1: Hunting Feeding
            if np.random.rand() > 0.5:
                # STRATEGY 1: FEEDING BY EATING CARRION (EXPLORATION PHASE)
                # CARRION selection using (3)
                kk = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                if self.compare_agent(self.pop[kk], self.pop[idx]):
                    pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.pop[kk][self.ID_POS] - np.random.randint(1, 3)*self.pop[idx][self.ID_POS])
                else:
                    pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.pop[idx][self.ID_POS] - self.pop[kk][self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                tar_new = self.get_target_wrapper(pos_new)
                if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                    self.pop[idx] = [pos_new, tar_new]
            else:
            # STRATEGY 2: FEEDING BY EATING PREY (EXPLOITATION PHASE)
            # stage1: prey selection and attack it
                kk = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                if self.compare_agent(self.pop[kk], self.pop[idx]):
                    pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.pop[kk][self.ID_POS] - np.random.randint(1, 3) * self.pop[idx][self.ID_POS])
                else:
                    pos_new = self.pop[idx][self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.pop[idx][self.ID_POS] - self.pop[kk][self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                tar_new = self.get_target_wrapper(pos_new)
                if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                    self.pop[idx] = [pos_new, tar_new]

            # stage2: prey chasing
            rr = 0.01 * (1 - (epoch+1)/self.epoch)      # Calculating the neighborhood radius using(9)
            pos_new = self.pop[idx][self.ID_POS] + (-rr + 2 * rr * np.random.rand(self.problem.n_dims)) * self.pop[idx][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]
