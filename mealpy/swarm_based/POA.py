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

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Northern Goshawk Optimization (NGO)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, POA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = POA.OriginalPOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. (2022). Pelican optimization algorithm: A novel nature-inspired
    algorithm for engineering applications. Sensors, 22(3), 855.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## UPDATE location of food
        kk = self.generator.permutation(self.pop_size)[0]
        for idx in range(0, self.pop_size):
            # PHASE 1: Moving towards prey (exploration phase)
            if self.compare_target(self.pop[kk].target, self.pop[idx].target, self.problem.minmax):     # Eq. 4
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[kk].solution - self.generator.integers(1, 3) * self.pop[idx].solution)
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[idx].solution - self.pop[kk].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

            # PHASE 2: Winging on the water surface (exploitation phase)        # Eq. 6
            pos_new = self.pop[idx].solution + 0.2 * (1 - epoch/self.epoch) *(2*self.generator.random(self.problem.n_dims) - 1) * self.pop[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
