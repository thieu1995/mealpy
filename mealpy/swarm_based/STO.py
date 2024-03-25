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

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Osprey Optimization Algorithm (OOA)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Northern Goshawk Optimization (NGO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, STO
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
    >>> model = STO.OriginalSTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., Dehghani, M., & Hanuš, P. (2022). Siberian Tiger Optimization: A New Bio-Inspired
    Metaheuristic Algorithm for Solving Engineering Optimization Problems. IEEE Access, 10, 132396-132431.
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

    def get_indexes_better__(self, pop, idx):
        fits = np.array([agent.target.fitness for agent in self.pop])
        if self.problem.minmax == "min":
            idxs = np.where(fits < pop[idx].target.fitness)
        else:
            idxs = np.where(fits > pop[idx].target.fitness)
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
                if self.generator.random() < 0.5:
                    sf = self.g_best
                else:
                    kk = self.generator.permutation(idxs)[0]
                    sf = self.pop[kk]
            r1 = self.generator.integers(1, 3)
            pos_new = self.pop[idx].solution + self.generator.random() * (sf.solution - r1 * self.pop[idx].solution)     # Eq. 5
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            pos_new = self.pop[idx].solution + self.generator.random() * (self.problem.ub - self.problem.lb) / epoch     # Eq. 7
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
