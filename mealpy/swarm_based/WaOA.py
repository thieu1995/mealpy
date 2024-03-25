#!/usr/bin/env python
# Created by "Thieu" at 23:18, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalWaOA(Optimizer):
    """
    The original version of: Walrus Optimization Algorithm (WaOA)

    Links:
        1. https://www.researchgate.net/publication/364684780_Walrus_Optimization_Algorithm_A_New_Bio-Inspired_Metaheuristic_Algorithm

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Northern Goshawk Optimization (NGO)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Northern Goshawk Optimization (NGO), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WaOA
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
    >>> model = WaOA.OriginalWaOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. (2022). Walrus Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm.
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
        for idx in range(0, self.pop_size):
            # Phase 1: Feeding strategy (exploration)
            kk = self.generator.permutation(self.pop_size)[0]
            if self.compare_target(self.pop[kk].target, self.pop[idx].target, self.problem.minmax):     # Eq. 4
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[kk].solution - self.generator.integers(1, 3) * self.pop[idx].solution)
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[idx].solution - self.pop[kk].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

            # PHASE 2 Exploitation
            LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
            pos_new = self.pop[idx].solution + LB + (UB - self.generator.random() * LB)     # Eq. 7
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
