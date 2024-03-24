#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalCoatiOA(Optimizer):
    """
    The original version of: Coati Optimization Algorithm (CoatiOA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/S0950705122011042
        2. https://www.mathworks.com/matlabcentral/fileexchange/116965-coa-coati-optimization-algorithm

    Notes:
        1. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        3. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CoatiOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = CoatiOA.OriginalCoatiOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., Montazeri, Z., Trojovská, E., & Trojovský, P. (2023). Coati Optimization Algorithm: A new
    bio-inspired metaheuristic algorithm for solving optimization problems. Knowledge-Based Systems, 259, 110011.
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
        # Phase1: Hunting and attacking strategy on iguana (Exploration Phase)
        size2 = int(self.pop_size/2)
        for idx in range(0, size2):
            pos_new = self.pop[idx].solution + self.generator.random() * (self.g_best.solution - self.generator.integers(1, 3) * self.pop[idx].solution) # Eq. 4
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

        for idx in range(size2, self.pop_size):
            iguana = self.generate_agent()
            if self.compare_target(iguana.target, self.pop[idx].target, self.problem.minmax):
                pos_new = self.pop[idx].solution + self.generator.random() * (iguana.solution - self.generator.integers(1, 3) * self.pop[idx].solution)  # Eq. 6
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[idx].solution - iguana.solution)  # Eq. 6
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

        # Phase2: The process of escaping from predators (Exploitation Phase)
        for idx in range(0, self.pop_size):
            LO, HI = self.problem.lb / epoch, self.problem.ub / epoch
            pos_new = self.pop[idx].solution + (1 - 2 * self.generator.random()) * (LO + self.generator.random() * (HI - LO))     # Eq. 8
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
