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

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Osprey Optimization Algorithm (OOA)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Osprey Optimization Algorithm (OOA), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TDO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = TDO.OriginalTDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., Hubálovský, Š., & Trojovský, P. (2022). Tasmanian devil optimization: a new bio-inspired
    optimization algorithm for solving optimization algorithm. IEEE Access, 10, 19599-19620.
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
            # PHASE1: Hunting Feeding
            if self.generator.random() > 0.5:
                # STRATEGY 1: FEEDING BY EATING CARRION (EXPLORATION PHASE)
                # CARRION selection using (3)
                kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                if self.compare_target(self.pop[kk].target, self.pop[idx].target, self.problem.minmax):
                    pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (self.pop[kk].solution - self.generator.integers(1, 3)*self.pop[idx].solution)
                else:
                    pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (self.pop[idx].solution - self.pop[kk].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = agent
            else:
            # STRATEGY 2: FEEDING BY EATING PREY (EXPLOITATION PHASE)
            # stage1: prey selection and attack it
                kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                if self.compare_target(self.pop[kk].target, self.pop[idx].target, self.problem.minmax):
                    pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (self.pop[kk].solution - self.generator.integers(1, 3) * self.pop[idx].solution)
                else:
                    pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (self.pop[idx].solution - self.pop[kk].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = agent

            # stage2: prey chasing
            rr = 0.01 * (1 - epoch/self.epoch)      # Calculating the neighborhood radius using(9)
            pos_new = self.pop[idx].solution + (-rr + 2 * rr * self.generator.random(self.problem.n_dims)) * self.pop[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
