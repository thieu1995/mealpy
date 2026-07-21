#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalFFO(Optimizer):
    """
    The original version of: Fennec Fox Optimization (FFO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    gamma : float
        Light Absorption Coefficient, in range (0.0, 1.0). Default is 0.001.
    beta_base : float
        Attraction Coefficient Base Value, in range (0.0, 3.0). Default is 2.0.
    alpha : float
        Mutation Coefficient, in range (0.0, 1.0). Default is 0.2.
    alpha_damp : float
        Mutation Coefficient Damp Rate, in range (0.0, 1.0). Default is 0.99.
    delta : float
        Mutation Step Size, in range (0.0, 1.0). Default is 0.05.
    exponent : int
        Exponent (m in the paper), in range [2, 4]. Default is 2.


    .. error::
       1. This is somewhat concerning, as there appears to be a high degree of similarity between
          the source code for this algorithm and the Pelican Optimization Algorithm (POA).
       2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA),
          Coati Optimization Algorithm (CoatiOA), Siberian Tiger Optimization (STO), Language Education
          Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
          Pelican Optimization Algorithm (POA), Three-periods optimization algorithm (TPOA), Teamwork optimization
          algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO),
          Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
       3. It may be useful to compare the Matlab code of this algorithm with those of the
          similar algorithms to ensure its accuracy and completeness.
       4. The article may share some similarities with previous work by the same authors, further
          investigation may be warranted to verify the benchmark results reported in the papers
          and ensure their reliability and accuracy.

    References
    ~~~~~~~~~~
    1. Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Fennec Fox Optimization: A New Nature-Inspired
       Optimization Algorithm. IEEE Access, 10, 84417-84443. https://doi.org/10.1109/ACCESS.2022.3197745

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FFO
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
    >>> model = FFO.OriginalFFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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
            # PHASE 1: THE DIGGING TO LOOK FOR PREY UNDER THE SAND (EXPLOITATION)
            rr = 0.2 * (1 - epoch / self.epoch) * self.pop[idx].solution
            pos_new = self.pop[idx].solution + (2 * self.generator.random() * 1) * rr
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

            # PHASE 2: ESCAPE STRATEGY FROM THE PREDATORS’ ATTACK (EXPLORATION)
            kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_target(self.pop[kk].target, self.pop[idx].target, self.problem.minmax):
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[kk].solution - self.generator.integers(1, 3) * self.pop[idx].solution)
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[idx].solution - self.pop[kk].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
