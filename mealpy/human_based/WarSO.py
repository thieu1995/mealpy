#!/usr/bin/env python
# Created by "Thieu" at 17:41, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWarSO(Optimizer):
    """
    The original version of: War Strategy Optimization (WarSO) algorithm

    Links:
       1. https://www.researchgate.net/publication/358806739_War_Strategy_Optimization_Algorithm_A_New_Effective_Metaheuristic_Algorithm_for_Global_Optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + rr (float): [0.1, 0.9], the probability of switching position updating, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WarSO
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
    >>> model = WarSO.OriginalWarSO(epoch=1000, pop_size=50, rr=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
    war strategy optimization algorithm." International Journal of Energy Research (2022).
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, rr: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            rr (float): the probability of switching position updating, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.rr = self.validator.check_float("rr", rr, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "rr"])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        self.wl = 2 * np.ones(self.pop_size)
        self.wg = np.zeros(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_sorted, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
        self.wl = self.wl[indices]
        self.wg = self.wg[indices]
        com = self.generator.permutation(self.pop_size)
        for idx in range(0, self.pop_size):
            r1 = self.generator.random()
            if r1 < self.rr:
                pos_new = 2*r1*(self.g_best.solution - self.pop[com[idx]].solution) + \
                          self.wl[idx]*self.generator.random()*(pop_sorted[idx].solution - self.pop[idx].solution)
            else:
                pos_new = 2*r1*(pop_sorted[idx].solution - self.g_best.solution) + \
                          self.generator.random() * (self.wl[idx] * self.g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.wg[idx] += 1
                self.wl[idx] = 1 * self.wl[idx] * (1 - self.wg[idx] / self.epoch)**2
