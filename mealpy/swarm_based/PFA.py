#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalPFA(Optimizer):
    """
    The original version of: Pathfinder Algorithm (PFA)

    Links:
        1. https://doi.org/10.1016/j.asoc.2019.03.012

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PFA
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
    >>> model = PFA.OriginalPFA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yapici, H. and Cetinkaya, N., 2019. A new meta-heuristic optimizer: Pathfinder algorithm.
    Applied soft computing, 78, pp.545-568.
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
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        alpha, beta = self.generator.uniform(1, 2, 2)
        A = self.generator.uniform(self.problem.lb, self.problem.ub) * np.exp(-2 * epoch / self.epoch)
        t = 1. - epoch * 1.0 / self.epoch
        space = self.problem.ub - self.problem.lb
        ## Update the position of pathfinder and check the bound
        pos_new = self.pop[0].solution + 2 * self.generator.uniform() * (self.g_best.solution - self.pop[0].solution) + A
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)
        pop_new = [agent, ]
        ## Update positions of members, check the bound and calculate new fitness
        for idx in range(1, self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for k in range(1, self.pop_size):
                dist = np.sqrt(np.sum((self.pop[k].solution - self.pop[idx].solution) ** 2)) / self.problem.n_dims
                t2 = alpha * self.generator.uniform() * (self.pop[k].solution - self.pop[idx].solution)
                ## First stabilize the distance
                t3 = self.generator.uniform() * t * (dist / space)
                pos_new += t2 + t3
            ## Second stabilize the population size
            t1 = beta * self.generator.uniform() * (self.g_best.solution - self.pop[idx].solution)
            pos_new = (pos_new + t1) / self.pop_size
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
