#!/usr/bin/env python
# Created by "User" at 2024 ----------%
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalChOA(Optimizer):
    """
    The original version of: Chimp Optimization Algorithm (ChOA)

    Links:
        1. https://doi.org/10.1016/j.eswa.2019.113338

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 10000
        + pop_size (int): number of population size, default = 100

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ChOA
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
    >>> model = ChOA.OriginalChOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Khishe, M. and Mosavi, M.R., 2020. Chimp optimization algorithm.
    Expert systems with applications, 149, p.113338.
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # f decreases linearly from 2.5 to 0 over iterations, Eq. (3)
        f = 2.5 - 2.5 * (epoch / self.epoch)
        
        # Get the four best chimps (Attacker, Barrier, Chaser, Driver)
        _, list_best, _ = self.get_special_agents(self.pop, n_best=4, minmax=self.problem.minmax)
        x_attacker = list_best[0].solution
        x_barrier = list_best[1].solution
        x_chaser = list_best[2].solution
        x_driver = list_best[3].solution

        pop_new = []
        for idx in range(self.pop_size):
            # Attacker chimp position update
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            a1 = 2 * f * r1 - f
            c1 = 2 * r2
            d_attacker = np.abs(c1 * x_attacker - self.pop[idx].solution)
            x1 = x_attacker - a1 * d_attacker

            # Barrier chimp position update
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            a2 = 2 * f * r1 - f
            c2 = 2 * r2
            d_barrier = np.abs(c2 * x_barrier - self.pop[idx].solution)
            x2 = x_barrier - a2 * d_barrier

            # Chaser chimp position update
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            a3 = 2 * f * r1 - f
            c3 = 2 * r2
            d_chaser = np.abs(c3 * x_chaser - self.pop[idx].solution)
            x3 = x_chaser - a3 * d_chaser

            # Driver chimp position update
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            a4 = 2 * f * r1 - f
            c4 = 2 * r2
            d_driver = np.abs(c4 * x_driver - self.pop[idx].solution)
            x4 = x_driver - a4 * d_driver

            # Calculate new position as the average of all four leaders, Eq. (7)
            pos_new = (x1 + x2 + x3 + x4) / 4.0
            
            # Correct and check boundaries
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
                
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
