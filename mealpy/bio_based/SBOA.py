#!/usr/bin/env python
# Created by "ozgurk33" on 05/01/2026
# Github: https://github.com/ozgurk33
# --------------------------------------------------%
# Updated by "Thieu" on 16/07/2026
# Github: https://github.com/thieu1995
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSBOA(Optimizer):
    """
    The original version of: Secretary Bird Optimization Algorithm (SBOA)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size (number of trees), default = 100

    Links
    -----
    1. https://doi.org/10.1007/s10462-024-10729-y
    2. https://www.mathworks.com/matlabcentral/fileexchange/164456-secretary-bird-optimization-algorithm-sboa

    References
    ----------
    .. [1] Fu, Y., Liu, D., Chen, J., & He, L. (2024). Secretary bird optimization algorithm: a new
    metaheuristic for solving global optimization problems. Artificial Intelligence Review, 57(5), 123.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, SBOA
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
    >>> model = SBOA.OriginalSBOA(epoch=1000, pop_size=50)
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Convergence Factor (Eq. 9)
        CF = (1.0 - epoch/self.epoch) ** (2.0 * epoch/self.epoch)

        pop_new = []
        # Hunting Strategies (Exploration Phase)
        for idx in range(0, self.pop_size):
            if epoch < self.epoch / 3:
                # Stage 1: Secretary bird search prey (Eq. 4-5)
                r1, r2 = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=2, replace=True)
                R1 = self.generator.random(self.problem.n_dims)
                pos_new = self.pop[idx].solution + (self.pop[r1].solution - self.pop[r2].solution) * R1

            elif epoch < 2 * self.epoch / 3:
                # Stage 2: Secretary bird approaching prey (Eq. 7-8)
                RB = self.generator.normal(0, 1, self.problem.n_dims)
                term = np.exp((epoch / self.epoch)**4)
                pos_new = self.g_best.solution + term * (RB - 0.5) * (self.g_best.solution - self.pop[idx].solution)
                
            else:
                # Stage 3: Secretary bird attacks prey (Eq. 9-10)
                RL = self.get_levy_flight_step(beta=1.5, multiplier=0.5, size=self.problem.n_dims, case=-1)
                pos_new = self.g_best.solution + CF * self.pop[idx].solution * RL
            
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # Update population in parallel mode
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        
        # Escaping Strategies (Exploitation Phase)
        r = self.generator.random()
        k = self.generator.integers(0, self.pop_size)
        x_random_global = self.pop[k].solution
        
        pop_new_escape = []
        for idx in range(0, self.pop_size):
            if r < 0.5:
                # C1: Secretary birds use their environment to hide (Eq. 14)
                RB = self.generator.random(self.problem.n_dims)
                factor = (1 - epoch/self.epoch) ** 2
                pos_new = self.g_best.solution + factor * (2 * RB - 1) * self.pop[idx].solution
            else:
                # C2: Secretary birds fly or run away (Eq. 14, 16)
                K = int(round(1 + self.generator.random()))
                R2 = self.generator.random(self.problem.n_dims)
                pos_new = self.pop[idx].solution + R2 * (x_random_global - K * self.pop[idx].solution)
                
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new_escape.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        
        # Update population after exploitation
        if self.mode in self.AVAILABLE_MODES:
            pop_new_escape = self.update_target_for_population(pop_new_escape)
            self.pop = self.greedy_selection_population(self.pop, pop_new_escape, self.problem.minmax)
