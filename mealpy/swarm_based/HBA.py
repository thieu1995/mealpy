#!/usr/bin/env python
# Created by "Thieu" at 17:55, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHBA(Optimizer):
    """
    The original version of: Honey Badger Algorithm (HBA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0378475421002901
        2. https://www.mathworks.com/matlabcentral/fileexchange/98204-honey-badger-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HBA
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
    >>> model = HBA.OriginalHBA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F. A., Houssein, E. H., Hussain, K., Mabrouk, M. S., & Al-Atabany, W. (2022). Honey Badger Algorithm: New metaheuristic
    algorithm for solving optimization problems. Mathematics and Computers in Simulation, 192, 84-110.
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

    def initialize_variables(self):
        self.beta = 6       # the ability of HB to get the food  Eq.(4)
        self.C = 2          # constant in Eq. (3)

    def get_intensity__(self, best, pop):
        size = len(pop)
        di = np.zeros(size)
        si = np.zeros(size)
        for idx in range(0, size):
            di[idx] = (np.linalg.norm(pop[idx].solution - best.solution) + self.EPSILON) ** 2
            if idx == size - 1:
                si[idx] = (np.linalg.norm(pop[idx].solution - self.pop[0].solution) + self.EPSILON) ** 2
            else:
                si[idx] = (np.linalg.norm(pop[idx].solution - self.pop[idx + 1].solution) + self.EPSILON) ** 2
        r2 = self.generator.random(size)
        return r2 * si / (4 * np.pi * di)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        tt = self.epoch
        alpha= self.C * np.exp(-tt/self.epoch)   # density factor in Eq. (3)
        I = self.get_intensity__(self.g_best, self.pop)        # intensity in Eq. (2)
        pop_new = []
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            F = self.generator.choice([1, -1])
            di = self.g_best.solution - self.pop[idx].solution
            r3 = self.generator.random(self.problem.n_dims)
            r4 = self.generator.random(self.problem.n_dims)
            r5 = self.generator.random(self.problem.n_dims)
            r6 = self.generator.random(self.problem.n_dims)
            r7 = self.generator.random(self.problem.n_dims)
            temp1 = self.g_best.solution + F * self.beta * I[idx] * self.g_best.solution + F*r3*alpha*di*np.abs(np.cos(2*np.pi*r4) * (1 - np.cos(2*np.pi*r5)))
            temp2 = self.g_best.solution + F * r7 * alpha * di
            pos_new = np.where(r6 < 0.5, temp1, temp2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
