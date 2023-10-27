#!/usr/bin/env python
# Created by "Thieu" at 11:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalSSDO(Optimizer):
    """
    The original version of: Social Ski-Driver Optimization (SSDO)

    Links:
       1. https://doi.org/10.1007/s00521-019-04159-z
       2. https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SSDO
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
    >>> model = SSDO.OriginalSSDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tharwat, A. and Gabel, T., 2020. Parameters optimization of support vector machines for imbalanced
    data using social ski driver algorithm. Neural Computing and Applications, 32(11), pp.6925-6938.
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

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_local = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=pos_local)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0
        ## Calculate the mean of the best three solutions in each dimension. Eq 9
        _, pop_best3, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pos_mean = np.mean(np.array([agent.solution for agent in pop_best3]))
        pop_new = [agent.copy() for agent in self.pop]
        # Updating velocity vectors
        r1 = self.generator.uniform()  # r1, r2 is a random number in [0,1]
        r2 = self.generator.uniform()
        for i in range(0, self.pop_size):
            if r2 <= 0.5:  ## Use Sine function to move
                vel_new = c * np.sin(r1) * (self.pop[i].local_solution - self.pop[i].solution) + (2-c)*np.sin(r1) * (pos_mean - self.pop[i].solution)
            else:  ## Use Cosine function to move
                vel_new = c * np.cos(r1) * (self.pop[i].local_solution - self.pop[i].solution) + (2-c)*np.cos(r1) * (pos_mean - self.pop[i].solution)
            pop_new[i].velocity = vel_new
        ## Reproduction
        for idx in range(0, self.pop_size):
            pos_new = self.generator.normal(0, 1, self.problem.n_dims) * pop_new[idx].solution + self.generator.random() * pop_new[idx].velocity
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.local_solution = self.pop[idx].solution.copy()
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, pop_new[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
