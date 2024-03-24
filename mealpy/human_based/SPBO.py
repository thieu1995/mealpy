#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSPBO(Optimizer):
    """
    The original version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. This algorithm is a weak algorithm in solving several problems
        2. It also consumes too much time because of ndim * pop_size updating times.

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0965997820301484
       2. https://www.mathworks.com/matlabcentral/fileexchange/80991-student-psycology-based-optimization-spbo-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SPBO
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
    >>> model = SPBO.OriginalSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Das, B., Mukherjee, V., & Das, D. (2020). Student psychology based optimization algorithm: A new population based
    optimization algorithm for solving optimization problems. Advances in Engineering software, 146, 102804.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
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
        for jdx in range(0, self.problem.n_dims):
            idx_best = self.get_index_best(self.pop, self.problem.minmax)
            mid = self.generator.integers(1, self.pop_size-1)
            x_mean = np.mean([agent.solution for agent in self.pop], axis=0)
            pop_new = []
            for idx in range(0, self.pop_size):
                if idx == idx_best:
                    k = self.generator.choice([1, 2])
                    j = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                    new_pos = self.g_best.solution + (-1)**k * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[j].solution)
                elif idx < mid:
                    ## Good Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.g_best.solution + self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
                    else:
                        new_pos = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution) + \
                            self.generator.random() * (self.pop[idx].solution - x_mean)
                else:
                    ## Average Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * (x_mean - self.pop[idx].solution)
                    else:
                        new_pos = self.problem.generate_solution()
                new_pos = self.correct_solution(new_pos)
                agent = self.generate_empty_agent(new_pos)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(new_pos)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace uniform random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SPBO
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
    >>> model = SPBO.DevSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        good = int(self.pop_size / 3)
        average = 2 * int(self.pop_size / 3)
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                j = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_pos = self.g_best.solution + self.generator.normal(0, 1, self.problem.n_dims) * (self.g_best.solution - self.pop[j].solution)
            elif idx < good:    ## Good Student
                if self.generator.random() > self.generator.random():
                    new_pos = self.g_best.solution + self.generator.normal(0, 1, self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
                else:
                    ra = self.generator.random(self.problem.n_dims)
                    new_pos = self.pop[idx].solution + ra * (self.g_best.solution - self.pop[idx].solution) + (1 - ra) * (self.pop[idx].solution - x_mean)
            elif idx < average:  ## Average Student
                new_pos = self.pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * (x_mean - self.pop[idx].solution)
            else:
                new_pos = self.problem.generate_solution()
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(new_pos)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
