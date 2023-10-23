#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevVCS(Optimizer):
    """
    The developed version: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes:
        + In Immune response process, updates the whole position instead of updating each variable in position

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, VCS
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
    >>> model = VCS.DevVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.5, sigma: float = 1.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Percentage of the number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.sigma = self.validator.check_float("sigma", sigma, (0, 5.0))
        self.n_best = int(self.lamda * self.pop_size)
        self.set_parameters(["epoch", "pop_size", "lamda", "sigma"])
        self.sort_flag = True

    def calculate_xmean__(self, pop):
        """
        Calculate the mean position of list of solutions (population)

        Args:
            pop (list): List of solutions (population)

        Returns:
            list: Mean position
        """
        ## Calculate the weighted mean of the λ best individuals by
        pop = self.get_sorted_population(pop, self.problem.minmax)
        pos_list = [agent.solution for agent in pop[:self.n_best]]
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(np.prod(range(1, self.n_best + 1)))
        weight = np.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        x_mean = weight * np.sum(pos_list, axis=0)
        return x_mean

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Viruses diffusion
        pop = []
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch + 1) / self.epoch) * (self.pop[idx].solution - self.g_best.solution)
            gauss = self.generator.normal(self.generator.normal(self.g_best.solution, np.abs(sigma)))
            pos_new = gauss + self.generator.uniform() * self.g_best.solution - self.generator.uniform() * self.pop[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        ## Host cells infection
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - epoch / self.epoch)
        pop = []
        for idx in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        ## Calculate the weighted mean of the λ best individuals by
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
        ## Immune response
        pop = []
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            temp = self.pop[id1].solution - (self.pop[id2].solution - self.pop[idx].solution) * self.generator.uniform()
            condition = self.generator.random(self.problem.n_dims) < pr
            pos_new = np.where(condition, self.pop[idx].solution, temp)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)


class OriginalVCS(DevVCS):
    """
    The original version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, VCS
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
    >>> model = VCS.OriginalVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm
    for optimization: Virus colony search. Advances in Engineering Software, 92, pp.65-88.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.5, sigma: float = 1.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(epoch, pop_size, lamda, sigma, **kwargs)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.clip(solution, self.problem.lb, self.problem.ub)
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Viruses diffusion
        pop = []
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch) / self.epoch) * (self.pop[idx].solution - self.g_best.solution)
            gauss = np.array([self.generator.normal(self.g_best.solution[j], np.abs(sigma[j])) for j in range(0, self.problem.n_dims)])
            pos_new = gauss + self.generator.uniform() * self.g_best.solution - self.generator.uniform() * self.pop[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        ## Host cells infection
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - epoch / self.epoch)
        pop = []
        for idx in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        ## Immune response
        pop = []
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            pos_new = self.pop[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() > pr:
                    id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new[j] = self.pop[id1].solution[j] - (self.pop[id2].solution[j] - self.pop[idx].solution[j]) * self.generator.uniform()
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
