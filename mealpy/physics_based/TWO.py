#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalTWO(Optimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TWO
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
    >>> model = TWO.OriginalTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
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
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop = self.update_weight__(self.pop)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with solution

        Args:
            solution (np.ndarray): The solution
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return Agent(solution=solution, weight=0.0)

    def update_weight__(self, teams):
        list_fits = np.array([agent.target.fitness for agent in teams])
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = self.generator.uniform(0.0, 1.0, self.pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights/np.sum(list_weights) + 0.1
        for idx in range(self.pop_size):
            teams[idx].weight = list_weights[idx]
        return teams

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[jdx].weight:
                    force = max(self.pop[idx].weight * self.muy_s, self.pop[jdx].weight * self.muy_s)
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[jdx].solution - self.pop[idx].solution
                    acceleration = resultant_force * g / (self.pop[idx].weight * self.muy_k)
                    delta_x = 0.5 * acceleration + np.power(self.alpha, epoch) * self.beta * \
                              (self.problem.ub - self.problem.lb) * self.generator.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[jdx] + self.generator.standard_normal() / epoch * (self.g_best.solution[jdx] - pos_new[jdx])
                        if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.lb[jdx]:
                            pos_new[jdx] = self.problem.lb[jdx]
                        if pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.problem.ub[jdx]
            pos_new = self.correct_solution(pos_new)
            pop_new[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[idx], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop = self.update_weight__(self.pop)


class OppoTWO(OriginalTWO):
    """
    The opossition-based learning version: Tug of War Optimization (OTWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TWO
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
    >>> model = TWO.OppoTWO(epoch=1000, pop_size=50)
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
        super().__init__(epoch, pop_size, **kwargs)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        list_idx = self.generator.choice(range(0, self.pop_size), int(self.pop_size/2), replace=False)
        pop_temp = [self.pop[list_idx[idx]] for idx in range(0, int(self.pop_size/2))]
        pop_oppo = []
        for idx in range(len(pop_temp)):
            pos_opposite = self.problem.ub + self.problem.lb - pop_temp[idx].solution
            pos_opposite = self.correct_solution(pos_opposite)
            agent = self.generate_empty_agent(pos_opposite)
            pop_oppo.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[-1].target = self.get_target(pos_opposite)
        pop_oppo = self.update_target_for_population(pop_oppo)
        self.pop = pop_temp + pop_oppo
        self.pop = self.update_weight__(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Apply force of others solution on each individual solution
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[jdx].weight:
                    force = max(self.pop[idx].weight * self.muy_s, self.pop[jdx].weight * self.muy_s)
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[jdx].solution - self.pop[idx].solution
                    acceleration = resultant_force * g / (self.pop[idx].weight * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch) * self.beta * \
                              (self.problem.ub - self.problem.lb) * self.generator.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            self.pop[idx].solution = pos_new
        ## Amend solution and update fitness value
        for idx in range(self.pop_size):
            pos_new = self.g_best.solution + self.generator.normal(0, 1, self.problem.n_dims) / (epoch) * (self.g_best.solution - pop_new[idx].solution)
            conditions = np.logical_or(pop_new[idx].solution < self.problem.lb, pop_new[idx].solution > self.problem.ub)
            conditions = np.logical_and(conditions, self.generator.random(self.problem.n_dims) < 0.5)
            pos_new = np.where(conditions, pos_new, self.pop[idx].solution)
            pop_new[idx].solution = self.correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[idx], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        ## Opposition-based here
        pop = []
        for idx in range(self.pop_size):
            C_op = self.generate_opposition_solution(self.pop[idx], self.g_best)
            pos_new = self.correct_solution(C_op)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        self.pop = self.update_weight__(self.pop)


class LevyTWO(OriginalTWO):
    """
    The Levy-flight version of: Tug of War Optimization (LevyTWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TWO
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
    >>> model = TWO.LevyTWO(epoch=1000, pop_size=50)
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
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for kdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[kdx].weight:
                    force = max(self.pop[idx].weight * self.muy_s, self.pop[kdx].weight * self.muy_s)
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[kdx].solution - self.pop[idx].solution
                    acceleration = resultant_force * g / (self.pop[idx].weight * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch) * self.beta * \
                              (self.problem.ub - self.problem.lb) * self.generator.normal(0, 1, self.problem.n_dims)
                    pos_new +=delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[jdx] + self.generator.standard_normal() / epoch * (self.g_best.solution[jdx] - pos_new[jdx])
                        if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.lb[jdx]:
                            pos_new[jdx] = self.problem.lb[jdx]
                        if pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.problem.ub[jdx]
            pop_new[idx].solution = self.correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[idx], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        ### Apply levy-flight here
        for idx in range(self.pop_size):
            ## Chance for each agent to update using levy is 50%
            if self.generator.random() < 0.5:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.01, size=self.problem.n_dims, case=-1)
                pos_new = pop_new[idx].solution + levy_step
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, pop_new[idx].target, self.problem.minmax):
                    pop_new[idx] = agent
        self.pop = self.update_weight__(pop_new)


class EnhancedTWO(OppoTWO, LevyTWO):
    """
    The original version of: Enhenced Tug of War Optimization (ETWO)

    Links:
        1. https://doi.org/10.1016/j.procs.2020.03.063

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TWO
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
    >>> model = TWO.EnhancedTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Hoang, B., Nguyen, G. and Nguyen, B.M., 2020. A new workload prediction model using
    extreme learning machine and enhanced tug of war optimization. Procedia Computer Science, 170, pp.362-369.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        pop_oppo = self.pop.copy()
        for idx in range(self.pop_size):
            pos_opposite = self.problem.ub + self.problem.lb - self.pop[idx].solution
            pos_new = self.correct_solution(pos_opposite)
            pop_oppo[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[idx].target = self.get_target(pos_new)
        pop_oppo = self.update_target_for_population(pop_oppo)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_oppo, self.pop_size, self.problem.minmax)
        self.pop = self.update_weight__(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for kdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[kdx].weight:
                    force = max(self.pop[idx].weight * self.muy_s, self.pop[kdx].weight * self.muy_s)
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[kdx].solution - self.pop[idx].solution
                    acceleration = resultant_force * g / (self.pop[idx].weight * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch) * self.beta * \
                              (self.problem.ub - self.problem.lb) * self.generator.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[jdx] + self.generator.standard_normal() / epoch * (self.g_best.solution[jdx] - pos_new[jdx])
                        if pos_new[jdx] < self.problem.lb[jdx] or pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.lb[jdx]:
                            pos_new[jdx] = self.problem.lb[jdx]
                        if pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.problem.ub[jdx]
            pop_new[idx].solution = self.correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[idx], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        for idx in range(self.pop_size):
            C_op = self.generate_opposition_solution(pop_new[idx], self.g_best)
            pos_new = self.correct_solution(C_op)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, pop_new[idx].target, self.problem.minmax):
                pop_new[idx] = agent
            else:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=1.0, size=self.problem.n_dims, case=-1)
                pos_new = pop_new[idx].solution + 1.0 / np.sqrt(epoch) * levy_step
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, pop_new[idx].target, self.problem.minmax):
                    pop_new[idx] = agent
        self.pop = self.update_weight__(pop_new)
