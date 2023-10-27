#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWOA(Optimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.g_best.solution - A * D
                else:
                    # x_rand = pop[self.generator.self.generator.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.problem.generate_solution()
                    D = np.abs(C * x_rand - self.pop[idx].solution)
                    pos_new = x_rand - A * D
            else:
                D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
                pos_new = self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class HI_WOA(Optimizer):
    """
    The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)

    Links:
        1. https://ieenp.explore.ieee.org/document/8900003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + feedback_max (int): maximum iterations of each feedback, default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.HI_WOA(epoch=1000, pop_size=50, feedback_max = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
    In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, feedback_max: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            feedback_max (int): maximum iterations of each feedback, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.feedback_max = self.validator.check_int("feedback_max", feedback_max, [2, 2+int(self.epoch/2)])
        # The maximum of times g_best doesn't change -> need to change half of population
        self.set_parameters(["epoch", "pop_size", "feedback_max"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_changes = int(self.pop_size / 2)
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        pop_new = []
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.g_best.solution - A * D
                else:
                    # x_rand = pop[self.generator.self.generator.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.problem.generate_solution()
                    D = np.abs(C * x_rand - self.pop[idx].solution)
                    pos_new = x_rand - A * D
            else:
                D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
                pos_new = self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        ## Feedback Mechanism
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        if current_best.target.fitness == self.g_best.target.fitness:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = self.generator.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_child = self.generate_population(self.n_changes)
            for idx_counter, idx in enumerate(idx_list):
                self.pop[idx] = pop_child[idx_counter]
