#!/usr/bin/env python
# Created by "Thieu" at 22:46, 26/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalARO(Optimizer):
    """
    The original version of: Artificial Rabbits Optimization (ARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ARO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = ARO.OriginalARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        theta = 2 * (1 - epoch/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # detour foraging strategy
                rand_idx = self.generator.integers(0, self.pop_size)
                pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
                    np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1
            else:       # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx].solution + H * gr * self.pop[idx].solution       # Eq. 13
                pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)


class LARO(Optimizer):
    """
    The improved version of:  Lévy flight, and the selective opposition version of the artificial rabbit algorithm (LARO)

    Links:
        1. https://doi.org/10.3390/sym14112282

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ARO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = ARO.LARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, Y., Huang, L., Zhong, J., & Hu, G. (2022). LARO: Opposition-based learning boosted
    artificial rabbits-inspired optimization algorithm with Lévy flight. Symmetry, 14(11), 2282.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        theta = 2 * (1 - (epoch+1)/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # # detour foraging strategy
                rand_idx = self.generator.integers(0, self.pop_size)
                pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
                    np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1
            else:       # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx].solution + H * gr * self.pop[idx].solution        # Eq. 13
                levy = self.get_levy_flight_step(beta=1.5, multiplier=0.1)
                pos_new = self.pop[idx].solution + R * (levy * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)
        # Selective Opposition (SO) Strategy
        TS = 2 - (2 * epoch / self.epoch)
        for idx in range(0, self.pop_size):
            if self.pop[idx].target.fitness != self.g_best.target.fitness:
                dd = np.abs(self.g_best.solution - self.pop[idx].solution)
                idx_far = np.sign(dd - TS) < 0
                n_df = np.sum(idx_far)
                n_dc = np.sum(np.sign(dd - TS) > 0)
                src = 1 - 6*np.sum(dd**2) / np.dot(dd, (dd**2 - 1))
                if len(dd[idx_far]) == 0:
                    df_lb, df_ub = np.min(dd), np.max(dd)
                else:
                    df_lb, df_ub = np.min(dd[idx_far]), np.max(dd[idx_far])
                if src <= 0 and n_df > n_dc:
                    pos_new = df_lb + df_ub - self.pop[idx].solution
                    pos_new = self.correct_solution(pos_new)
                    target = self.get_target(pos_new)
                    if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                        self.pop[idx].update(solution=pos_new, target=target)


class IARO(Optimizer):
    """
    The improved version of: Improved Artificial Rabbits Optimization (IARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ARO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = ARO.IARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        theta = 2 * (1 - (epoch+1)/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # # detour foraging strategy
                rand_idx = self.generator.integers(0, self.pop_size)
                pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
                    np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1
            else:       # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx].solution + H * gr * self.pop[idx].solution        # Eq. 13
                pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)
