#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevGCO(Optimizer):
    """
    The developed version: Germinal Center Optimization (GCO)

    Notes:
        + The global best solution and 2 random solutions are used instead of randomizing 3 solutions

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GCO
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
    >>> model = GCO.DevGCO(epoch=1000, pop_size=50, cr = 0.7, wf = 1.25)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, cr: float = 0.7, wf: float = 1.25, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.wf = self.validator.check_float("wf", wf, (0, 3.0))
        self.set_parameters(["epoch", "pop_size", "cr", "wf"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_list_cell_counter = np.ones(self.pop_size)  # CEll Counter
        self.dyn_list_life_signal = 70 * np.ones(self.pop_size)  # 70% to duplicate, and 30% to die  # LIfe-Signal

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Dark-zone process    (can be parallelization)
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1
            # Mutate process
            r1, r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = self.g_best.solution + self.wf * (self.pop[r2].solution - self.pop[r1].solution)
            condition = self.generator.random(self.problem.n_dims) < self.cr
            pos_new = np.where(condition, pos_new, self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.dyn_list_cell_counter[idx] += 10
                self.pop[idx] = pop_new[idx].copy()
        ## Light-zone process   (no needs parallelization)
        for idx in range(0, self.pop_size):
            self.dyn_list_cell_counter[idx] = 10
            fit_list = np.array([agent.target.fitness for agent in self.pop])
            fit_max = np.max(fit_list)
            fit_min = np.min(fit_list)
            self.dyn_list_cell_counter[idx] += 10 * (self.pop[idx].target.fitness - fit_max) / (fit_min - fit_max + self.EPSILON)


class OriginalGCO(DevGCO):
    """
    The original version of: Germinal Center Optimization (GCO)

    Links:
        1. https://doi.org/10.2991/ijcis.2018.25905179
        2. https://www.atlantis-press.com/journals/ijcis/25905179/view

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GCO
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
    >>> model = GCO.OriginalGCO(epoch=1000, pop_size=50, cr = 0.7, wf = 1.25)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Villaseñor, C., Arana-Daniel, N., Alanis, A.Y., López-Franco, C. and Hernandez-Vargas, E.A., 2018.
    Germinal center optimization algorithm. International Journal of Computational Intelligence Systems, 12(1), p.13.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, cr: float = 0.7, wf: float = 1.25, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(epoch, pop_size, cr, wf, **kwargs)
        self.is_parallelizable = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Dark-zone process (can't be parallelization)
        for idx in range(0, self.pop_size):
            if self.generator.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            elif self.dyn_list_cell_counter[idx] > 1:
                self.dyn_list_cell_counter[idx] -= 1
            # Mutate process
            p = self.dyn_list_cell_counter / np.sum(self.dyn_list_cell_counter)
            r1, r2, r3 = self.generator.choice(list(set(range(0, self.pop_size))), 3, replace=False, p=p)
            pos_new = self.pop[r1].solution + self.wf * (self.pop[r2].solution - self.pop[r3].solution)
            condition = self.generator.random(self.problem.n_dims) < self.cr
            pos_new = np.where(condition, pos_new, self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # for each pos_new, generate the fitness
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.dyn_list_life_signal[idx] += 10
        ## Light-zone process   (no needs parallelization)
        self.dyn_list_life_signal -= 10
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        fit_max = np.max(fit_list)
        fit_min = np.min(fit_list)
        fit = (fit_list - fit_max) / (fit_min - fit_max)
        if self.problem.minmax != 'min':
            fit = 1 - fit
        self.dyn_list_life_signal += 10 * fit
