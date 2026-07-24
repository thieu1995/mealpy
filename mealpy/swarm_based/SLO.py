#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.opt_info import OptInfo


class OriginalSLO(Optimizer):
    """
    The original version of: Sea Lion Optimization Algorithm (SLO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.


    .. caution::
       + There are some unclear equations and parameters in the original paper https://doi.org/10.14569/IJACSA.2019.0100548

    References
    ~~~~~~~~~~
    1. Masadeh, R., Mahafzah, B.A. and Sharieh, A., 2019. Sea lion optimization algorithm. Sea, 10(5), p.388.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SLO
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
    >>> model = SLO.OriginalSLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Sea Lion Optimization Algorithm", year=2019, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2 - 2 * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)  # In the paper this is not clear how to calculate
        pop_new = []
        for idx in range(0, self.pop_size):
            if SP_leader < 0.25:
                if c < 1:
                    pos_new = self.g_best.solution - c * np.abs(2 * self.generator.random() * self.g_best.solution - self.pop[idx].solution)
                else:
                    ri = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))  # random index
                    pos_new = self.pop[ri].solution - c * np.abs(2 * self.generator.random() * self.pop[ri].solution - self.pop[idx].solution)
            else:
                pos_new = np.abs(self.g_best.solution - self.pop[idx].solution) * np.cos(2 * np.pi * self.generator.uniform(-1, 1)) + self.g_best.solution
            # In the paper doesn't check also doesn't update old solution at this point
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class ModifiedSLO(Optimizer):
    """
    Our modified version: Modified Sea Lion Optimization (M-SLO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.

    Note
    ----
    + Local best idea in PSO is inspired
    + Levy-flight technique is used
    + Shrink encircling idea is used

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SLO
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
    >>> model = SLO.ModifiedSLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Modified Sea Lion Optimization (Dev)", difficulty="medium", kind="developed")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_pos = self.problem.lb + self.problem.ub - solution
        local_pos = self.correct_solution(local_pos)
        return Agent(solution=solution, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        target = self.get_target(agent.solution)
        local_target = self.get_target(agent.local_solution)
        if self.compare_target(target, local_target, self.problem.minmax):
            t1 = agent.local_solution.copy()
            t2 = agent.solution.copy()
            agent.update(solution=t1, target=local_target, local_solution=t2, local_target=target)
        else:
            t1 = agent.solution.copy()
            t2 = agent.local_solution.copy()
            agent.update(solution=t1, target=target, local_solution=t2, local_target=local_target)
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        c = 2. - 2. * epoch / self.epoch
        if c > 1:
            pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
        else:
            pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
        SP_leader = self.generator.uniform(0, 1)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if SP_leader >= 0.6:
                pos_new = np.cos(2 * np.pi * self.generator.normal(0, 1)) * np.abs(self.g_best.solution - self.pop[idx].solution) + self.g_best.solution
            else:
                if self.generator.uniform() < pa:
                    dist1 = self.generator.uniform() * np.abs(2 * self.g_best.solution - self.pop[idx].solution)
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.01, size=self.problem.n_dims, case=-1) * dist1 * c
                    pos_new = (self.pop[idx].solution - np.sqrt(epoch) * np.sign(self.generator.random() - 0.5)) * levy_step * self.generator.uniform(self.problem.lb, self.problem.ub)
                else:
                    rand_SL = self.pop[self.generator.integers(0, self.pop_size)].local_solution
                    rand_SL = 2 * self.g_best.solution - rand_SL
                    pos_new = rand_SL - c * np.abs(self.generator.uniform() * rand_SL - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(agent.solution)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx].copy()
                if self.compare_target(pop_new[idx].target, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].local_solution = pop_new[idx].solution.copy()
                    self.pop[idx].local_target = pop_new[idx].target.copy()


class ImprovedSLO(ModifiedSLO):
    """
    The original version: Improved Sea Lion Optimization (ImprovedSLO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    c1 : float
        Local coefficient same as PSO, in range (0.0, 5.0). Default is 1.2.
    c2 : float
        Global coefficient same as PSO, in range (0.0, 5.0). Default is 1.2.

    References
    ~~~~~~~~~~
    1. Nguyen, Binh Minh, Trung Tran, Thieu Nguyen, and Giang Nguyen. "An improved sea lion optimization
       for workload elasticity prediction with neural networks." International Journal of Computational
       Intelligence Systems 15, no. 1 (2022): 90. https://doi.org/10.1007/s44196-022-00156-8

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SLO
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
    >>> model = SLO.ImprovedSLO(epoch=1000, pop_size=50, c1=1.2, c2=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Improved Sea Lion Optimization", year=2022, difficulty="medium", kind="variant")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 1.2, c2: float = 1.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): Local coefficient same as PSO, default = 1.2
            c2 (float): Global coefficient same as PSO, default = 1.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2. - 2. * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if SP_leader < 0.5:
                if c < 1:  # Exploitation improved by historical movement + global best affect
                    # pos_new = g_best.solution - c * np.abs(2 * rand() * g_best.solution - pop[i].solution)
                    dif1 = np.abs(2 * self.generator.random() * self.g_best.solution - self.pop[idx].solution)
                    dif2 = np.abs(2 * self.generator.random() * self.pop[idx].local_solution - self.pop[idx].solution)
                    pos_new = self.c1 * self.generator.random() * (self.pop[idx].solution - dif1) + \
                              self.c2 * self.generator.random() * (self.pop[idx].solution - dif2)
                else:  # Exploration improved by opposition-based learning
                    # Create a new solution by equation below
                    # Then create an opposition solution of above solution
                    # Compare both of them and keep the good one (Searching at both direction)
                    pos_new = self.g_best.solution + c * self.generator.normal(0, 1, self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    target_new = self.get_target(pos_new)
                    pos_new_oppo = self.problem.lb + self.problem.ub - self.g_best.solution + self.generator.random() * (self.g_best.solution - pos_new)
                    pos_new_oppo = self.correct_solution(pos_new_oppo)
                    target_new_oppo = self.get_target(pos_new_oppo)
                    if self.compare_target(target_new_oppo, target_new, self.problem.minmax):
                        pos_new = pos_new_oppo
            else:  # Exploitation
                pos_new = self.g_best.solution + np.cos(2 * np.pi * self.generator.uniform(-1, 1)) * np.abs(self.g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx].copy()
                if self.compare_target(pop_new[idx].target, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].local_solution = pop_new[idx].solution.copy()
                    self.pop[idx].local_target = pop_new[idx].target.copy()
