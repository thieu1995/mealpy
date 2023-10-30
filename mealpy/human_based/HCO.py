#!/usr/bin/env python
# Created by "Thieu" at 08:57, 12/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHCO(Optimizer):
    """
    The original version of: Human Conception Optimizer (HCO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/124200-human-conception-optimizer-hco
        2. https://www.nature.com/articles/s41598-022-25031-6

    Notes:
        1. This algorithm shares some similarities with the PSO algorithm (equations)
        2. The implementation of Matlab code is kinda different to the paper

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wfp (float): (0, 1.) - weight factor for probability of fitness selection, default=0.65
        + wfv (float): (0, 1.0) - weight factor for velocity update stage, default=0.05
        + c1 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4
        + c2 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HCO
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
    >>> model = HCO.OriginalHCO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Acharya, D., & Das, D. K. (2022). A novel Human Conception Optimizer for solving optimization problems. Scientific Reports, 12(1), 21631.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, wfp: float = 0.65,
                 wfv: float = 0.05, c1: float = 1.4, c2: float = 1.4, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wfp (float): weight factor for probability of fitness selection, default=0.65
            wfv (float): weight factor for velocity update stage, default=0.05
            c1 (float): acceleration coefficient, same as PSO, default=1.4
            c2 (float): acceleration coefficient, same as PSO, default=1.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wfp = self.validator.check_float("wfp", wfp, [0, 1.0])
        self.wfv = self.validator.check_float("wfv", wfv, [0, 1.0])
        self.c1 = self.validator.check_float("c1", c1, [0., 100.])
        self.c2 = self.validator.check_float("c2", c2, [1., 100.])
        self.set_parameters(["epoch", "pop_size", "wfp", "wfv", "c1", "c2"])
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        pop_op = []
        for idx in range(0, self.pop_size):
            pos_new = self.problem.ub + self.problem.lb - self.pop[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_op.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_op = self.update_target_for_population(pop_op)
            self.pop = self.greedy_selection_population(self.pop, pop_op, self.problem.minmax)
        _, (best,), (worst,) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pfit = (worst.target.fitness - best.target.fitness) * self.wfp + best.target.fitness
        for idx in range(0, self.pop_size):
            if self.compare_fitness(pfit, self.pop[idx].target.fitness, self.problem.minmax):
                while True:
                    agent = self.generate_agent()
                    if self.compare_fitness(agent.target.fitness, pfit, self.problem.minmax):
                        self.pop[idx] = agent
                        break
        self.vec = self.generator.uniform(self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims))
        self.pop_p = [agent.copy() for agent in self.pop]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        lamda = self.generator.random()
        neu = 2
        fits = np.array([agent.target.fitness for agent in self.pop])
        fit_mean = np.mean(fits)
        RR = (self.g_best.target.fitness - fits) ** 2
        rr = (fit_mean - fits) ** 2
        ll = RR - rr
        LL = (self.g_best.target.fitness - fit_mean)
        VV = lamda * (ll / (4 * neu * LL))
        pop_new = []
        for idx in range(0, self.pop_size):
            a1 = self.pop_p[idx].solution - self.pop[idx].solution
            a2 = self.g_best.solution - self.pop[idx].solution
            self.vec[idx] = self.wfv * (VV[idx] + self.vec[idx]) + self.c1 * a1*np.sin(2*np.pi*epoch/self.epoch) + self.c2*a2*np.sin(2*np.pi*epoch/self.epoch)
            pos_new = self.pop[idx].solution + self.vec[idx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx].copy()
                if self.compare_target(pop_new[idx].target, self.pop_p[idx].target, self.problem.minmax):
                    self.pop_p[idx] = pop_new[idx].copy()
