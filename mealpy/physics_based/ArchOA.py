#!/usr/bin/env python
# Created by "Thieu" at 16:10, 08/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalArchOA(Optimizer):
    """
    The original version of: Archimedes Optimization Algorithm (ArchOA)

    Links:
        1. https://doi.org/10.1007/s10489-020-01893-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (int): factor, default belongs to [1, 2]
        + c2 (int): factor, Default belongs to [2, 4, 6]
        + c3 (int): factor, Default belongs to [1, 2]
        + c4 (float): factor, Default belongs to [0.5, 1]
        + acc_max (float): acceleration max, Default 0.9
        + acc_min (float): acceleration min, Default 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ArchOA
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
    >>> model = ArchOA.OriginalArchOA(epoch=1000, pop_size=50, c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Hussain, K., Houssein, E.H., Mabrouk, M.S. and Al-Atabany, W., 2021. Archimedes optimization
    algorithm: a new metaheuristic algorithm for solving optimization problems. Applied Intelligence, 51(3), pp.1531-1551.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2, c2: float = 6,
                 c3: float = 2, c4: float = 0.5, acc_max: float = 0.9, acc_min: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): factor, default belongs [1, 2]
            c2 (float): factor, Default belongs [2, 4, 6]
            c3 (float): factor, Default belongs [1, 2]
            c4 (float): factor, Default belongs [0.5, 1]
            acc_max (float): acceleration max, Default 0.9
            acc_min (float): acceleration min, Default 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, [1, 3])
        self.c2 = self.validator.check_float("c2", c2, [2, 6])
        self.c3 = self.validator.check_float("c3", c3, [1, 3])
        self.c4 = self.validator.check_float("c4", c4, (0, 1.0))
        self.acc_max = self.validator.check_float("acc_max", acc_max, (0.3, 1.0))
        self.acc_min = self.validator.check_float("acc_min", acc_min, (0, 0.3))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "c3", "c4", "acc_max", "acc_min"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        den = self.generator.uniform(self.problem.lb, self.problem.ub)          # Density
        vol = self.generator.uniform(self.problem.lb, self.problem.ub)          # Volume
        acc = self.problem.lb + self.generator.uniform(self.problem.lb, self.problem.ub) * (self.problem.ub - self.problem.lb)  # Acceleration
        return Agent(solution=solution, den=den, vol=vol, acc=acc)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Transfer operator Eq. 8
        tf = np.exp(epoch / self.epoch)
        ## Density decreasing factor Eq. 9
        ddf = np.exp(1. - epoch / self.epoch) - epoch / self.epoch
        list_acc = []
        ## Calculate new density, volume and acceleration
        for idx in range(0, self.pop_size):
            # Update density and volume of each object using Eq. 7
            new_den = self.pop[idx].den + self.generator.uniform() * (self.g_best.den - self.pop[idx].den)
            new_vol = self.pop[idx].vol + self.generator.uniform() * (self.g_best.vol - self.pop[idx].vol)
            # Exploration phase
            if tf <= 0.5:
                # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                id_rand = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_acc = (self.pop[id_rand].den + self.pop[id_rand].vol * self.pop[id_rand].acc) / (new_den * new_vol)
            else:
                new_acc = (self.g_best.den + self.g_best.vol * self.g_best.acc) / (new_den * new_vol)
            list_acc.append(new_acc)
            self.pop[idx].den = new_den
            self.pop[idx].vol = new_vol
        min_acc = np.min(list_acc)
        max_acc = np.max(list_acc)
        ## Normalize acceleration using Eq. 12
        for idx in range(0, self.pop_size):
            self.pop[idx].acc = self.acc_max * (list_acc[idx] - min_acc) / (max_acc - min_acc) + self.acc_min
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if tf <= 0.5:  # update position using Eq. 13
                id_rand = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                pos_new = self.pop[idx].solution + self.c1 * self.generator.uniform() * \
                          self.pop[idx].acc * ddf * (self.pop[id_rand].solution - self.pop[idx].solution)
            else:
                p = 2 * self.generator.random() - self.c4
                f = 1 if p <= 0.5 else -1
                t = self.c3 * tf
                pos_new = self.g_best.solution + f * self.c2 * self.generator.random() * self.pop[idx].acc * \
                          ddf * (t * self.g_best.solution - self.pop[idx].solution)
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
