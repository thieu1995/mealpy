#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMRFO(Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MRFO
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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, somersault_range: float = 2.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.somersault_range = self.validator.check_float("somersault_range", somersault_range, [1.0, 5.0])
        self.set_parameters(["epoch", "pop_size", "somersault_range"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if self.generator.random() < 0.5:
                r1 = self.generator.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if (epoch + 1) / self.epoch < self.generator.random():
                    x_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = x_rand + self.generator.uniform() * (x_rand - self.pop[idx].solution) + \
                               beta * (x_rand - self.pop[idx].solution)
                    else:
                        x_t1 = x_rand + self.generator.uniform() * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                               beta * (x_rand - self.pop[idx].solution)
                else:
                    if idx == 0:
                        x_t1 = self.g_best.solution + self.generator.uniform() * (self.g_best.solution - self.pop[idx].solution) + \
                               beta * (self.g_best.solution - self.pop[idx].solution)
                    else:
                        x_t1 = self.g_best.solution + self.generator.uniform() * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                               beta * (self.g_best.solution - self.pop[idx].solution)
            # Chain foraging (Eq. 1,2)
            else:
                r = self.generator.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = self.pop[idx].solution + r * (self.g_best.solution - self.pop[idx].solution) + \
                           alpha * (self.g_best.solution - self.pop[idx].solution)
                else:
                    x_t1 = self.pop[idx].solution + r * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                           alpha * (self.g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, g_best = self.update_global_best_agent(self.pop, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.pop[idx].solution + self.somersault_range * \
                   (self.generator.uniform() * g_best.solution - self.generator.uniform() * self.pop[idx].solution)
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class WMQIMRFO(Optimizer):
    """
    The original version of: Wavelet Mutation and Quadratic Interpolation MRFO (WMQIMRFO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.108071

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2
        + pm (float): (0.0, 1.0), probability mutation, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MRFO
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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0, pm=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] G. Hu, M. Li, X. Wang et al., An enhanced manta ray foraging optimization algorithm for shape optimization of
    complex CCG-Ball curves, Knowledge-Based Systems (2022), doi: https://doi.org/10.1016/j.knosys.2021.108071.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, somersault_range: float = 2.0, pm: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
            pm (float): probability mutation, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.somersault_range = self.validator.check_float("somersault_range", somersault_range, [1.0, 5.0])
        self.pm = self.validator.check_float("pm", pm, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "somersault_range", "pm"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            x_t = self.pop[idx].solution
            x_t1 = self.pop[idx-1].solution

            ## Morlet wavelet mutation strategy
            ## Goal is to jump out of local optimum --> Performed in exploration stage
            s_constant = 2.0
            a = s_constant * (1.0 / s_constant) ** (1.0 - epoch / self.epoch)
            theta = self.generator.uniform(-2.5 * a, 2.5 * a)
            x = theta / a
            w = np.exp(-x ** 2 / 2) * np.cos(5 * x)
            xichma = 1.0 / np.sqrt(a) * w

            if self.generator.random() < 0.5:  # Control parameter adjustment
                coef = np.log(1 + (np.e - 1.) * epoch / self.epoch)  # Eq. 3.11

                r1 = self.generator.uniform()
                beta = 2 * np.exp(r1 * (self.epoch - epoch) / self.epoch) * np.sin(2 * np.pi * r1)

                if coef < self.generator.random():     # Cyclone foraging
                    x_rand = self.problem.generate_solution()
                    if self.generator.random() < self.pm:      # Morlet wavelet mutation
                        if idx == 0:
                            pos_new = x_rand + self.generator.random() * (x_rand - x_t) + beta * (x_rand - x_t)
                        else:
                            pos_new = x_rand + self.generator.random() * (x_t1 - x_t) + beta * (x_rand - x_t)
                    else:
                        conditions = self.generator.uniform(0, 1, self.problem.n_dims) > 0.5
                        if idx == 0:
                            t1 = x_rand + self.generator.random(self.problem.n_dims) * (x_rand - x_t) + beta * (x_rand - x_t) + xichma * (self.problem.ub - x_t)
                            t2 = x_rand + self.generator.random(self.problem.n_dims) * (x_rand - x_t) + beta * (x_rand - x_t) + xichma * (x_t - self.problem.lb)
                        else:
                            t1 = x_rand + self.generator.random(self.problem.n_dims) * (x_t1 - x_t) + beta * (x_rand - x_t) + xichma * (self.problem.ub - x_t)
                            t2 = x_rand + self.generator.random(self.problem.n_dims) * (x_t1 - x_t) + beta * (x_rand - x_t) + xichma * (x_t - self.problem.lb)
                        pos_new = np.where(conditions, t1, t2)
                else:
                    if idx == 0:
                        pos_new = self.g_best.solution + self.generator.random() * (self.g_best.solution - x_t) + beta * (self.g_best.solution - x_t)
                    else:
                        pos_new = self.g_best.solution + self.generator.random() * (x_t1 - x_t) + beta * (self.g_best.solution - x_t)
            else:   # Chain foraging (Eq. 1,2)
                r = self.generator.random()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    pos_new = x_t + r * (self.g_best.solution - x_t) + alpha * (self.g_best.solution - x_t)
                else:
                    pos_new = x_t + r * (x_t1 - x_t) + alpha * (self.g_best.solution - x_t)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, g_best = self.update_global_best_agent(self.pop, save=False)

        # Somersault foraging   (Eq. 8)
        pop_child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.somersault_range * \
                   (self.generator.random() * g_best.solution - self.generator.random() * self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)
        self.pop, g_best = self.update_global_best_agent(self.pop, save=False)

        # Quadratic Interpolation
        pop_new = []
        for idx in range(0, self.pop_size):
            idx2, idx3 = idx + 1, idx + 2
            if idx == self.pop_size-2:
                idx2, idx3 = idx + 1, 0
            if idx == self.pop_size-1:
                idx2, idx3 = 0, 1
            f1, f2, f3 = self.pop[idx].target.fitness, self.pop[idx2].target.fitness, self.pop[idx3].target.fitness
            x1, x2, x3 = self.pop[idx].solution, self.pop[idx2].solution, self.pop[idx3].solution
            a = f1 / ((x1 - x2) * (x1 - x3)) + f2 / ((x2 - x1) * (x2 - x3)) + f3 / ((x3 - x1) * (x3 - x2))
            gx = ((x3 ** 2 - x2 ** 2) * f1 + (x1 ** 2 - x3 ** 2) * f2 + (x2 ** 2 - x1 ** 2) * f3) / (2 * ((x3 - x2) * f1 + (x1 - x3) * f2 + (x2 - x1) * f3))
            pos_new = np.where(a > 0, gx, x1)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
