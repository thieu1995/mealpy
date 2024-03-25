#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGWO(Optimizer):
    """
    The original version of: Grey Wolf Optimizer (GWO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2013.12.007
        2. https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
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
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Lewis, A., 2014. Grey wolf optimizer. Advances in engineering software, 69, pp.46-61.
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
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class RW_GWO(Optimizer):
    """
    The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
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
    >>> model = GWO.RW_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S. and Deep, K., 2019. A novel random walk grey wolf optimizer. Swarm and evolutionary computation, 44, pp.101-112.
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
        # linearly decreased from 2 to 0, Eq. 5
        b = 2. - 2. * epoch / self.epoch
        # linearly decreased from 2 to 0
        a = 2. - 2. * epoch / self.epoch
        _, leaders, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)

        ## Random walk here
        leaders_new = []
        for idx in range(0, len(leaders)):
            pos_new = leaders[idx].solution + a * self.generator.standard_cauchy(self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            leaders_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                leaders[idx] = self.get_better_agent(agent, leaders[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            leaders_new = self.update_target_for_population(leaders_new)
            leaders = self.greedy_selection_population(leaders, leaders_new, self.problem.minmax)

        ## Update other wolfs
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq. 3 and 4
            miu1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            c1 = 2 * self.generator.random(self.problem.n_dims)
            c2 = 2 * self.generator.random(self.problem.n_dims)
            c3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = leaders[0].solution - miu1 * np.abs(c1 * self.g_best.solution - self.pop[idx].solution)
            X2 = leaders[1].solution - miu2 * np.abs(c2 * self.g_best.solution - self.pop[idx].solution)
            X3 = leaders[2].solution - miu3 * np.abs(c3 * self.g_best.solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + leaders, self.pop_size, self.problem.minmax)


class GWO_WOA(OriginalGWO):
    """
    The original version of: Hybrid Grey Wolf - Whale Optimization Algorithm (GWO_WOA)

    Links:
        1. https://sci-hub.se/https://doi.org/10.1177/10775463211003402

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
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
    >>> model = GWO.GWO_WOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Obadina, O. O., Thaha, M. A., Althoefer, K., & Shaheed, M. H. (2022). Dynamic characterization of a master–slave
    robotic manipulator using a hybrid grey wolf–whale optimization algorithm. Journal of Vibration and Control, 28(15-16), 1992-2003.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.bb = 1.0
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2. - epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            if self.generator.random() < 0.5:
                da = self.generator.random() * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            else:
                P, L = self.generator.random(), self.generator.uniform(-1, 1)
                da = P * np.exp(self.bb * L) * np.cos(2*np.pi*L) * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X1 = list_best[0].solution - A1 * da
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class IGWO(OriginalGWO):
    """
    The original version of: Improved Grey Wolf Optimization (IGWO)

    Notes:
        1. Link: https://doi.org/10.1007/s00366-017-0567-1
        2. Inspired by: Mohammadtaher Abbasi (https://github.com/mtabbasi)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_min (float): Lower bound of a, default = 0.02
        + a_max (float): Upper bound of a, default = 2.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
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
    >>> model = GWO.IGWO(epoch=1000, pop_size=50, a_min = 0.02, a_max = 2.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
    Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, a_min: float = 0.02, a_max: float = 2.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_min (float): Lower bound of a, default = 0.02
            a_max (float): Upper bound of a, default = 2.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.a_min = self.validator.check_float("a_min", a_min, (0.0, 1.6))
        self.a_max = self.validator.check_float("a_max", a_max, [1., 4.])
        self.set_parameters(["epoch", "pop_size", "a_min", "a_max"])
        self.growth_alpha = 2
        self.growth_delta = 3

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm.

        Args:
            epoch (int): The current iteration
        """
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            # IGWO functions
            a_alpha = self.a_max * np.exp((epoch / self.epoch) ** self.growth_alpha * np.log(self.a_min / self.a_max))
            a_delta = self.a_max * np.exp((epoch / self.epoch) ** self.growth_delta * np.log(self.a_min / self.a_max))
            a_beta = (a_alpha + a_delta) * 0.5
            A1 = a_alpha * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a_beta * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a_delta * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
