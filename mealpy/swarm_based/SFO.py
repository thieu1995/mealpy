#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSFO(Optimizer):
    """
    The original version of: SailFish Optimizer (SFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.01.001

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        + AP (float): coefficient for decreasing the value of Attack Power linearly from AP to 0
        + epsilon (float): should be 0.0001, 0.001

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SFO
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
    >>> model = SFO.OriginalSFO(epoch=1000, pop_size=50, pp = 0.1, AP = 4.0, epsilon = 0.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shadravan, S., Naji, H.R. and Bardsiri, V.K., 2019. The Sailfish Optimizer: A novel nature-inspired metaheuristic
    algorithm for solving constrained engineering optimization problems. Engineering Applications of Artificial Intelligence, 80, pp.20-34.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pp: float = 0.1, AP: float = 4.0, epsilon: float = 0.0001, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            AP (float): coefficient for decreasing the value of Power Attack linearly from AP to 0
            epsilon (float): should be 0.0001, 0.001
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pp = self.validator.check_float("pp", pp, (0, 1.0))
        self.AP = self.validator.check_float("AP", AP, (0, 100))
        self.epsilon = self.validator.check_float("epsilon", epsilon, (0, 0.1))
        self.set_parameters(["epoch", "pop_size", "pp", "AP", "epsilon"])
        self.sort_flag = True
        self.s_size = int(self.pop_size / self.pp)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)      # pop = sailfish
        self.s_pop = self.generate_population(self.s_size)
        self.s_gbest = self.get_best_agent(self.s_pop, self.problem.minmax)          # s_pop = sardines

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        pop_new = []
        PD = 1 - self.pop_size / (self.pop_size + self.s_size)
        for idx in range(0, self.pop_size):
            lamda_i = 2 * self.generator.uniform() * PD - PD
            pos_new = self.s_gbest.solution - lamda_i * \
                (self.generator.uniform() * (self.pop[idx].solution + self.s_gbest.solution) / 2 - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        ## Calculate AttackPower using Eq.(10)
        AP = self.AP * (1. - 2. * epoch * self.epsilon)
        if AP < 0.5:
            alpha = int(self.s_size * np.abs(AP))
            beta = int(self.problem.n_dims * np.abs(AP))
            ### Random self.generator.choice number of sardines which will be updated their position
            list1 = self.generator.choice(range(0, self.s_size), alpha)
            for idx in range(0, self.s_size):
                if idx in list1:
                    #### Random self.generator.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                    pos_new = self.s_pop[idx].solution.copy()
                    list2 = self.generator.choice(range(0, self.problem.n_dims), beta, replace=False)
                    pos_new[list2] = (self.generator.uniform(0, 1, self.problem.n_dims) * (self.s_gbest.solution - self.s_pop[idx].solution + AP))[list2]
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    if self.mode not in self.AVAILABLE_MODES:
                        agent.target = self.get_target(pos_new)
                        self.s_pop[idx] = agent
        else:
            ### Update the position of all sardine using Eq.(9)
            for idx in range(0, self.s_size):
                pos_new = self.generator.uniform() * (self.g_best.solution - self.s_pop[idx].solution + AP)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.s_pop[idx] = agent
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_target_for_population(self.s_pop)
        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)
        self.s_pop = self.get_sorted_and_trimmed_population(self.s_pop, len(self.s_pop), self.problem.minmax)
        for idx in range(0, self.pop_size):
            for jdx in range(0, self.s_size):
                ### If there is a better position in sardine population.
                if self.compare_target(self.s_pop[jdx].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = self.s_pop[jdx].copy()
                    del self.s_pop[jdx]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        temp = self.s_size - len(self.s_pop)
        if temp == 1:
            self.s_pop = self.s_pop + [self.generate_agent()]
        else:
            self.s_pop = self.s_pop + self.generate_population(self.s_size - len(self.s_pop))
        self.s_gbest = self.get_best_agent(self.s_pop, self.problem.minmax)


class ImprovedSFO(Optimizer):
    """
    The original version: Improved Sailfish Optimizer (I-SFO)

    Notes:
        + Energy equation is reformed
        + AP (A) and epsilon parameters are removed
        + Opposition-based learning technique is used

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SFO
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
    >>> model = SFO.ImprovedSFO(epoch=1000, pop_size=50, pp = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, pp: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pp = self.validator.check_float("pp", pp, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pp"])
        self.sort_flag = True
        self.s_size = int(self.pop_size / self.pp)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.s_pop = self.generate_population(self.s_size)
        self.s_gbest = self.get_best_agent(self.s_pop, self.problem.minmax)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        pop_new = []
        for idx in range(0, self.pop_size):
            PD = 1 - len(self.pop) / (len(self.pop) + len(self.s_pop))
            lamda_i = 2 * self.generator.uniform() * PD - PD
            pos_new = self.s_gbest.solution - \
                lamda_i * (self.generator.uniform() * (self.g_best.solution + self.s_gbest.solution) / 2 - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        ## ## Calculate AttackPower using my Eq.thieu
        #### This is our proposed, simple but effective, no need A and epsilon parameters
        AP = 1 - epoch * 1.0 / self.epoch
        if AP < 0.5:
            for idx in range(0, len(self.s_pop)):
                temp = (self.g_best.solution + AP) / 2
                pos_new = self.problem.lb + self.problem.ub - temp + self.generator.uniform() * (temp - self.s_pop[idx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.s_pop[idx] = agent
        else:
            ### Update the position of all sardine using Eq.(9)
            for idx in range(0, len(self.s_pop)):
                pos_new = self.generator.uniform() * (self.g_best.solution - self.s_pop[idx].solution + AP)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.s_pop[idx] = agent
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_target_for_population(self.s_pop)
        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)
        self.s_pop = self.get_sorted_and_trimmed_population(self.s_pop, len(self.s_pop), self.problem.minmax)
        for idx in range(0, self.pop_size):
            for jdx in range(0, len(self.s_pop)):
                ### If there is a better position in sardine population.
                if self.compare_target(self.s_pop[jdx].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = self.s_pop[jdx].copy()
                    del self.s_pop[jdx]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        self.s_pop = self.s_pop + self.generate_population(self.s_size - len(self.s_pop))
        self.s_gbest = self.get_best_agent(self.s_pop, self.problem.minmax)
