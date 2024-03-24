#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalNMRA(Optimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (NMRA)

    Links:
        1. https://www.doi.org10.1007/s00521-019-04464-7

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, NMRA
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
    >>> model = NMRA.OriginalNMRA(epoch=1000, pop_size=50, pb = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm. Neural Computing and Applications, 31(12), pp.8837-8857.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, pb: float = 0.75, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): probability of breeding, default = 0.75
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pb"])
        self.sort_flag = True
        self.size_b = int(self.pop_size / 5)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    alpha = self.generator.uniform()
                    pos_new = (1 - alpha) * self.pop[idx].solution + alpha * (self.g_best.solution - self.pop[idx].solution)
            else:  # working operators
                t1, t2 = self.generator.choice(range(self.size_b, self.pop_size), 2, replace=False)
                pos_new = self.pop[idx].solution + self.generator.uniform() * (self.pop[t1].solution - self.pop[t2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class ImprovedNMRA(Optimizer):
    """
    The developed version of: Improved Naked Mole-Rat Algorithm (I-NMRA)

    Notes:
        + Use mutation probability idea
        + Use crossover operator
        + Use Levy-flight technique

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75
        + pm (float): [0.01, 0.1], probability of mutation, default = 0.01

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, NMRA
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
    >>> model = NMRA.ImprovedNMRA(epoch=1000, pop_size=50, pb = 0.75, pm = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, pb=0.75, pm=0.01, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): breeding probability, default = 0.75
            pm (float): probability of mutation, default = 0.01
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pb", "pm"])
        self.sort_flag = True
        self.size_b = int(self.pop_size / 5)

    def crossover_random__(self, pop, g_best):
        start_point = self.generator.integers(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[self.generator.integers(0, self.pop_size)].solution
        new_temp = g_best.solution.copy()
        new_temp[0:id1] = g_best.solution[0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best.solution[id2:id3]
        return new_temp

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    pos_new = self.pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * \
                              (self.g_best.solution - self.pop[idx].solution)
                else:
                    levy_step = self.get_levy_flight_step(beta=1, multiplier=0.001, case=-1)
                    pos_new = self.pop[idx].solution + 1.0 / np.sqrt(epoch) * np.sign(self.generator.random() - 0.5) * \
                              levy_step * (self.pop[idx].solution - self.g_best.solution)
            # Exploitation
            else:  # working operators
                if self.generator.uniform() < 0.5:
                    t1, t2 = self.generator.choice(range(self.size_b, self.pop_size), 2, replace=False)
                    pos_new = self.pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * \
                              (self.pop[t1].solution - self.pop[t2].solution)
                else:
                    pos_new = self.crossover_random__(self.pop, self.g_best)
            # Mutation
            temp = self.generator.uniform(self.problem.lb, self.problem.ub)
            condition = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
            pos_new = np.where(condition, temp, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
