#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevSMA(Optimizer):
    """
    The developed version: Slime Mould Algorithm (SMA)

    Notes:
        + Selected 2 unique and random solution to create new solution (not to create variable)
        + Check bound and compare old position with new position to get the best one

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SMA
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
    >>> model = SMA.DevSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_t: float = 0.03, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_t = self.validator.check_float("p_t", p_t, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_t"])
        self.sort_flag = True

    def initialize_variables(self):
        self.weights = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # plus eps to avoid denominator zero
        ss = self.g_best.target.fitness - self.pop[-1].target.fitness + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, self.pop_size):
            # Eq.(2.5)
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
            else:
                self.weights[idx] = 1 - self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
        a = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        b = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(np.abs(self.pop[idx].target.fitness - self.g_best.target.fitness))  # Eq.(2.2)
                vb = self.generator.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-b, b, self.problem.n_dims)
                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_1 = self.g_best.solution + vb * (self.weights[idx] * self.pop[id_a].solution - self.pop[id_b].solution)
                pos_2 = vc * self.pop[idx].solution
                condition = self.generator.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
            # Check bound and re-calculate fitness after each individual move
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalSMA(DevSMA):
    """
    The original version of: Slime Mould Algorithm (SMA)

    Links:
        1. https://doi.org/10.1016/j.future.2020.03.055
        2. https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SMA
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
    >>> model = SMA.OriginalSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, S., Chen, H., Wang, M., Heidari, A.A. and Mirjalili, S., 2020. Slime mould algorithm: A new method for
    stochastic optimization. Future Generation Computer Systems, 111, pp.300-323.
    """

    def __init__(self, epoch=10000, pop_size=100, p_t=0.03, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(epoch, pop_size, p_t, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # plus eps to avoid denominator zero
        ss = self.g_best.target.fitness - self.pop[-1].target.fitness + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, self.pop_size):
            # Eq.(2.5)
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
            else:
                self.weights[idx] = 1 - self.generator.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)

        aa = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        bb = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            pos_new = self.pop[idx].solution.copy()
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(np.abs(self.pop[idx].target.fitness - self.g_best.target.fitness))  # Eq.(2.2)
                vb = self.generator.uniform(-aa, aa, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-bb, bb, self.problem.n_dims)
                for jdx in range(0, self.problem.n_dims):
                    # two positions randomly selected from population
                    id_a, id_b = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    if self.generator.uniform() < p:  # Eq.(2.1)
                        pos_new[jdx] = self.g_best.solution[jdx] + vb[jdx] * (self.weights[idx, jdx] * self.pop[id_a].solution[jdx] - self.pop[id_b].solution[jdx])
                    else:
                        pos_new[jdx] = vc[jdx] * pos_new[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
