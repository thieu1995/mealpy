#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevEFO(Optimizer):
    """
    The developed version: Electromagnetic Field Optimization (EFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EFO
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
    >>> model = EFO.DevEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, r_rate: float = 0.3,
                 ps_rate: float = 0.85, p_field: float = 0.1, n_field: float = 0.45, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.r_rate = self.validator.check_float("r_rate", r_rate, (0, 1.0))
        self.ps_rate = self.validator.check_float("ps_rate", ps_rate, (0, 1.0))
        self.p_field = self.validator.check_float("p_field", p_field, (0, 1.0))
        self.n_field = self.validator.check_float("n_field", n_field, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "r_rate", "ps_rate", "p_field", "n_field"])
        self.phi = (1 + np.sqrt(5)) / 2  # golden ratio
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            r_idx1 = self.generator.integers(0, int(self.pop_size * self.p_field))  # top
            r_idx2 = self.generator.integers(int(self.pop_size * (1 - self.n_field)), self.pop_size)  # bottom
            r_idx3 = self.generator.integers(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))  # middle
            if self.generator.random() < self.ps_rate:
                pos_new = self.pop[r_idx1].solution + self.phi * self.generator.random() * (self.g_best.solution - self.pop[r_idx3].solution) \
                          + self.generator.random() * (self.g_best.solution - self.pop[r_idx2].solution)
            else:
                pos_new = self.problem.generate_solution()
            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if self.generator.random() < self.r_rate:
                RI = self.generator.integers(0, self.problem.n_dims)
                pos_new[self.generator.integers(0, self.problem.n_dims)] = self.generator.uniform(self.problem.lb[RI], self.problem.ub[RI])
            # checking whether the generated number is inside boundary or not
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalEFO(DevEFO):
    """
    The original version of: Electromagnetic Field Optimization (EFO)

    Links:
        2. https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EFO
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
    >>> model = EFO.OriginalEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z. and Jawawi, D.N., 2016.
    Electromagnetic field optimization: a physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation, 26, pp.8-22.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, r_rate: float = 0.3,
                 ps_rate: float = 0.85, p_field: float = 0.1, n_field: float = 0.45, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(epoch, pop_size, r_rate, ps_rate, p_field, n_field, **kwargs)
        self.support_parallel_modes = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rd)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        self.r_index1 = self.generator.integers(0, int(self.pop_size * self.p_field), (self.problem.n_dims, self.epoch))
        # random particles from positive field
        self.r_index2 = self.generator.integers(int(self.pop_size * (1 - self.n_field)), self.pop_size, (self.problem.n_dims, self.epoch))
        # random particles from negative field
        self.r_index3 = self.generator.integers(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)), (self.problem.n_dims, self.epoch))
        # random particles from neutral field
        self.ps = self.generator.uniform(0, 1, (self.problem.n_dims, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        self.r_force = self.generator.uniform(0, 1, self.epoch)
        # random force in each generation
        self.rp = self.generator.uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        self.randomization = self.generator.uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        self.RI = 0
        # index of the electromagnet (variable) which is going to be initialized by random number

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        iter01 = epoch-1
        r = self.r_force[iter01]
        x_new = np.zeros(self.problem.n_dims)  # temporary array to store generated particle
        for idx in range(0, self.problem.n_dims):
            if self.ps[idx, iter01] > self.ps_rate:
                x_new[idx] = self.pop[self.r_index3[idx, iter01]].solution[idx] + \
                           self.phi * r * (self.pop[self.r_index1[idx, iter01]].solution[idx] - self.pop[self.r_index3[idx, iter01]].solution[idx]) + \
                           r * (self.pop[self.r_index3[idx, iter01]].solution[idx] - self.pop[self.r_index2[idx, iter01]].solution[idx])
            else:
                x_new[idx] = self.pop[self.r_index1[idx, iter01]].solution[idx]
        # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
        if self.rp[iter01] < self.r_rate:
            x_new[self.RI] = self.problem.lb[self.RI] + (self.problem.ub[self.RI] - self.problem.lb[self.RI]) * self.randomization[iter01]
            RI = self.RI + 1
            if RI >= self.problem.n_dims:
                self.RI = 0
        # checking whether the generated number is inside boundary or not
        pos_new = self.correct_solution(x_new)
        agent = self.generate_agent(pos_new)
        # Updating the population if the fitness of the generated particle is better than worst fitness in
        #     the population (because the population is sorted by fitness, the last particle is the worst)
        self.pop[-1] = agent
