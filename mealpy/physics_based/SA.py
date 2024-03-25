#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSA(Optimizer):
    """
    The original version of: Simulated Annealing (OriginalSA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + step_size (float): the step size of random movement, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SA
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
    >>> model = SA.OriginalSA(epoch=1000, pop_size=50, temp_init = 100, step_size = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 2, temp_init: float = 100, step_size: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            step_size (float): the step size of random movement, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.step_size = self.validator.check_float("step_size", step_size, (-100., 100.))
        self.set_parameters(["epoch", "temp_init", "step_size"])

    def before_main_loop(self):
        self.agent_current = self.g_best.copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution

        pos_new = self.agent_current.solution + self.generator.standard_normal(self.problem.n_dims) * self.step_size
        agent = self.generate_agent(pos_new)
        # Accept or reject the new solution
        if self.compare_target(agent.target, self.agent_current.target, self.problem.minmax):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(self.agent_current.target.fitness - agent.target.fitness)
            # calculate probability acceptance criterion
            p_accept = np.exp(-delta_energy/ (self.temp_init / float(epoch + 1)))
            if self.generator.random() < p_accept:
                self.agent_current = agent
        self.pop = [self.g_best.copy(), self.agent_current.copy()]


class GaussianSA(Optimizer):
    """
    The developed version of: Gaussian Simulated Annealing (GaussianSA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm
        + The temp_init is very important factor. Should set it equal to the distance between LB and UB

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + cooling_rate (float): (0., 1.0), cooling rate, default=0.99
        + scale (float): (0., 100.), the scale in gaussian random, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SA
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
    >>> model = SA.GaussianSA(epoch=1000, pop_size=2, temp_init = 100, cooling_rate = 0.99, scale = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 2, temp_init: float = 100,
                 cooling_rate: float = 0.99, scale: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            cooling_rate (float): cooling rate, default=0.99
            scale (float): the scale in gaussian random, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.cooling_rate = self.validator.check_float("cooling_rate", cooling_rate, (0., 1.))
        self.scale = self.validator.check_float("scale", scale, (0., 100.))
        self.set_parameters(["epoch", "temp_init", "cooling_rate", "scale"])

    def before_main_loop(self):
        # Initialize the system
        self.temp_current = self.temp_init
        self.agent_current = self.g_best.copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution
        pos_new = self.agent_current.solution + self.generator.normal(scale=self.scale, size=self.problem.n_dims)
        agent = self.generate_agent(pos_new)
        # Accept or reject the new solution
        if self.compare_target(agent.target, self.agent_current.target, self.problem.minmax):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(self.agent_current.target.fitness - agent.target.fitness)
            p_accept = np.exp(-delta_energy/self.temp_current)
            if self.generator.random() < p_accept:
                self.agent_current = agent
        # Reduce the temperature
        self.temp_current *= self.cooling_rate
        self.pop = [self.g_best.copy(), self.agent_current.copy()]


class SwarmSA(Optimizer):
    """
    The swarm version of: Simulated Annealing (SwarmSA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + max_sub_iter (int): [5, 10, 15], Maximum Number of Sub-Iteration (within fixed temperature), default=5
        + t0 (int): Fixed parameter, Initial Temperature, default=1000
        + t1 (int): Fixed parameter, Final Temperature, default=1
        + move_count (int): [5, 20], Move Count per Individual Solution, default=5
        + mutation_rate (float): [0.01, 0.2], Mutation Rate, default=0.1
        + mutation_step_size (float): [0.05, 0.1, 0.15], Mutation Step Size, default=0.1
        + mutation_step_size_damp (float): [0.8, 0.99], Mutation Step Size Damp, default=0.99

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SA
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
    >>> model = SA.SwarmSA(epoch=1000, pop_size=50, max_sub_iter = 5, t0 = 1000, t1 = 1,
    >>>         move_count = 5, mutation_rate = 0.1, mutation_step_size = 0.1, mutation_step_size_damp = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Laarhoven, P.J. and Aarts, E.H., 1987. Simulated annealing. In Simulated
    annealing: Theory and applications (pp. 7-15). Springer, Dordrecht.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, max_sub_iter: int = 5, t0: int = 1000, t1: int = 1, move_count: int = 5,
                 mutation_rate: float = 0.1, mutation_step_size: float = 0.1, mutation_step_size_damp: float = 0.99, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (int): Maximum Number of Sub-Iteration (within fixed temperature), default=5
            t0 (int): Initial Temperature, default=1000
            t1 (int): Final Temperature, default=1
            move_count (int): Move Count per Individual Solution, default=5
            mutation_rate (float): Mutation Rate, default=0.1
            mutation_step_size (float): Mutation Step Size, default=0.1
            mutation_step_size_damp (float): Mutation Step Size Damp, default=0.99
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.max_sub_iter = self.validator.check_int("max_sub_iter", max_sub_iter, [1, 100000])
        self.t0 = self.validator.check_int("t0", t0, [500, 2000])
        self.t1 = self.validator.check_int("t1", t1, [1, 100])
        self.move_count = self.validator.check_int("move_count", move_count, [2, int(self.pop_size/2)])
        self.mutation_rate = self.validator.check_float("mutation_rate", mutation_rate, (0, 1.0))
        self.mutation_step_size = self.validator.check_float("mutation_step_size", mutation_step_size, (0, 1.0))
        self.mutation_step_size_damp = self.validator.check_float("mutation_step_size_damp", mutation_step_size_damp, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "max_sub_iter", "t0", "t1", "move_count",
                             "mutation_rate", "mutation_step_size", "mutation_step_size_damp"])
        self.sort_flag = True
        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def mutate__(self, position, sigma):
        # Select Mutating Variables
        pos_new = position + sigma * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(self.generator.random(self.problem.n_dims) < self.mutation_rate, position, pos_new)
        if np.all(pos_new == position):  # Select at least one variable to mutate
            pos_new[self.generator.integers(0, self.problem.n_dims)] = self.generator.uniform()
        return self.correct_solution(pos_new)

    def initialization(self):
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):
            # Create new population
            pop_new = []
            for idx in range(0, self.pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    pos_new = self.mutate__(self.pop[idx].solution, self.dyn_sigma)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_new[-1].target = self.get_target(pos_new)
            pop_new = self.update_target_for_population(pop_new)
            # Columnize and Sort Newly Created Population
            pop_new = self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)
            # Randomized Selection
            for idx in range(0, self.pop_size):
                # Check if new solution is better than current
                if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = pop_new[idx].copy()
                else:
                    # Compute difference according to problem type
                    delta = np.abs(pop_new[idx].target.fitness - self.pop[idx].target.fitness)
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if self.generator.uniform() <= p:  # Accept / Reject
                        self.pop[idx] = pop_new[idx].copy()
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma
