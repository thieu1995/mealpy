#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalBA(Optimizer):
    """
    The original version of: Bat-inspired Algorithm (BA)

    Notes
    ~~~~~
    + The value of A and r parameters are constant

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness (float): (1.0, 2.0), loudness, default = 0.8
        + pulse_rate (float): (0.15, 0.85), pulse rate / emission rate, default = 0.95
        + pulse_frequency (list, tuple): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BA.OriginalBA(epoch=1000, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, loudness: float = 0.8,
                 pulse_rate: float = 0.95, pf_min: float = 0., pf_max: float = 10., **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness (float): (A_min, A_max): loudness, default = 0.8
            pulse_rate (float): (r_min, r_max): pulse rate / emission rate, default = 0.95
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.loudness = self.validator.check_float("loudness", loudness, (0, 1.0))
        self.pulse_rate = self.validator.check_float("pulse_rate", pulse_rate, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [0., 3.0])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [5., 20.])
        self.set_parameters(["epoch", "pop_size", "loudness", "pulse_rate", "pf_min", "pf_max"])
        self.alpha = self.gamma = 0.9
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        pulse_frequency = self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform()
        return Agent(solution=solution, velocity=velocity, pulse_frequency=pulse_frequency)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            vec = agent.velocity + self.pop[idx].pulse_frequency * (self.pop[idx].solution - self.g_best.solution)
            x_new = self.pop[idx].solution + agent.velocity
            ## Local Search around g_best position
            if self.generator.random() > self.pulse_rate:
                x_new = self.g_best.solution + 0.001 * self.generator.normal(self.problem.n_dims)
            pos_new = self.correct_solution(x_new)
            agent.update(solution=pos_new, velocity=vec)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax) and self.generator.random() < self.loudness:
                self.pop[idx].update(solution=pop_new[idx].solution, target=pop_new[idx].target)


class AdaptiveBA(Optimizer):
    """
    The original version of: Adaptive Bat-inspired Algorithm (ABA)

    Notes
    ~~~~~
    + The value of A and r are changing after each iteration

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness_min (float): A_min - loudness, default=1.0
        + loudness_max (float): A_max - loudness, default=2.0
        + pr_min (float): pulse rate / emission rate min, default = 0.15
        + pr_max (float): pulse rate / emission rate max, default = 0.85
        + pf_min (float): pulse frequency min, default = 0
        + pf_max (float): pulse frequency max, default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BA.AdaptiveBA(epoch=1000, pop_size=50, loudness_min = 1.0, loudness_max = 2.0, pr_min = -2.5, pr_max = 0.85, pf_min = 0.1, pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch: int = 10000, pop_size: object = 100, loudness_min: float = 1.0, loudness_max: float = 2.0,
                 pr_min: float = 0.15, pr_max: float = 0.85, pf_min: float = -10., pf_max: float = 10., **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness_min (float): A_min - loudness, default=1.0
            loudness_max (float): A_max - loudness, default=2.0
            pr_min (float): pulse rate / emission rate min, default = 0.15
            pr_max (float): pulse rate / emission rate max, default = 0.85
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.loudness_min = self.validator.check_float("loudness_min", loudness_min, [0.5, 1.5])
        self.loudness_max = self.validator.check_float("loudness_max", loudness_max, [1.5, 3.0])
        self.pr_min = self.validator.check_float("pr_min", pr_min, (0, 1.0))
        self.pr_max = self.validator.check_float("pr_max", pr_max, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [-10, 0])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [0, 10])
        self.alpha = self.gamma = 0.9
        self.set_parameters(["epoch", "pop_size", "loudness_min", "loudness_max", "pr_min", "pr_max", "pf_min", "pf_max"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        loudness = self.generator.uniform(self.loudness_min, self.loudness_max)
        pulse_rate = self.generator.uniform(self.pr_min, self.pr_max)
        return Agent(solution=solution, velocity=velocity, loudness=loudness, pulse_rate=pulse_rate)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        mean_a = np.mean([agent.loudness for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            pulse_frequency = self.generator.uniform(self.pf_min, self.pf_max)
            agent.velocity = agent.velocity + pulse_frequency * (self.pop[idx].solution - self.g_best.solution)
            x_new = self.pop[idx].solution + agent.velocity
            ## Local Search around g_best position
            if self.generator.random() > agent.pulse_rate:
                x_new = self.g_best.solution + mean_a * self.generator.normal(-1, 1)
            pos_new = self.correct_solution(x_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax) and self.generator.random() < pop_new[idx].loudness:
                loudness = self.alpha * pop_new[idx].loudness
                pulse_rate = pop_new[idx].pulse_rate * (1 - np.exp(-self.gamma * epoch))
                self.pop[idx].update(solution=pop_new[idx].solution, target=pop_new[idx].target, loudness=loudness, pulse_rate=pulse_rate)


class DevBA(Optimizer):
    """
    The original version of: Developed Bat-inspired Algorithm (DBA)

    Notes
    ~~~~~
    + A (loudness) parameter is removed
    + Flow is changed:
        + 1st: the exploration phase is proceed (using frequency)
        + 2nd: If new position has better fitness, replace the old position
        + 3rd: Otherwise, proceed exploitation phase (using finding around the best position so far)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pulse_rate (float): [0.7, 1.0], pulse rate / emission rate, default = 0.95
        + pulse_frequency (tuple, list): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BA.DevBA(epoch=1000, pop_size=50, pulse_rate = 0.95, pf_min = 0., pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, pulse_rate=0.95, pf_min=0., pf_max=10., **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pulse_rate = self.validator.check_float("pulse_rate", pulse_rate, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [0, 2])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [2, 10])
        self.alpha = self.gamma = 0.9
        self.set_parameters(["epoch", "pop_size", "pulse_rate", "pf_min", "pf_max"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_list_velocity = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pf = self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform()  # Eq. 2
            self.dyn_list_velocity[idx] = self.generator.uniform() * self.dyn_list_velocity[idx] + (self.g_best.solution - self.pop[idx].solution) * pf  # Eq. 3
            x = self.pop[idx].solution + self.dyn_list_velocity[idx]  # Eq. 4
            pos_new = self.correct_solution(x)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        pop_child_idx = []
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pop_new[idx].solution.copy(), target=pop_new[idx].target)
            else:
                if self.generator.random() > self.pulse_rate:
                    x = self.g_best.solution + 0.01 * self.generator.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.correct_solution(x)
                    agent = self.generate_empty_agent(pos_new)
                    pop_child_idx.append(idx)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
        pop_child = self.update_target_for_population(pop_child)
        for idx, idx_selected in enumerate(pop_child_idx):
            if self.compare_target(pop_child[idx].target, pop_new[idx_selected].target, self.problem.minmax):
                pop_new[idx_selected].update(solution=pop_child[idx].solution, target=pop_child[idx].target)
        self.pop = pop_new
