#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


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
    >>> from mealpy.swarm_based.BA import OriginalBA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> loudness = 0.8
    >>> pulse_rate = 0.95
    >>> pf_min = 0.
    >>> pf_max = 10.
    >>> model = OriginalBA(epoch, pop_size, loudness, pulse_rate, pf_min, pf_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    ID_VEC = 2  # Velocity
    ID_PFRE = 3  # Pulse Frequency

    def __init__(self, epoch=10000, pop_size=100, loudness=0.8, pulse_rate=0.95, pf_min=0., pf_max=10., **kwargs):
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
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.loudness = self.validator.check_float("loudness", loudness, (0, 1.0))
        self.pulse_rate = self.validator.check_float("pulse_rate", pulse_rate, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [0., 3.0])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [5., 20.])
        self.set_parameters(["epoch", "pop_size", "loudness", "pulse_rate", "pf_min", "pf_max"])
        self.alpha = self.gamma = 0.9
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: a solution with format [position, target, velocity, pulse_frequency]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(lb, ub)
        pulse_frequency = self.pf_min + (self.pf_max - self.pf_min) * np.random.uniform()
        return [position, target, velocity, pulse_frequency]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            agent[self.ID_VEC] = agent[self.ID_VEC] + self.pop[idx][self.ID_PFRE] * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            x = self.pop[idx][self.ID_POS] + agent[self.ID_VEC]
            ## Local Search around g_best position
            if np.random.uniform() > self.pulse_rate:
                x = self.g_best[self.ID_POS] + 0.0001 * np.random.normal(self.problem.n_dims)  # gauss
            pos_new = self.amend_position(x, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_agent(pop_new[idx], self.pop[idx]) and np.random.rand() < self.loudness:
                self.pop[idx] = deepcopy(pop_new[idx])


class AdaptiveBA(Optimizer):
    """
    The original version of: Adaptive Bat-inspired Algorithm (BA)

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
    >>> from mealpy.swarm_based.BA import AdaptiveBA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> loudness_min = 1.0 
    >>> loudness_max = 2.0 
    >>> pr_min = 0.15
    >>> pr_max = 0.85
    >>> pf_min = 0.
    >>> pf_max = 10.
    >>> model = AdaptiveBA(epoch, pop_size, loudness_min, loudness_max, pr_min, pr_max, pf_min, pf_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    ID_VEC = 2  # Velocity
    ID_LOUD = 3  # Loudness
    ID_PRAT = 4  # Pulse Rate
    ID_PFRE = 5  # Pulse Frequency

    def __init__(self, epoch=10000, pop_size=100, loudness_min=1.0, loudness_max=2.0, pr_min=0.15, pr_max=0.85, pf_min=0., pf_max=10., **kwargs):
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
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.loudness_min = self.validator.check_float("loudness_min", loudness_min, [0.5, 1.5])
        self.loudness_max = self.validator.check_float("loudness_max", loudness_max, [1.5, 3.0])
        self.pr_min = self.validator.check_float("pr_min", pr_min, (0, 1.0))
        self.pr_max = self.validator.check_float("pr_max", pr_max, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [0, 2])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [2, 10])
        self.alpha = self.gamma = 0.9
        self.set_parameters(["epoch", "pop_size", "loudness_min", "loudness_max", "pr_min", "pr_max", "pf_min", "pf_max"])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: a solution with format [position, target, velocity, loudness, pulse_rate, pulse_frequency]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(lb, ub)
        loudness = np.random.uniform(self.loudness_min, self.loudness_max)
        pulse_rate = np.random.uniform(self.pr_min, self.pr_max)
        pulse_frequency = self.pf_min + (self.pf_max - self.pf_min) * np.random.uniform()
        return [position, target, velocity, loudness, pulse_rate, pulse_frequency]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        mean_a = np.mean([agent[self.ID_LOUD] for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            agent[self.ID_VEC] = agent[self.ID_VEC] + self.pop[idx][self.ID_PFRE] * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            x = self.pop[idx][self.ID_POS] + agent[self.ID_VEC]
            ## Local Search around g_best position
            if np.random.uniform() > agent[self.ID_PRAT]:
                # print(f"{epoch}, {mean_a}, {self.dyn_r}")
                x = self.g_best[self.ID_POS] + mean_a * np.random.normal(-1, 1)
            pos_new = self.amend_position(x, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx in range(0, self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_agent(pop_new[idx], self.pop[idx]) and np.random.rand() < pop_new[idx][self.ID_LOUD]:
                pop_new[idx][self.ID_LOUD] = self.alpha * pop_new[idx][self.ID_LOUD]
                pop_new[idx][self.ID_PRAT] = pop_new[idx][self.ID_PRAT] * (1 - np.exp(-self.gamma * (epoch + 1)))
                self.pop[idx] = deepcopy(pop_new[idx])


class ModifiedBA(Optimizer):
    """
    The original version of: Modified Bat-inspired Algorithm (MBA)

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
    >>> from mealpy.swarm_based.BA import ModifiedBA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> pulse_rate = 0.95
    >>> pf_min = 0.
    >>> pf_max = 10.
    >>> model = ModifiedBA(epoch, pop_size, pulse_rate, pf_min, pf_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, pulse_rate=0.95, pf_min=0., pf_max=10., **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
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
            pf = self.pf_min + (self.pf_max - self.pf_min) * np.random.uniform()  # Eq. 2
            self.dyn_list_velocity[idx] = np.random.uniform() * self.dyn_list_velocity[idx] + \
                                          (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * pf  # Eq. 3
            x = self.pop[idx][self.ID_POS] + self.dyn_list_velocity[idx]  # Eq. 4
            pos_new = self.amend_position(x, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_child_idx = []
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
            else:
                if np.random.random() > self.pulse_rate:
                    x = self.g_best[self.ID_POS] + 0.01 * np.random.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.amend_position(x, self.problem.lb, self.problem.ub)
                    pop_child_idx.append(idx)
                    pop_child.append([pos_new, None])
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_child = self.update_target_wrapper_population(pop_child)
        for idx, idx_selected in enumerate(pop_child_idx):
            if self.compare_agent(pop_child[idx], pop_new[idx_selected]):
                pop_new[idx_selected] = deepcopy(pop_child[idx])
        self.pop = pop_new
