#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSCA(Optimizer):
    """
    The developed version: Sine Cosine Algorithm (SCA)

    Notes
    ~~~~~
    + The flow and few equations are changed
    + Third loops are removed faster computational time

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SCA import BaseSCA
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
    >>> model = BaseSCA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a - (epoch + 1) * (a / self.epoch)
            # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
            r2 = 2 * np.pi * np.random.uniform(0, 1, self.problem.n_dims)
            r3 = 2 * np.random.uniform(0, 1, self.problem.n_dims)
            # Eq. 3.3, 3.1 and 3.2
            pos_new1 = self.pop[idx][self.ID_POS] + r1 * np.sin(r2) * np.abs(r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new2 = self.pop[idx][self.ID_POS] + r1 * np.cos(r2) * np.abs(r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = np.where(np.random.random(self.problem.n_dims) < 0.5, pos_new1, pos_new2)
            # Check the bound
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalSCA(BaseSCA):
    """
    The original version of: Sine Cosine Algorithm (SCA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2015.12.022
        2. https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SCA import OriginalSCA
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
    >>> model = OriginalSCA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. SCA: a sine cosine algorithm for solving optimization problems. Knowledge-based systems, 96, pp.120-133.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = False

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        return np.where(np.logical_and(lb <= position, position <= ub), position, np.random.uniform(lb, ub))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a - (epoch + 1) * (a / self.epoch)
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(self.problem.n_dims):  # j-th dimension
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * np.pi * np.random.uniform()
                r3 = 2 * np.random.uniform()
                r4 = np.random.uniform()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    pos_new[j] = pos_new[j] + r1 * np.sin(r2) * np.abs(r3 * self.g_best[self.ID_POS][j] - pos_new[j])
                else:
                    pos_new[j] = pos_new[j] + r1 * np.cos(r2) * np.abs(r3 * self.g_best[self.ID_POS][j] - pos_new[j])
            # Check the bound
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class QTable:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        # Initialize the Q-table with zeros
        self.table = np.zeros((n_states, n_actions))
        # Define the ranges for r1 and r3
        self.r1_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        self.r3_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        # Define the ranges for density and distance
        self.density_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.distance_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.epsilon = 0.1

    def get_state(self, density, distance):
        density_range = next(i for i, r in enumerate(self.density_ranges) if density <= r[1])
        distance_range = next(i for i, r in enumerate(self.distance_ranges) if distance <= r[1])
        return density_range * 3 + distance_range

    def get_action(self, state):
        acts = self.table[state, :]
        # Find the maximum value in the array
        max_val = np.max(acts)
        # Create a boolean mask that identifies all elements with the maximum value
        max_indices = np.where(acts == max_val)[0]
        # Use np.random.choice to randomly select an index from the list of indices with maximum value
        return np.random.choice(max_indices)

    def get_action_params(self, action):
        r1_range = self.r1_ranges[action // 3]
        r3_range = self.r3_ranges[action % 3]
        return r1_range, r3_range

    def update(self, state, action, reward, alpha=0.1, gamma=0.9):
        self.table[state][action] += alpha * (reward + gamma * np.max(self.table[state]) - self.table[state][action])


class QleSCA(BaseSCA):
    """
    The original version of: QLE Sine Cosine Algorithm (QLE-SCA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417421017048

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.1-1.0], the is the learning rate in Q-learning, default=0.1
        + gamma (float): [0.1-1.0]: the discount factor, default=0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.SCA import QleSCA
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
    >>> model = QleSCA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Hamad, Q. S., Samma, H., Suandi, S. A., & Mohamad-Saleh, J. (2022). Q-learning embedded sine cosine
    algorithm (QLESCA). Expert Systems with Applications, 193, 116417.
    """
    ID_QTB = 2

    def __init__(self, epoch=10000, pop_size=100, alpha=0.1, gamma=0.9, **kwargs):
        """
        Args:

            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the learning rate, default=0.1
            gamma (float): the discount factor, default=0.9
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.alpha = self.validator.check_float("alpha", alpha, [0.0, 1.0])
        self.gamma = self.validator.check_float("gamma", gamma, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "alpha", "gamma"])
        self.sort_flag = False
        self.support_parallel_modes = False

    def create_solution(self, lb=None, ub=None, pos=None):
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        q_table = QTable(n_states=9, n_actions=9)
        return [position, target, q_table]

    def amend_position(self, position=None, lb=None, ub=None):
        return np.where(np.logical_and(lb <= position, position <= ub), position, np.random.uniform(lb, ub))

    def density__(self, pop):
        agents = np.array([agent[self.ID_POS] for agent in pop])
        # calculate the mean of each dimension of the agents
        Y = np.mean(agents, axis=0)
        # calculate the longest diagonal length L
        distances = np.sqrt(np.sum((agents[:, np.newaxis, :] - agents) ** 2, axis=-1))
        L = np.max(distances)
        # calculate the density
        return 1 / (len(pop) * L) * np.sum(np.sqrt(np.sum((agents - Y) ** 2, axis=1)))

    def distance__(self, best, pop, lb, ub):
        agents = np.array([agent[self.ID_POS] for agent in pop])
        # calculate the numerator of the distance
        numerator = np.sum(np.sqrt(np.sum((best[self.ID_POS] - agents) ** 2, axis=1)))
        # calculate the denominator of the distance
        denominator = np.sum([ np.sqrt(np.sum((ub - lb) ** 2)) for _ in range(0, len(pop))])
        # calculate the distance
        return numerator / denominator

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            ## Step 3: State computation
            den = self.density__(self.pop)
            dis = self.distance__(self.g_best, self.pop, self.problem.lb, self.problem.ub)
            ## Step 4: Action execution
            state = self.pop[idx][self.ID_QTB].get_state(density=den, distance=dis)
            action = self.pop[idx][self.ID_QTB].get_action(state=state)
            r1_bound, r3_bound = self.pop[idx][self.ID_QTB].get_action_params(action)
            r1 = np.random.uniform(r1_bound[0], r1_bound[1])
            r3 = np.random.uniform(r3_bound[0], r3_bound[1])
            r2 = 2 * np.pi * np.random.uniform()
            r4 = np.random.uniform()
            if r4 < 0.5:
                pos_new = self.pop[idx][self.ID_POS] + r1 * np.sin(r2) * (r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] + r1 * np.cos(r2) * (r3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            # Check the bound
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            agent[self.ID_TAR] = self.get_target_wrapper(pos_new)
            if self.compare_agent(agent, self.pop[idx]):
                self.pop[idx] = agent
                self.pop[idx][self.ID_QTB].update(state, action, reward=1, alpha=self.alpha, gamma=self.gamma)
            else:
                self.pop[idx][self.ID_QTB].update(state, action, reward=-1, alpha=self.alpha, gamma=self.gamma)
