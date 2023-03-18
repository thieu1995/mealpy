#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalABC(Optimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ABC import OriginalABC
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
    >>> n_limits = 50
    >>> model = OriginalABC(epoch, pop_size, n_limits)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """
    def __init__(self, epoch=10000, pop_size=100, n_limits=25, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size = onlooker bees = employed bees, default = 100
            n_limits (int): Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.support_parallel_modes = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

    def initialize_variables(self):
        self.trials = np.zeros(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            # Choose a random employed bee to generate a new solution
            t = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = np.random.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[idx][self.ID_POS] + phi * (self.pop[t][self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop[idx]):
                self.pop[idx] = [pos_new, target]
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.pop_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = self.get_index_roulette_wheel_selection(employed_fits)
            # Choose a random employed bee to generate a new solution
            t = np.random.choice(list(set(range(0, self.pop_size)) - {idx, selected_bee}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = np.random.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[selected_bee][self.ID_POS] + phi * (self.pop[t][self.ID_POS] - self.pop[selected_bee][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop[selected_bee]):
                self.pop[selected_bee] = [pos_new, target]
                self.trials[selected_bee] = 0
            else:
                self.trials[selected_bee] += 1

        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
            self.trials[idx] = 0
