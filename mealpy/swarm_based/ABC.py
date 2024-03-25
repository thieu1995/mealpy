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
    >>> from mealpy import FloatVar, ABC
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
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
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
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[idx].solution + phi * (self.pop[rdx].solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.pop_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = self.get_index_roulette_wheel_selection(employed_fits)
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, selected_bee}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[selected_bee].solution + phi * (self.pop[rdx].solution - self.pop[selected_bee].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[selected_bee].target, self.problem.minmax):
                self.pop[selected_bee] = agent
                self.trials[selected_bee] = 0
            else:
                self.trials[selected_bee] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
