#!/usr/bin/env python
# Created by "Thieu" at 13:42, 06/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSeaHO(Optimizer):
    """
    The original version of: Sea-Horse Optimization (SeaHO)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03994-3
        2. https://www.mathworks.com/matlabcentral/fileexchange/115945-sea-horse-optimizer

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SeaHO
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
    >>> model = SeaHO.OriginalSeaHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, S., Zhang, T., Ma, S., & Wang, M. (2022). Sea-horse optimizer: a novel nature-inspired
    meta-heuristic for global optimization problems. Applied Intelligence, 1-28.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.uu = 0.05
        self.vv = 0.05
        self.ll = 0.05

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # The motor behavior of sea horses
        step_length = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(self.pop_size, self.problem.n_dims), case=-1)
        pop_new = []
        for idx in range(0, self.pop_size):
            beta = self.generator.normal(0, 1, self.problem.n_dims)
            theta = 2 * np.pi * self.generator.random(self.problem.n_dims)
            row = self.uu * np.exp(theta * self.vv)
            xx, yy, zz = row * np.cos(theta), row * np.sin(theta), row * theta
            if self.generator.normal(0, 1) > 0:      # Eq. 4
                pos_new = self.pop[idx].solution + step_length[idx] * ((self.g_best.solution - self.pop[idx].solution) * xx * yy * zz + self.g_best.solution)
            else:                               # Eq. 7
                pos_new = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * self.ll * beta * (self.g_best.solution - beta * self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            pop_new.append(pos_new)

        # The predation behavior of sea horses
        pop_child = []
        alpha = (1 - epoch/self.epoch) ** (2 * epoch / self.epoch)
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            if self.generator.random() >= 0.1:
                pos_new = alpha * (self.g_best.solution - r1 * pop_new[idx]) + (1 - alpha) * self.g_best.solution        # Eq. 10
            else:
                pos_new = (1 - alpha) * (pop_new[idx] - r1 * self.g_best.solution) + alpha * pop_new[idx]               # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_child[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
        pop_child = self.get_sorted_population(pop_child, self.problem.minmax)         # Sorted population

        # The reproductive behavior of sea horses
        dads = pop_child[:int(self.pop_size/2)]
        moms = pop_child[int(self.pop_size/2):]
        pop_offspring = []
        for kdx in range(0, int(self.pop_size/2)):
            r3 = self.generator.random()
            pos_new = r3 * dads[kdx].solution + (1 - r3) * moms[kdx].solution           # Eq. 13
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_offspring.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_offspring[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_offspring = self.update_target_for_population(pop_offspring)
        # Sea horses selection
        self.pop = self.get_sorted_and_trimmed_population(pop_child + pop_offspring, self.pop_size, self.problem.minmax)
