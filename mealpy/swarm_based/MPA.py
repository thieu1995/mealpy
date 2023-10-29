#!/usr/bin/env python
# Created by "Thieu" at 17:28, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMPA(Optimizer):
    """
    The developed version: Marine Predators Algorithm (MPA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025
        2. https://www.mathworks.com/matlabcentral/fileexchange/74578-marine-predators-algorithm-mpa

    Notes:
        1. To use the original paper, set the training mode = "swarm"
        2. They update the whole population at the same time before update the fitness
        3. Two variables that they consider it as constants which are FADS = 0.2 and P = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MPA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = MPA.OriginalMPA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
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
        self.FADS = 0.2
        self.P = 0.5

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        CF = (1 - epoch/self.epoch)**(2 * epoch/self.epoch)
        # RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(self.pop_size, self.problem.n_dims), case=-1)
        RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(self.pop_size, self.problem.n_dims), case=-1)
        RB = self.generator.standard_normal((self.pop_size, self.problem.n_dims))
        per1 = self.generator.permutation(self.pop_size)
        per2 = self.generator.permutation(self.pop_size)
        pop_new = []
        for idx in range(0, self.pop_size):
            R = self.generator.random(self.problem.n_dims)
            t = self.epoch
            if t < self.epoch / 3:     # Phase 1 (Eq.12)
                step_size = RB[idx] * (self.g_best.solution - RB[idx] * self.pop[idx].solution)
                pos_new = self.pop[idx].solution + self.P * R * step_size
            elif self.epoch / 3 < t < 2*self.epoch / 3:     # Phase 2 (Eqs. 13 & 14)
                if idx > self.pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.g_best.solution + self.P * CF * step_size
                else:
                    step_size = RL[idx] * (self.g_best.solution - RL[idx] * self.pop[idx].solution)
                    pos_new = self.pop[idx].solution + self.P * R * step_size
            else:       # Phase 3 (Eq. 15)
                step_size = RL[idx] * (RL[idx] * self.g_best.solution - self.pop[idx].solution)
                pos_new = self.g_best.solution + self.P * CF * step_size
            pos_new = self.correct_solution(pos_new)
            if self.generator.random() < self.FADS:
                u = np.where(self.generator.random(self.problem.n_dims) < self.FADS, 1, 0)
                pos_new = pos_new + CF * (self.problem.lb + self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb)) * u
            else:
                r = self.generator.random()
                step_size = (self.FADS * (1 - r) + r) * (self.pop[per1[idx]].solution - self.pop[per2[idx]].solution)
                pos_new = pos_new + step_size
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
