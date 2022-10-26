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

    Notes
    ~~~~~
        1. To use the original paper, set the training mode = "swarm"
        2. They update the whole population at the same time before update the fitness
        3. Two variables that they consider it as constants which are FADS = 0.2 and P = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MPA import OriginalMPA
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
    >>> model = OriginalMPA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        self.FADS = 0.2
        self.P = 0.5

    def levy_step(self, beta, size=None):
        num = np.random.gamma(1 + beta) * np.sin(np.pi * beta /2)
        den = np.random.gamma((1+beta)/2) * beta * 2**((beta-1)/2)
        sigma_u = (num/den) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        CF = (1 - (epoch+1)/self.epoch)**(2 * (epoch+1)/self.epoch)
        # RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(self.pop_size, self.problem.n_dims), case=-1)
        RL = 0.05 * self.levy_step(1.5, (self.pop_size, self.problem.n_dims))
        RB = np.random.randn(self.pop_size, self.problem.n_dims)
        per1 = np.random.permutation(self.pop_size)
        per2 = np.random.permutation(self.pop_size)
        pop_new = []
        for idx in range(0, self.pop_size):
            R = np.random.rand(self.problem.n_dims)
            t = self.epoch + 1
            if t < self.epoch / 3:     # Phase 1 (Eq.12)
                step_size = RB[idx] * (self.g_best[self.ID_POS] - RB[idx] * self.pop[idx][self.ID_POS])
                pos_new = self.pop[idx][self.ID_POS] + self.P * R * step_size
            elif self.epoch / 3 < t < 2*self.epoch / 3:     # Phase 2 (Eqs. 13 & 14)
                if idx > self.pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] + self.P * CF * step_size
                else:
                    step_size = RL[idx] * (self.g_best[self.ID_POS] - RL[idx] * self.pop[idx][self.ID_POS])
                    pos_new = self.pop[idx][self.ID_POS] + self.P * R * step_size
            else:       # Phase 3 (Eq. 15)
                step_size = RL[idx] * (RL[idx] * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + self.P * CF * step_size
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            if np.random.rand() < self.FADS:
                u = np.where(np.random.rand(self.problem.n_dims) < self.FADS, 1, 0)
                pos_new = pos_new + CF * (self.problem.lb + np.random.rand(self.problem.n_dims) * (self.problem.ub - self.problem.lb)) * u
            else:
                r = np.random.rand()
                step_size = (self.FADS * (1 - r) + r) * (self.pop[per1[idx]][self.ID_POS] - self.pop[per2[idx]][self.ID_POS])
                pos_new = pos_new + step_size
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
