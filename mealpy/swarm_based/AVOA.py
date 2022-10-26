#!/usr/bin/env python
# Created by "Thieu" at 21:45, 26/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAVOA(Optimizer):
    """
    The original version of: African Vultures Optimization Algorithm (AVOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0360835221003120
        2. https://www.mathworks.com/matlabcentral/fileexchange/94820-african-vultures-optimization-algorithm

    Notes (parameters):
        + p1 (float): probability of status transition, default 0.6
        + p2 (float): probability of status transition, default 0.4
        + p3 (float): probability of status transition, default 0.6
        + alpha (float): probability of 1st best, default = 0.8
        + gama (float): a factor in the paper (not much affect to algorithm), default = 2.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.AVOA import OriginalAVOA
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
    >>> p1 = 0.6
    >>> p2 = 0.4
    >>> p3 = 0.6
    >>> alpha = 0.8
    >>> gama = 2.5
    >>> model = OriginalAVOA(epoch, pop_size, p1, p2, p3, alpha, gama)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
    nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
    """

    def __init__(self, epoch=10000, pop_size=100, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))
        self.p2 = self.validator.check_float("p2", p2, (0, 1))
        self.p3 = self.validator.check_float("p3", p3, (0, 1))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1))
        self.gama = self.validator.check_float("gama", gama, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def get_levy_flight__(self, beta=1.0, size=None):
        sigma = np.random.gamma(1 + beta) * np.sin(np.pi * beta/2) / (np.random.gamma((1+beta)/2) * beta * 2**((beta-1)/2)) ** (1 / beta)
        u = np.random.normal(0, 1, size) * sigma
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = np.random.uniform(-2, 2) * ((np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama) + np.cos((np.pi / 2) * (epoch / self.epoch)) - 1)
        ppp = (2 * np.random.rand() + 1) * (1 - epoch/self.epoch) + a

        _, best_list, _ = self.get_special_solutions(self.pop, best=2)
        pop_new = []
        for idx in range(0, self.pop_size):
            F = ppp * (2 * np.random.rand() -1)
            rand_idx = np.random.choice([0, 1], p=[self.alpha, 1-self.alpha])
            rand_pos = best_list[rand_idx][self.ID_POS]
            if np.abs(F) >= 1:      # Exploration
                if np.random.rand() < self.p1:
                    pos_new = rand_pos - (np.abs((2 * np.random.rand()) * rand_pos - self.pop[idx][self.ID_POS])) * F
                else:
                    pos_new = rand_pos - F + np.random.rand()*((self.problem.ub - self.problem.lb)*np.random.rand() + self.problem.lb)
            else:                   # Exploitation
                if np.abs(F) < 0.5:      # Phase 1
                    best_x1 = best_list[0][self.ID_POS]
                    best_x2 = best_list[1][self.ID_POS]
                    if np.random.rand() < self.p2:
                        A = best_x1 - ((best_x1 * self.pop[idx][self.ID_POS]) / (best_x1 - self.pop[idx][self.ID_POS]**2))*F
                        B = best_x2-((best_x2 * self.pop[idx][self.ID_POS]) / (best_x2 - self.pop[idx][self.ID_POS]**2))*F
                        pos_new = (A + B) / 2
                    else:
                        pos_new = rand_pos - np.abs(rand_pos - self.pop[idx][self.ID_POS]) * F * self.get_levy_flight__(beta=1.5, size=self.problem.n_dims)
                else:       # Phase 2
                    if np.random.rand() < self.p3:
                        pos_new = (np.abs((2 * np.random.rand()) * rand_pos - self.pop[idx][self.ID_POS])) * (F + np.random.rand()) - \
                                  (rand_pos - self.pop[idx][self.ID_POS])
                    else:
                        s1 = rand_pos * (np.random.rand() * self.pop[idx][self.ID_POS] / (2 * np.pi)) * np.cos(self.pop[idx][self.ID_POS])
                        s2 = rand_pos * (np.random.rand() * self.pop[idx][self.ID_POS] / (2 * np.pi)) * np.sin(self.pop[idx][self.ID_POS])
                        pos_new = rand_pos - (s1 + s2)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop_new)
