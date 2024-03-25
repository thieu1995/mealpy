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
    >>> from mealpy import FloatVar, AVOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AVOA.OriginalAVOA(epoch=1000, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
    nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p1: float = 0.6, p2: float = 0.4,
                 p3: float = 0.6, alpha: float = 0.8, gama: float = 2.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))
        self.p2 = self.validator.check_float("p2", p2, (0, 1))
        self.p3 = self.validator.check_float("p3", p3, (0, 1))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1))
        self.gama = self.validator.check_float("gama", gama, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = self.generator.uniform(-2, 2) * ((np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama) + np.cos((np.pi / 2) * (epoch / self.epoch)) - 1)
        ppp = (2 * self.generator.random() + 1) * (1 - epoch/self.epoch) + a
        _, best_list, _ = self.get_special_agents(self.pop, n_best=2, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            F = ppp * (2 * self.generator.random() -1)
            rand_idx = self.generator.choice([0, 1], p=[self.alpha, 1-self.alpha])
            rand_pos = best_list[rand_idx].solution
            if np.abs(F) >= 1:      # Exploration
                if self.generator.random() < self.p1:
                    pos_new = rand_pos - (np.abs((2 * self.generator.random()) * rand_pos - self.pop[idx].solution)) * F
                else:
                    pos_new = rand_pos - F + self.generator.random()*((self.problem.ub - self.problem.lb)*self.generator.random() + self.problem.lb)
            else:                   # Exploitation
                if np.abs(F) < 0.5:      # Phase 1
                    best_x1 = best_list[0].solution
                    best_x2 = best_list[1].solution
                    if self.generator.random() < self.p2:
                        A = best_x1 - ((best_x1 * self.pop[idx].solution) / (best_x1 - self.pop[idx].solution**2))*F
                        B = best_x2 - ((best_x2 * self.pop[idx].solution) / (best_x2 - self.pop[idx].solution**2))*F
                        pos_new = (A + B) / 2
                    else:
                        pos_new = rand_pos - np.abs(rand_pos - self.pop[idx].solution) * F * \
                                  self.get_levy_flight_step(beta=1.5, multiplier=1., size=self.problem.n_dims, case=-1)
                else:       # Phase 2
                    if self.generator.random() < self.p3:
                        pos_new = (np.abs((2 * self.generator.random()) * rand_pos - self.pop[idx].solution)) * (F + self.generator.random()) - \
                                  (rand_pos - self.pop[idx].solution)
                    else:
                        s1 = rand_pos * (self.generator.random() * self.pop[idx].solution / (2 * np.pi)) * np.cos(self.pop[idx].solution)
                        s2 = rand_pos * (self.generator.random() * self.pop[idx].solution / (2 * np.pi)) * np.sin(self.pop[idx].solution)
                        pos_new = rand_pos - (s1 + s2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)
