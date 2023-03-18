#!/usr/bin/env python
# Created by "Thieu" at 21:58, 16/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from math import gamma
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGTO(Optimizer):
    """
    The original version of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. This algorithm costs a huge amount of computational resources in each epoch.
        Therefore, be careful when using the maximum number of generations as a stopping condition.
        2. Other algorithms update around K*pop_size times in each epoch, this algorithm updates around 2*pop_size^2 + pop_size times

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GTO import OriginalGTO
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
    >>> model = OriginalGTO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """
    def __init__(self, epoch=10000, pop_size=100, p1=0.03, p2=0.8, beta=3.0, **kwargs):
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

    def levy__(self, beta=1.0, size=None, step=0.01):
        num = gamma(1 + beta) * np.sin(np.pi * beta/2)
        den = gamma((1+beta)/2) * beta * 2**((beta-1)/2)
        sigma_u = (num/den)**(1.0/beta)
        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)
        return u/(np.abs(v)**(1.0/beta)) * step

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Step 1: Extensive Search
        for idx in range(0, self.pop_size):
            pop_new = []
            for jdx in range(0, self.pop_size):
                if idx == jdx:
                    continue
                # foraging movement patterns of giant trevallies are simulated using Eq.(4)
                pos_new = self.g_best[self.ID_POS] * np.random.rand() + ((self.problem.ub - self.problem.lb) * np.random.rand() + self.problem.lb) * \
                          self.levy__(beta=1.5, size=self.problem.n_dims)
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new = self.update_target_wrapper_population(pop_new)
            _, self.pop[idx] = self.get_global_best_solution(pop_new + [self.pop[idx]])
        _, self.g_best = self.update_global_best_solution(self.pop, save=False)

        # Step 2: Choosing Area
        pos_list = np.array([agent[self.ID_POS] for agent in self.pop])
        pos_m = np.mean(pos_list, axis=0)
        A = 0.4
        pop_new = []
        for idx in range(0, self.pop_size):
            # In the choosing area step, giant trevallies identify and select the best area in terms of
            # the amount of food (seabirds) within the selected search space where they can hunt for prey.
            r3 = np.random.rand()
            pos_new = self.g_best[self.ID_POS] * A * r3 + pos_m - self.pop[idx][self.ID_POS] * r3        # Eq. 7
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, self.g_best = self.update_global_best_solution(self.pop, save=False)

        # Step 3: Attacking
        H = np.random.rand() * (2.0 - (epoch+1) * 2.0 / self.epoch)     #  Eq.(15)
        for idx in range(0, self.pop_size):
            pop_new = []
            for jdx in range(0, self.pop_size):
                if idx == jdx:
                    continue
                # the distance between the prey and the attacker, and can be calculated using (12):
                dist = np.sum(np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]))
                theta2 = (360 - 0) * np.random.rand() + 0
                theta1 = 1.3296 * np.sin(np.radians(theta2))        # calculate theta_1 using (10)
                # visual distortion indicates the apparent height of the bird, which is always seen
                # to be higher than its actual height due to the refraction of the light.
                VD = np.sin(np.radians(theta1)) * dist      # Eq. 11
                # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated using (13)
                pos_new = self.pop[idx][self.ID_POS] * np.sin(np.radians(theta2)) * self.pop[idx][self.ID_TAR][self.ID_FIT] + VD + H
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new = self.update_target_wrapper_population(pop_new)
            _, self.pop[idx] = self.get_global_best_solution(pop_new + [self.pop[idx]])
