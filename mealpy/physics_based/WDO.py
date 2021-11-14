#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseWDO(Optimizer):
    """
    The original version of : Wind Driven Optimization (WDO)
        The Wind Driven Optimization Technique and its Application in Electromagnetics
    Link:
        https://ieeexplore.ieee.org/abstract/document/6407788
    Note:
        # pop is the set of "air parcel" - "position"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in position and has its own velocity
        # pressure represented by fitness value
    """

    def __init__(self, problem, epoch=10000, pop_size=100, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            RT (int): RT coefficient, default = 3
            g_c (float): gravitational constant, default = 0.2
            alp (float): constants in the update equation, default=0.4
            c_e (float): coriolis effect, default=0.4
            max_v (float): maximum allowed speed, default=0.3
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.RT = RT
        self.g_c = g_c
        self.alp = alp
        self.c_e = c_e
        self.max_v = max_v

        ## Dynamic variable
        self.dyn_list_velocity = self.max_v * np.random.uniform(self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            rand_dim = np.random.randint(0, self.problem.n_dims)
            temp = self.dyn_list_velocity[idx][rand_dim] * np.ones(self.problem.n_dims)
            vel = (1 - self.alp) * self.dyn_list_velocity[idx] - self.g_c * self.pop[idx][self.ID_POS] + \
                  (1 - 1.0 / (idx + 1)) * self.RT * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + self.c_e * temp / (idx + 1)
            vel = np.clip(vel, -self.max_v, self.max_v)

            # Update air parcel positions, check the bound and calculate pressure (fitness)
            self.dyn_list_velocity[idx] = vel
            pos = self.pop[idx][self.ID_POS] + vel
            pos_new = self.amend_position_faster(pos)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
