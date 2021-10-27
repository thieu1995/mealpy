#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
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

    def create_child(self, idx, pop, g_best):
        rand_dim = np.random.randint(0, self.problem.n_dims)
        temp = self.dyn_list_velocity[idx][rand_dim] * np.ones(self.problem.n_dims)
        vel = (1 - self.alp) * self.dyn_list_velocity[idx] - self.g_c * pop[idx][self.ID_POS] + \
              (1 - 1.0 / (idx + 1)) * self.RT * (g_best[self.ID_POS] - pop[idx][self.ID_POS]) + self.c_e * temp / (idx + 1)
        vel = np.clip(vel, -self.max_v, self.max_v)

        # Update air parcel positions, check the bound and calculate pressure (fitness)
        self.dyn_list_velocity[idx] = vel
        pos = pop[idx][self.ID_POS] + vel
        pos_new = self.amend_position_faster(pos)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

        # ## batch size idea
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) % self.pop_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best) for idx in pop_idx]
        return child
