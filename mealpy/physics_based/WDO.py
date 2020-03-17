#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import ones, clip
from mealpy.root import Root


class BaseWDO(Root):
    """
    Basic: Wind Driven Optimization (WDO)
        The Wind Driven Optimization Technique and its Application in Electromagnetics
    """
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2      # Velocity

    def __init__(self, root_paras=None, epoch=750, pop_size=100, RT=3, g=0.2, alp=0.4, c=0.4, max_v=0.3):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size
        self.RT = RT                # RT coefficient
        self.g = g                  # gravitational constant
        self.alp = alp              # constants in the update equation
        self.c = c                  # coriolis effect
        self.max_v = max_v          # maximum allowed speed

    def _create_solution__(self, minmax=0):
        solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution)
        v = self.max_v * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        return [solution, fitness, v]

    def _train__(self):
        """
        # pop is the set of "air parcel" - "solution"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in solution and has its own velocity
        # pressure represented by fitness value
        """
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            # Update velocity based on random dimensions and position of global best
            for i in range(self.pop_size):

                rand_dim = randint(0, self.problem_size)
                temp = pop[i][self.ID_VEL][rand_dim] * ones(self.problem_size)
                vel = (1 - self.alp)*pop[i][self.ID_VEL] - self.g * pop[i][self.ID_POS] + \
                      (1 - 1.0/(i+1)) * self.RT * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + self.c * temp / (i+1)
                vel = clip(vel, -self.max_v, self.max_v)

                # Update air parcel positions, check the bound and calculate pressure (fitness)
                pos = pop[i][self.ID_POS] + vel
                pos = self._amend_solution_faster__(pos)
                fit = self._fitness_model__(pos)
                pop[i] = [pos, fit, vel]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

