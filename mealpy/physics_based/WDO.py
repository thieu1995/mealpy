#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:18, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import ones, clip
from mealpy.root import Root


class BaseWDO(Root):
    """
    The original version of : Wind Driven Optimization (WDO)
        The Wind Driven Optimization Technique and its Application in Electromagnetics
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 RT=3, g=0.2, alp=0.4, c=0.4, max_v=0.3, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.RT = RT                # RT coefficient
        self.g = g                  # gravitational constant
        self.alp = alp              # constants in the update equation
        self.c = c                  # coriolis effect
        self.max_v = max_v          # maximum allowed speed

    def train(self):
        """
        # pop is the set of "air parcel" - "position"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in position and has its own velocity
        # pressure represented by fitness value
        """
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        list_velocity = self.max_v * uniform(self.lb, self.ub, (self.pop_size, self.problem_size))

        for epoch in range(self.epoch):

            # Update velocity based on random dimensions and position of global best
            for i in range(self.pop_size):

                rand_dim = randint(0, self.problem_size)
                temp = list_velocity[i][rand_dim] * ones(self.problem_size)
                vel = (1 - self.alp)*list_velocity[i] - self.g * pop[i][self.ID_POS] + \
                      (1 - 1.0/(i+1)) * self.RT * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + self.c * temp / (i+1)
                vel = clip(vel, -self.max_v, self.max_v)

                # Update air parcel positions, check the bound and calculate pressure (fitness)
                pos = pop[i][self.ID_POS] + vel
                pos = self.amend_position_faster(pos)
                fit = self.get_fitness_position(pos)
                pop[i] = [pos, fit]
                list_velocity[i] = vel

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

