#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:22, 29/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%


from numpy.random import uniform, normal
from numpy import abs, exp, ones
from copy import deepcopy
from mealpy.root import Root


class BaseSpaSA(Root):
    """
        My version of: Sparrow Search Algorithm
            (A novel swarm intelligence optimization approach: sparrow search algorithm)
        Link:
            https://doi.org/10.1080/21642583.2019.1708830
        Noted:
            + In Eq. 4, Instead of using A+ and L, I used normal(). Because at the end L*A+ is only a random number
            + Their algorithm 1 flow is missing all important component such as g_best, fitness updated,
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.ST = 0.8       # ST in [0.5, 1.0]
        self.PD = 0.2       # number of producers
        self.SD = 0.1       # number of sparrows who perceive the danger


    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
        n1 = int(self.PD * self.pop_size)
        n2 = int(self.SD * self.pop_size)

        for epoch in range(self.epoch):
            r2 = uniform()              # R2 in [0, 1], the alarm value, random value

            # Using equation (3) update the sparrow’s location;
            for i in range(0, n1):
                if r2 < self.ST:
                    x_new = pop[i][self.ID_POS] * exp((i+1) / (uniform() * self.epoch))
                else:
                    x_new = pop[i][self.ID_POS] + normal() * ones(self.problem_size)
                x_new = self._amend_solution_random_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_new, fit]

            x_p = deepcopy(sorted(pop[:n1], key=lambda item: item[self.ID_FIT])[0][self.ID_POS])
            worst = deepcopy(sorted(pop, key=lambda item: item[self.ID_FIT])[-1])

            # Using equation (4) update the sparrow’s location;
            for i in range(n1, self.pop_size):
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((worst[self.ID_POS] - pop[i][self.ID_POS]) / (i+1)**2)
                else:
                    x_new = x_p + abs(pop[i][self.ID_POS] - x_p) * normal()
                x_new = self._amend_solution_random_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_new, fit]

            #  Using equation (5) update the sparrow’s location;
            for i in range(0, n2):
                if pop[i][self.ID_FIT] > g_best[self.ID_FIT]:
                    x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                            (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                x_new = self._amend_solution_random_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_new, fit]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

