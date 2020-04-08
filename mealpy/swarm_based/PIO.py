#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:41, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import exp, sum
from mealpy.root import Root


class BasePIO(Root):
    """
        My improved version of: Pigeon-Inspired Optimization (PIO)
            (Pigeon-inspired optimization: a new swarm intelligence optimizer for air robot path planning)
        Link:
            + DOI: 10.1108/IJICC-02-2014-0005
        Noted:
            + The paper is very unclear most the parameters and the flow of algorithm (some points even wrong)
            + This is my version, I changed almost everything, even parameters and flow of algorithm
            + Also the personal best no need in this version (So it is now much different than PSO)
    """
    ID_POS = 0  # current position
    ID_FIT = 1  # current fitness
    ID_VEL = 2  # velocity of bird

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, R=0.2, n_switch=0.75):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch                                  # Nc1 + Nc2
        self.pop_size = pop_size                            # Np
        self.R = R
        if n_switch < 1:
            self.n_switch = int(self.epoch * n_switch)
        else:
            self.n_switch = int(n_switch)                    # Represent Nc1 and Nc2 in the paper

    def _create_solution__(self, minmax=0):
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)      # x: current position
        fit = self._fitness_model__(solution=x, minmax=minmax)
        v = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)      # v: velocity of this bird (same number of dimension of x)
        return [x, fit, v]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
        n_p = int(self.pop_size / 2)

        for epoch in range(0, self.epoch):

            if epoch < self.n_switch:  # Map and compass operations
                for i in range(0, self.pop_size):
                    v_new = pop[i][self.ID_POS] * exp(-self.R * (epoch + 1)) + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    x_new = pop[i][self.ID_POS] + v_new
                    x_new = self._amend_solution_random_faster__(x_new)
                    fit = self._fitness_model__(x_new)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = [x_new, fit, v_new]

            else:  # Landmark operations
                pop = sorted(pop, key=lambda item: item[self.ID_FIT])
                list_fit = [pop[i][self.ID_FIT] for i in range(0, n_p)]
                list_pos = [pop[i][self.ID_FIT] for i in range(0, n_p)]
                frac_up = sum([list_fit[i] * list_pos[i] for i in range(0, n_p)], axis=0)
                frac_down = n_p * sum(list_fit)
                x_c = frac_up / frac_down

                ## Move all pigeon based on target x_c
                for i in range(0, self.pop_size):
                    x_new = pop[i][self.ID_POS] + uniform() * (x_c - pop[i][self.ID_POS])
                    fit = self._fitness_model__(x_new)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i][self.ID_POS] = x_new
                        pop[i][self.ID_FIT] = fit

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyPIO(Root):
    """
        My levy version of: Pigeon-Inspired Optimization (PIO)
            (Pigeon-inspired optimization: a new swarm intelligence optimizer for air robot path planning)
        Noted:
            + The paper is very unclear most the parameters and the flow of algorithm (some points even wrong)
            + This is my version, I changed almost everything, even parameters and flow of algorithm
            + Also the personal best no need in this version (So it is now much different than PSO)
            + I applied the levy-flight here for more robust
    """
    ID_POS = 0  # current position
    ID_FIT = 1  # current fitness
    ID_VEL = 2  # velocity of bird

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, R=0.2, n_switch=0.75):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch  # Nc1 + Nc2
        self.pop_size = pop_size  # Np
        self.R = R
        if n_switch < 1:
            self.n_switch = int(self.epoch * n_switch)
        else:
            self.n_switch = int(n_switch)  # Represent Nc1 and Nc2 in the paper

    def _create_solution__(self, minmax=0):
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)  # x: current position
        fit = self._fitness_model__(solution=x, minmax=minmax)
        v = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)  # v: velocity of this bird (same number of dimension of x)
        return [x, fit, v]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
        n_p = int(self.pop_size / 2)

        for epoch in range(0, self.epoch):

            if epoch < self.n_switch:  # Map and compass operations
                for i in range(0, self.pop_size):
                    v_new = pop[i][self.ID_POS] * exp(-self.R * (epoch + 1)) + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    x_new = pop[i][self.ID_POS] + v_new
                    x_new = self._amend_solution_random_faster__(x_new)
                    fit = self._fitness_model__(x_new)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i] = [x_new, fit, v_new]

            else:  # Landmark operations
                pop = sorted(pop, key=lambda item: item[self.ID_FIT])
                list_fit = [pop[i][self.ID_FIT] for i in range(0, n_p)]
                list_pos = [pop[i][self.ID_FIT] for i in range(0, n_p)]
                frac_up = sum([list_fit[i] * list_pos[i] for i in range(0, n_p)], axis=0)
                frac_down = n_p * sum(list_fit)
                x_c = frac_up / frac_down

                ## Move all pigeon based on target x_c
                for i in range(0, self.pop_size):
                    if uniform() < 0.5:
                        x_new = pop[i][self.ID_POS] + uniform() * (x_c - pop[i][self.ID_POS])
                    else:
                        x_new = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                    fit = self._fitness_model__(x_new)
                    if fit < pop[i][self.ID_FIT]:
                        pop[i][self.ID_POS] = x_new
                        pop[i][self.ID_FIT] = fit

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
