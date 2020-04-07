#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:40, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, pi, power, cos, sin, argmax, argmin
from numpy.random import uniform, choice, randint
from copy import deepcopy
from mealpy.root import Root


class OriginalAAA(Root):
    """
    The original version of: Artificial Algae Algorithm  (SBO)
        (Artificial algae algorithm (AAA) for nonlinear global optimization)
    Link:
        https://doi.org/10.1016/j.asoc.2015.03.003
    """

    ID_POS = 0
    ID_FIT = 1
    ID_SIZE = 2
    ID_ENERGY = 3           # energy
    ID_FRIC = 4             # friction surface
    ID_STAR = 5             # starvation

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, energy=0.3, delta=2, ap=0.5):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.energy = energy
        self.delta = delta
        self.ap = ap  # the loss of energy, e = 0.3, the shear force, delta = 2 and the adaptation probability constant, Ap = 0.5.

    def _get_solution_based_kway_tournament_selection__(self, pop=None, k_way=0.2):
        if k_way < 1:
            k_way = int(k_way * self.pop_size)
        list_id = choice(range(self.pop_size), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[self.ID_MIN_PROB]

    def _create_solution__(self, minmax=0):
        pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(pos)
        size = (1 * fit) / (fit/2 + fit)                        # G
        energy = self.energy                                # miu_max = 1, is the maximum specific growth rate, size = g_i, Eq. 8
        friction_surface = 2*pi*power( power(3*size/(4*pi), 1.0/3), 2)         # Eq. 15
        starvation = 0
        return [pos, fit, size, energy, friction_surface, starvation]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Calculate Energy E and friction surface (t) of n algae

            for i in range(0, self.pop_size):
                starvation = True
                pos_new = deepcopy(pop[i][self.ID_POS])
                ## Helical movement phrase
                while (pop[i][self.ID_ENERGY] > 0):
                    ## Choice j among all solution via tournament selection
                    solution_j = self._get_solution_based_kway_tournament_selection__(pop, k_way=0.2)
                    k1, k2, k3 = choice(range(self.problem_size), 3, replace=False)
                    alpha, beta, p = uniform(0, 2*pi), uniform(0, 2*pi), uniform(-1, 1)
                    pos_new[k1] = pop[i][self.ID_POS][k1] + cos(alpha)*(self.delta-pop[i][self.ID_FRIC])*(solution_j[self.ID_POS][k1] - pop[i][self.ID_POS][k1])
                    pos_new[k2] = pop[i][self.ID_POS][k2] + sin(beta)*(self.delta-pop[i][self.ID_FRIC])*(solution_j[self.ID_POS][k2] - pop[i][self.ID_POS][k2])
                    pos_new[k3] = pop[i][self.ID_POS][k3] + p * (self.delta - pop[i][self.ID_FRIC]) * (solution_j[self.ID_POS][k3] - pop[i][self.ID_POS][k3])

                    fit = self._fitness_model__(pos_new)
                    pop[i][self.ID_ENERGY] = pop[i][self.ID_ENERGY] - self.energy / 2
                    if fit < pop[i][self.ID_FIT]:
                        starvation = False
                        pop[i][self.ID_POS] = deepcopy(pos_new)
                        pop[i][self.ID_FIT] = fit
                    else:
                        pop[i][self.ID_ENERGY] = pop[i][self.ID_ENERGY] - self.energy / 2
                if starvation:
                    pop[i][self.ID_STAR] += self.ap

            ## Evaluate size and friction surface
            for i in range(0, self.pop_size):
                fit = pop[i][self.ID_FIT]
                pop[i][self.ID_SIZE] = (1 * fit) / (fit / 2 + fit)
                pop[i][self.ID_FRIC] = 2 * pi * power(power(3 * pop[i][self.ID_SIZE] / (4 * pi), 1.0 / 3), 2)           # Eq. 15

            ## Reproduction process
            k4 = randint(0, self.problem_size)
            size_list = array([item[self.ID_SIZE] for item in pop])
            minn, maxx = argmin(size_list), argmax(size_list)
            pop[minn][self.ID_POS][k4] = pop[maxx][self.ID_POS][k4]
            fit = self._fitness_model__(pop[minn][self.ID_POS])
            pop[minn][self.ID_FIT] = fit
            pop[minn][self.ID_SIZE] = (1 * fit) / (fit / 2 + fit)
            pop[minn][self.ID_FRIC] = 2 * pi * power(power(3 * pop[minn][self.ID_SIZE] / (4 * pi), 1.0 / 3), 2)         # Eq. 15

            ## Adaptation phrase
            if uniform() < self.ap:
                starvation_list = array([item[self.ID_STAR] for item in pop])
                maxx_star = argmax(starvation)
                pop[maxx_star][self.ID_STAR] += uniform() * (pop[maxx][self.ID_STAR] - pop[maxx_star][self.ID_STAR])

            ## Update global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
