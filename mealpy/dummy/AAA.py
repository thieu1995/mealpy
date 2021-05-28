#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:40, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, pi, power, cos, sin, argmax, argmin, ones, zeros
from numpy.random import uniform, choice, randint
from copy import deepcopy
from mealpy.root import Root


class BaseAAA(Root):
    """
    My version of: Artificial Algae Algorithm (AAA)
        (Artificial algae algorithm (AAA) for nonlinear global optimization)
    Link:
        https://doi.org/10.1016/j.asoc.2015.03.003
    Notes:
        + Remove size value and replace by fitness. Then friction surface will change by time
        + In Adaptation phase, instead of change starving value, I change position value
        + Still not working
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 energy=0.3, delta=2, ap=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.energy = energy        # the loss of energy, e = 0.3,
        self.delta = delta          # the shear force, delta = 2
        self.ap = ap                # the adaptation probability constant, Ap = 0.5.

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        list_fit = array([item[self.ID_FIT] for item in pop])       # Fitness list
        list_energy = self.energy * ones(self.pop_size)             # Energy list, miu_max = 1, is the maximum specific growth rate, size = g_i, Eq. 8
        list_friction_surface = 2 * pi * power(power(3 * list_fit / (4 * pi), 1.0 / 3), 2)      # Friction surface list, Eq. 15
        list_starvation = zeros(self.pop_size)                      # Starvation list

        for epoch in range(self.epoch):
            ## Calculate Energy E and friction surface (t) of n algae

            for i in range(0, self.pop_size):
                starvation = True
                pos_new = deepcopy(pop[i][self.ID_POS])
                ## Helical movement phase
                while (list_energy[i] > 0):
                    ## Choice j among all position via tournament selection
                    solution_j = self.get_parent_kway_tournament_selection(pop, output=1)[0]
                    k1, k2, k3 = choice(range(self.problem_size), 3, replace=False)
                    alpha, beta, p = uniform(0, 2 * pi), uniform(0, 2 * pi), uniform(-1, 1)

                    pos_new[k1] = pop[i][self.ID_POS][k1] + cos(alpha) * (self.delta - list_friction_surface[i]) * \
                                  (solution_j[self.ID_POS][k1] - pop[i][self.ID_POS][k1])
                    pos_new[k2] = pop[i][self.ID_POS][k2] + sin(beta) * (self.delta - list_friction_surface[i]) * \
                                  (solution_j[self.ID_POS][k2] - pop[i][self.ID_POS][k2])
                    pos_new[k3] = pop[i][self.ID_POS][k3] + p * (self.delta - list_friction_surface[i]) * \
                                  (solution_j[self.ID_POS][k3] - pop[i][self.ID_POS][k3])

                    fit_new = self.get_fitness_position(pos_new)
                    list_energy[i] -= self.energy / 2.0
                    if fit_new < pop[i][self.ID_FIT]:
                        starvation = False
                        pop[i] = [deepcopy(pos_new), fit_new]
                    else:
                        list_energy[i] -= self.energy / 2
                if starvation:
                    list_starvation[i] += self.ap

                ## Evaluate size and friction surface
                list_friction_surface[i] = 2 * pi * power(power(3 * pop[i][self.ID_FIT] / (4 * pi), 1.0 / 3), 2)  # Eq. 15

            ## Reproduction process
            k4 = randint(0, self.problem_size)
            minn, maxx = argmax(list_fit), argmin(list_fit)
            pop[minn][self.ID_POS][k4] = pop[maxx][self.ID_POS][k4]
            fit_new = self.get_fitness_position(pop[minn][self.ID_POS])
            list_friction_surface[minn] = 2 * pi * power(power(3 * fit_new / (4 * pi), 1.0 / 3), 2)            # Eq. 15

            ## Adaptation phase
            if uniform() < self.ap:
                maxx_star = argmax(list_starvation)
                pos_new = pop[maxx_star][self.ID_POS] + uniform() * (pop[maxx][self.ID_POS] - pop[maxx_star][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pop[maxx_star] = [pos_new, fit_new]
                list_friction_surface[maxx_star] = 2 * pi * power(power(3 * fit_new / (4 * pi), 1.0 / 3), 2)

            ## Update global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalAAA(Root):
    """
    The original version of: Artificial Algae Algorithm (SBO)
        (Artificial algae algorithm (AAA) for nonlinear global optimization)
    Link:
        https://doi.org/10.1016/j.asoc.2015.03.003

        + I realize in the original paper, parameters and equations not clear.
        + In Adaptation phase, what is the point of saving starving value when it doesn't effect to solution at all?
        + The size of solution always = 2/3, so the friction surface will always stay at the same value.
        + The idea of equation seem like taken from DE, the adaptation and reproduction process seem like taken from CRO.
        + Appearance from 2015, but still now 2020 none of matlab code or python code about this algorithm.
    """

    ID_POS = 0
    ID_FIT = 1
    ID_SIZE = 2
    ID_ENERGY = 3           # energy
    ID_FRIC = 4             # friction surface
    ID_STAR = 5             # starvation

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 energy=0.3, delta=2, ap=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.energy = energy        # the loss of energy, e = 0.3,
        self.delta = delta          # the shear force, delta = 2
        self.ap = ap                # the adaptation probability constant, Ap = 0.5.

    def create_solution(self, minmax=0):
        pos = uniform(self.lb, self.ub)
        fit = self.get_fitness_position(pos)
        size = (1 * fit) / (fit/2 + fit)                                # G
        energy = self.energy                                            # miu_max = 1, is the maximum specific growth rate, size = g_i, Eq. 8
        friction_surface = 2*pi*power(power(3*size/(4*pi), 1.0/3), 2)   # Eq. 15
        starvation = 0
        return [pos, fit, size, energy, friction_surface, starvation]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Calculate Energy E and friction surface (t) of n algae

            for i in range(0, self.pop_size):
                starvation = True
                pos_new = deepcopy(pop[i][self.ID_POS])
                ## Helical movement phrase
                while (pop[i][self.ID_ENERGY] > 0):
                    ## Choice j among all position via tournament selection
                    solution_j = self.get_parent_kway_tournament_selection(pop, output=1)[0]
                    k1, k2, k3 = choice(range(self.problem_size), 3, replace=False)
                    alpha, beta, p = uniform(0, 2*pi), uniform(0, 2*pi), uniform(-1, 1)
                    pos_new[k1] = pop[i][self.ID_POS][k1] + cos(alpha)*(self.delta-pop[i][self.ID_FRIC])*(solution_j[self.ID_POS][k1] - pop[i][self.ID_POS][k1])
                    pos_new[k2] = pop[i][self.ID_POS][k2] + sin(beta)*(self.delta-pop[i][self.ID_FRIC])*(solution_j[self.ID_POS][k2] - pop[i][self.ID_POS][k2])
                    pos_new[k3] = pop[i][self.ID_POS][k3] + p * (self.delta - pop[i][self.ID_FRIC]) * (solution_j[self.ID_POS][k3] - pop[i][self.ID_POS][k3])

                    fit = self.get_fitness_position(pos_new)
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
                pop[i][self.ID_SIZE] = (1 * fit) / (fit / 2 + fit) * pop[i][self.ID_SIZE]
                pop[i][self.ID_FRIC] = 2 * pi * power(power(3 * pop[i][self.ID_SIZE] / (4 * pi), 1.0 / 3), 2)           # Eq. 15

            ## Reproduction process
            k4 = randint(0, self.problem_size)
            size_list = array([item[self.ID_SIZE] for item in pop])
            minn, maxx = argmin(size_list), argmax(size_list)
            pop[minn][self.ID_POS][k4] = pop[maxx][self.ID_POS][k4]
            fit = self.get_fitness_position(pop[minn][self.ID_POS])
            pop[minn][self.ID_FIT] = fit
            pop[minn][self.ID_SIZE] = (1 * fit) / (fit / 2 + fit)
            pop[minn][self.ID_FRIC] = 2 * pi * power(power(3 * pop[minn][self.ID_SIZE] / (4 * pi), 1.0 / 3), 2)         # Eq. 15

            ## Adaptation phrase
            if uniform() < self.ap:
                starvation_list = array([item[self.ID_STAR] for item in pop])
                maxx_star = argmax(starvation_list)
                pop[maxx_star][self.ID_STAR] += uniform() * (pop[maxx][self.ID_STAR] - pop[maxx_star][self.ID_STAR])

            ## Update global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
