#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, sin, pi , array, mean, zeros, where
from numpy.random import uniform, randint
from numpy.linalg import norm
from copy import deepcopy
from mealpy.root import Root


class BaseASO(Root):
    """
    Original: Atom Search Optimization (WDO)
        https://doi.org/10.1016/j.knosys.2018.08.030
        https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2      # Velocity
    ID_M = 3        # Mass of atom

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, alpha=50, beta=0.2):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha                  # Depth weight
        self.beta = beta                    # Multiplier weight

    def _create_solution__(self, minmax=0):
        pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=pos)
        velocity = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        mass = 0.0
        return [pos, fitness, velocity, mass]

    def _update_mass__(self, population):
        pop = sorted(population, key=lambda it: it[self.ID_FIT])
        best_fit = pop[0][self.ID_FIT]
        worst_fit = pop[-1][self.ID_FIT]
        sum_fit = sum([item[self.ID_FIT] for item in pop])
        for it in population:
            it[self.ID_M] = exp( (it[self.ID_FIT] - best_fit)/(worst_fit - best_fit) ) / sum_fit
        return population

    def _find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * sin((iteration+1) / self.epoch * pi / 2)
        rsmax = 1.24
        if radius/average_dist < rsmin:
            rs = rsmin
        else:
            if radius/average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs)**(-13) - 6 * (-rs)**(-7))
        return potential

    def _acceleration__(self, population, g_best, iteration):
        eps = 2**(-52)
        pop = self._update_mass__(population)

        G = exp(-20.0 * (iteration+1) / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5) + 1
        k_best_pop = deepcopy(sorted(pop, key=lambda it: it[self.ID_M], reverse=True)[:k_best])
        k_best_pos = [item[self.ID_POS] for item in k_best_pop]
        mk_average = mean(array(k_best_pos))

        acc_list = zeros((self.pop_size, self.problem_size))
        for i in range(0, self.pop_size):
            dist_average = norm(pop[i][self.ID_POS] - mk_average)
            temp = zeros((self.problem_size))

            for atom in k_best_pop:
                # calculate LJ-potential
                radius = norm(pop[i][self.ID_POS]-atom[self.ID_POS])
                potential = self._find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * uniform(0, 1, self.problem_size) * \
                    ((atom[self.ID_POS]-pop[i][self.ID_POS])/(radius + eps))
            temp = self.alpha * temp + self.beta * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            # calculate acceleration
            acc = G * temp / pop[i][self.ID_M]
            acc_list[i] = acc
        return acc_list


    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Calculate acceleration.
        atom_acc_list = self._acceleration__(pop, g_best, iteration=0)

        for epoch in range(0, self.epoch):
            # Update velocity based on random dimensions and position of global best

            for i in range(0, self.pop_size):
                velocity_rand = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                velocity = velocity_rand * pop[i][self.ID_VEL] + atom_acc_list[i]
                temp = pop[i][self.ID_POS] + velocity

                # Relocate atom out of range
                temp = where((temp < self.domain_range[0]) | (temp > self.domain_range[1]),
                                uniform(self.domain_range[0], self.domain_range[1]), temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit, pop[i][self.ID_VEL], 0.0]

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            else:
                r = randint(0, self.pop_size)
                pop[r] = deepcopy(g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

