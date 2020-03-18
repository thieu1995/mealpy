#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import max, min, array
from copy import deepcopy
from mealpy.root import Root


class BaseGCO(Root):
    """
    Original version of: Germinal Center Optimization (GCO)
        (Germinal Center Optimization Algorithm)
    Link:
        DOI: https://doi.org/10.2991/ijcis.2018.25905179
    """
    ID_POS = 0
    ID_FIT = 1
    ID_CEC = 2      # CEll Counter
    ID_LIS = 3      # LIfe-Signal

    def __init__(self, root_paras=None, epoch=750, pop_size=100, cr=0.7, f=1.25):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size
        self.cr = cr                # Same as DE algorithm  # default: 0.7
        self.f = f                  # Same as DE algorithm  # default: 1.25

    def _create_solution__(self, minmax=0):
        solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution)
        cell_counter = 1
        life_signal = 70        # 70% to duplicate, and 30% to die
        return [solution, fitness, cell_counter, life_signal]

    def _mutate_cell__(self, pop=None, g_best=None, cell=None):
        r1, r2, r3 = choice(range(0, self.pop_size), 3, replace=False)
        temp = deepcopy(cell)
        for j in range(0, self.problem_size):
            if uniform() < self.cr:
                temp[self.ID_POS][j] = pop[r1][self.ID_POS][j] + self.f * (pop[r2][self.ID_POS][j] - pop[r3][self.ID_POS][j])
        new_fit = self._fitness_model__(temp[self.ID_POS])
        if new_fit < cell[self.ID_FIT]:
            temp[self.ID_FIT] = new_fit
            temp[self.ID_CEC] += 10
            if new_fit < g_best[self.ID_FIT]:
                g_best = deepcopy([temp[self.ID_POS], new_fit])
        return temp, g_best


    def _train__(self):
        # B-cells population
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Dark-zone process
            for i in range(0, self.pop_size):
                if uniform(0, 100) < pop[i][self.ID_LIS]:
                    pop[i][self.ID_CEC] += 1
                else:
                    pop[i][self.ID_CEC] = 1

                # Mutate process
                pop[i], g_best = self._mutate_cell__(pop, g_best, pop[i])

            ## Light-zone process
            for i in range(0, self.pop_size):
                pop[i][self.ID_CEC] = 10
                fit_list = array([item[self.ID_FIT] for item in pop])
                fit_max = max(fit_list)
                fit_min = min(fit_list)
                fit = (pop[i][self.ID_FIT] - fit_max) / (fit_min - fit_max)
                pop[i][self.ID_CEC] += 10 * fit

            ## Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
