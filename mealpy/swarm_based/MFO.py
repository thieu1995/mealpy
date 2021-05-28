#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp, cos, pi, ones, where
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseMFO(Root):
    """
        My modified version of: Moth-Flame Optimization (MFO)
            (Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm)
        Notes:
            + Changed the flow of algorithm
            + Update the old solution
            + Remove third loop for faster
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop_moths = [self.create_solution() for _ in range(self.pop_size)]
        # Update the position best flame obtained so far
        pop_flames, g_best = self.get_sorted_pop_and_global_best_solution(pop_moths, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Number of flames Eq.(3.14) in the paper (linearly decreased)
            num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

            # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + (epoch + 1) * ((-1) / self.epoch)

            for i in range(self.pop_size):

                #   D in Eq.(3.13)
                distance_to_flame = abs(pop_flames[i][self.ID_POS] - pop_moths[i][self.ID_POS])
                t = (a - 1) * uniform(0, 1, self.problem_size) + 1
                b = 1

                # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
                temp_1 = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + pop_flames[i][self.ID_POS]

                # Update the position of the moth with respect to one flame Eq.(3.12).
                ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
                temp_2 = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + g_best[self.ID_POS]

                list_idx = i * ones(self.problem_size)
                pos_new = where(list_idx < num_flame, temp_1, temp_2)

                ## This is the way I make this algorithm working. I tried to run matlab code with large dimension and it will not convergence.
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop_moths[i][self.ID_FIT]:
                    pop_moths[i] = [pos_new, fit_new]

            # Update the global best flame
            pop_flames = pop_flames + pop_moths
            pop_flames, g_best = self.update_sorted_population_and_global_best_solution(pop_flames, self.ID_MIN_PROB, g_best)
            pop_flames = pop_flames[:self.pop_size]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalMFO(BaseMFO):
    """
        The original version of: Moth-flame optimization (MFO)
            (Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm)
        Link:
            + https://www.mathworks.com/matlabcentral/fileexchange/52269-moth-flame-optimization-mfo-algorithm?s_tid=FX_rc1_behav
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseMFO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop_moths = [self.create_solution() for _ in range(self.pop_size)]
        # Update the position best flame obtained so far
        pop_flames, g_best= self.get_sorted_pop_and_global_best_solution(pop_moths, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Number of flames Eq.(3.14) in the paper (linearly decreased)
            num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

            # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + (epoch + 1) * ((-1) / self.epoch)

            for i in range(self.pop_size):
                temp = deepcopy(pop_moths[i][self.ID_POS])
                for j in range(self.problem_size):
                    #   D in Eq.(3.13)
                    distance_to_flame = abs(pop_flames[i][self.ID_POS][j] - pop_moths[i][self.ID_POS][j])
                    t = (a - 1) * uniform() + 1
                    b = 1
                    if i <= num_flame:  # Update the position of the moth with respect to its corresponding flame
                        # Eq.(3.12)
                        temp[j] = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + pop_flames[i][self.ID_POS][j]
                    else:   # Update the position of the moth with respect to one flame
                        # Eq.(3.12).
                        ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
                        temp[j] = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + pop_flames[num_flame][self.ID_POS][j]

                fit = self.get_fitness_position(temp)
                pop_moths[i] = [temp, fit]

            # Update the global best flame
            pop_flames = pop_flames + pop_moths
            pop_flames, g_best = self.update_sorted_population_and_global_best_solution(pop_flames, self.ID_MIN_PROB, g_best)
            pop_flames = pop_flames[:self.pop_size]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

