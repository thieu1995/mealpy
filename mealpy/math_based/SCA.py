#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import pi, sin, abs, cos, where
from copy import deepcopy
from mealpy.root import Root


class BaseSCA(Root):
    """
    The original version of: Sine Cosine Algorithm (SCA)
        A Sine Cosine Algorithm for solving optimization problems
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
    Notes:
        + I changed the flow as well as the equations.
        + Removed third loop for faster computational time
        + Batch size ideas
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update the position of solutions with respect to destination
            for i in range(self.pop_size):              # i-th position
                # Eq 3.4, r1 decreases linearly from a to 0
                a = 2.0
                r1 = a - (epoch + 1) * (a / self.epoch)
                # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
                r2 = 2 * pi * uniform(0, 1, self.problem_size)
                r3 = 2 * uniform(0, 1, self.problem_size)
                # Eq. 3.3, 3.1 and 3.2
                pos_new1 = pop[i][self.ID_POS] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new2 = pop[i][self.ID_POS] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
                # Check the bound
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:               # My improved part
                    pop[i] = [pos_new, fit]

                ## Update the global best
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) * self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSCA(Root):
    """
    Original version of: Sine Cosine Algorithm (SCA)
        A Sine Cosine Algorithm for solving optimization problems
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update the position of solutions with respect to destination
            for i in range(self.pop_size):  # i-th position
                # Eq 3.4, r1 decreases linearly from a to 0
                a = 2.0
                r1 = a - (epoch + 1) * (a / self.epoch)
                pos_new = deepcopy(pop[i][self.ID_POS])
                for j in range(self.problem_size):  # j-th dimension
                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = 2 * pi * uniform()
                    r3 = 2 * uniform()
                    r4 = uniform()
                    # Eq. 3.3, 3.1 and 3.2
                    if r4 < 0.5:
                        pos_new[j] = pos_new[j] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS][j] - pos_new[j])
                    else:
                        pos_new[j] = pos_new[j] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS][j] - pos_new[j])
                # Check the bound
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class FasterSCA(Root):
    """
        A Sine Cosine Algorithm for solving optimization problems (SCA)
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    This is my version of SCA. The original version of SCA is not working. So I changed the flow of algorithm
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return {0: position, 1: fitness}

    def train(self):
        pop = {i: self.create_solution() for i in range(self.pop_size)}
        pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
        g_best = next(iter(pop_sorted.values()))

        for epoch in range(self.epoch):
            # Update the position of solutions with respect to destination
            for i, (idx, item) in enumerate(pop.items()):  # i-th position
                # Eq 3.4, r1 decreases linearly from a to 0
                a = 2.0
                r1 = a - (epoch + 1) * (a / self.epoch)
                # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
                r2 = 2 * pi * uniform(0, 1, self.problem_size)
                r3 = 2 * uniform(0, 1, self.problem_size)
                # Eq. 3.3, 3.1 and 3.2
                pos_new1 = pop[i][self.ID_POS] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new2 = pop[i][self.ID_POS] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
                # Check the bound
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < item[self.ID_FIT]:  # My improved part
                    pop[idx] = {0: pos_new, 1: fit}

                ## Update the global best
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
                        current_best = next(iter(pop_sorted.values()))
                        if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                            g_best = deepcopy(current_best)
                else:
                    if (i + 1) * self.pop_size == 0:
                        pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
                        current_best = next(iter(pop_sorted.values()))
                        if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                            g_best = deepcopy(current_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class FastestSCA(Root):
    """
        A Sine Cosine Algorithm for solving optimization problems (SCA)
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    This is my version of SCA. The original version of SCA is not working. So I changed the flow of algorithm
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return {0: position, 1: fitness}

    def train(self):
        pop = [self.create_solution() for i in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update the position of solutions with respect to destination
            for i in range(self.pop_size):  # i-th position
                # Eq 3.4, r1 decreases linearly from a to 0
                a = 2.0
                r1 = a - (epoch + 1) * (a / self.epoch)
                # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
                r2 = 2 * pi * uniform(0, 1, self.problem_size)
                r3 = 2 * uniform(0, 1, self.problem_size)
                # Eq. 3.3, 3.1 and 3.2
                pos_new1 = pop[i][self.ID_POS] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new2 = pop[i][self.ID_POS] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
                # Check the bound
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:  # My improved part
                    pop[i] = {0: pos_new, 1: fit}

                ## Update the global best
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) * self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

