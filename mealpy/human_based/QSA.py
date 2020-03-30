#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, exponential, random, normal
from numpy import power, sin, pi, abs
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseQSA(Root):
    """
    The original version of: Queuing search algorithm
        (Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems)
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _calculate_queue_length__(self, t1, t2, t3):
        """
        calculate length of each queue based on  t1, t2,t3
        """
        # t1 = t1 * 1.0e+100
        # t2 = t2 * 1.0e+100
        # t3 = t3 * 1.0e+100
        # print("t1",t1)
        if t1 > 1.0e-6:
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
            n3 = (1 / t3) / ((1 / t1) + (1 / t2) + (1 / t3))
        else:
            n1 = 1 / 3
            n2 = 1 / 3
            n3 = 1 / 3
            #print("1")
        q1 = int(n1 * self.pop_size)
        q2 = int(n2 * self.pop_size)
        q3 = self.pop_size - q1 - q2
        return q1, q2, q3

    def _update_bussiness_1__(self, pop, current_iter, max_iter):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT], pop[1][self.ID_FIT], pop[2][self.ID_FIT]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        case = None
        for i in range(self.pop_size):
            if i < q1:
                if i == 0:
                    case = 1
                A = deepcopy(A1)
            elif i >= q1 and i < q1 + q2:
                if i == q1:
                    case = 1
                A = deepcopy(A2)
            else:
                if i == q1 + q2:
                    case = 1
                A = deepcopy(A3)
            beta = power(current_iter, power(current_iter / max_iter, 0.5))
            alpha = uniform(-1, 1)
            E = exponential(0.5, self.problem_size)
            e = exponential(0.5)
            F1 = beta * alpha * (E * abs(A - pop[i][self.ID_POS])) + e * A - e * pop[i][self.ID_POS]
            F2 = beta * alpha * (E * abs(A - pop[i][self.ID_POS]))
            if case == 1:
                X_new = A + F1
                new_fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if new_fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, new_fit]
                    case = 1
                else:
                    case = 2
            else:
                X_new = pop[i][self.ID_POS] + F2
                new_fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if new_fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, new_fit]
                    case = 2
                else:
                    case = 1
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def _update_bussiness_2__(self, pop=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT], pop[1][self.ID_FIT], pop[2][self.ID_FIT]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        if t1 > 1.0e-005:
            cv = t1 / (t2 + t3)
        else:
            cv = 1.0 / 2
        for i in range(self.pop_size):
            if i < q1:
                A = deepcopy(A1)
            elif i >= q1 and i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if random() < pr[i]:
                i1, i2 = choice(self.pop_size, 2, replace=False)
                X1 = pop[i1][self.ID_POS]
                X2 = pop[i2][self.ID_POS]
                e = exponential(0.5)
                F1 = e * (X1 - X2)
                F2 = e * (A - X1)
                if random() < cv:
                    X_new = pop[i][self.ID_POS] + F1
                    fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                else:
                    X_new = pop[i][self.ID_POS] + F2
                    fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, fit]
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def _update_bussiness_3__(self, pop):
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        for i in range(self.pop_size):
            X_new = deepcopy(pop[i][self.ID_POS])
            for j in range(self.problem_size):
                if random() > pr[i]:
                    i1, i2 = choice(self.pop_size, 2, replace=False)
                    e = exponential(0.5)
                    X1 = pop[i1][self.ID_POS]
                    X2 = pop[i2][self.ID_POS]
                    X_new[j] = X1[j] + e * (X2[j] - pop[i][self.ID_POS][j])
            fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
            if fit < pop[i][self.ID_FIT]:
                pop[i] = [X_new, fit]
        return pop

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        for epoch in range(self.epoch):
            pop = self._update_bussiness_1__(pop, epoch, self.epoch)
            pop = self._update_bussiness_2__(pop)
            pop = self._update_bussiness_3__(pop)

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OppoQSA(BaseQSA):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseQSA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _opposition_based__(self, pop=None, g_best=None):
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        a = 0.3
        num_change = int(self.pop_size * a)
        for i in range(self.pop_size - num_change, self.pop_size):
            X_new = self._create_opposition_solution__(pop[i][self.ID_POS], g_best[self.ID_POS])
            fitness = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
            if fitness < pop[i][self.ID_FIT]:
                pop[i] = [X_new, fitness]
        return pop

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            pop = self._update_bussiness_1__(pop, epoch, self.epoch)
            pop = self._update_bussiness_2__(pop)
            pop = self._update_bussiness_3__(pop)
            pop = self._opposition_based__(pop, g_best)

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyQSA(BaseQSA):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseQSA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _update_bussiness_2__(self, pop=None, current_iter=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT], pop[1][self.ID_FIT], pop[2][self.ID_FIT]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        if t1 > 1.0e-6:
            cv = t1 / (t2 + t3)
        else:
            cv = 1 / 2
        for i in range(self.pop_size):
            if i < q1:
                A = deepcopy(A1)
            elif i >= q1 and i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if random() < pr[i]:
                i1, i2 = choice(self.pop_size, 2, replace=False)
                X1 = pop[i1][self.ID_POS]
                X2 = pop[i2][self.ID_POS]
                e = exponential(0.5)
                F1 = e * (X1 - X2)
                F2 = e * (A - X1)
                if random() < cv:
                    X_new = self._levy_flight__(current_iter, pop[i][self.ID_POS], A)
                    fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                else:
                    X_new = pop[i][self.ID_POS] + F2
                    fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, fit]
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        for epoch in range(self.epoch):
            pop = self._update_bussiness_1__(pop, epoch, self.epoch)
            pop = self._update_bussiness_2__(pop, epoch)
            pop = self._update_bussiness_3__(pop)

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedQSA(OppoQSA, LevyQSA):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        OppoQSA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)
        LevyQSA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            pop = self._update_bussiness_1__(pop, epoch, self.epoch)
            pop = self._update_bussiness_2__(pop, epoch)
            pop = self._update_bussiness_3__(pop)
            pop = self._opposition_based__(pop, g_best)

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
