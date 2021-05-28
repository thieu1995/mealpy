#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, exponential, random
from numpy import power, abs, array, where
from copy import deepcopy
from mealpy.root import Root


class BaseQSA(Root):
    """
    My version of: Queuing search algorithm (QSA)
        (Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890
    Notes:
        + Remove all third loop
        + Using g_best solution in business 3 instead of random solution
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def _calculate_queue_length__(self, t1, t2, t3):
        """ Calculate length of each queue based on  t1, t2,t3
                t1 = t1 * 1.0e+100
                t2 = t2 * 1.0e+100
                t3 = t3 * 1.0e+100
        """
        if t1 > 1.0e-6:
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
        else:
            n1 = 1.0 / 3
            n2 = 1.0 / 3
        q1 = int(n1 * self.pop_size)
        q2 = int(n2 * self.pop_size)
        q3 = self.pop_size - q1 - q2
        return q1, q2, q3

    def _update_business_1__(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT], pop[1][self.ID_FIT], pop[2][self.ID_FIT]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        case = None
        for i in range(self.pop_size):
            if i < q1:
                if i == 0:
                    case = 1
                A = deepcopy(A1)
            elif q1 <= i < q1 + q2:
                if i == q1:
                    case = 1
                A = deepcopy(A2)
            else:
                if i == q1 + q2:
                    case = 1
                A = deepcopy(A3)
            beta = power(current_epoch, power(current_epoch / self.epoch, 0.5))
            alpha = uniform(-1, 1)
            E = exponential(0.5, self.problem_size)
            F1 = beta * alpha * (E * abs(A - pop[i][self.ID_POS])) + exponential(0.5) * (A - pop[i][self.ID_POS])
            F2 = beta * alpha * (E * abs(A - pop[i][self.ID_POS]))
            if case == 1:
                X_new = A + F1
                new_fit = self.get_fitness_position(X_new)
                if new_fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, new_fit]
                    case = 1
                else:
                    case = 2
            else:
                X_new = pop[i][self.ID_POS] + F2
                new_fit = self.get_fitness_position(X_new)
                if new_fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, new_fit]
                    case = 2
                else:
                    case = 1
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def _update_business_2__(self, pop=None):
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
            elif q1 <= i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if random() < pr[i]:
                i1, i2 = choice(self.pop_size, 2, replace=False)
                if random() < cv:
                    X_new = pop[i][self.ID_POS] + exponential(0.5) * (pop[i1][self.ID_POS] - pop[i2][self.ID_POS])
                else:
                    X_new = pop[i][self.ID_POS] + exponential(0.5) * (A - pop[i1][self.ID_POS])
                fit = self.get_fitness_position(X_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, fit]
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def _update_business_3__(self, pop, g_best):
        pr = array([i / self.pop_size for i in range(1, self.pop_size + 1)])
        for i in range(self.pop_size):
            X_new = deepcopy(pop[i][self.ID_POS])
            id1= choice(self.pop_size)
            temp = g_best[self.ID_POS] + exponential(0.5, self.problem_size) * (pop[id1][self.ID_POS] - pop[i][self.ID_POS])
            X_new = where(random(self.problem_size) > pr[i], temp, X_new)
            fit = self.get_fitness_position(X_new)
            if fit < pop[i][self.ID_FIT]:
                pop[i] = [X_new, fit]
        return pop

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        for epoch in range(self.epoch):
            pop = self._update_business_1__(pop, epoch+1)
            pop = self._update_business_2__(pop)
            pop = self._update_business_3__(pop, g_best)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OppoQSA(BaseQSA):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseQSA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def _opposition_based__(self, pop=None, g_best=None):
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        for i in range(0, self.pop_size):
            X_new = self.create_opposition_position(pop[i][self.ID_POS], g_best[self.ID_POS])
            fitness = self.get_fitness_position(X_new)
            if fitness < pop[i][self.ID_FIT]:
                pop[i] = [X_new, fitness]
        return pop

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            pop = self._update_business_1__(pop, epoch)
            pop = self._update_business_2__(pop)
            pop = self._update_business_3__(pop, g_best)
            pop = self._opposition_based__(pop, g_best)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyQSA(BaseQSA):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseQSA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def _update_business_2__(self, pop=None, current_epoch=None):
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
            elif q1 <= i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if random() < pr[i]:
                id1= choice(self.pop_size)
                if random() < cv:
                    X_new = self.levy_flight(current_epoch, pop[i][self.ID_POS], A)
                else:
                    X_new = pop[i][self.ID_POS] + exponential(0.5) * (A - pop[id1][self.ID_POS])
                fit = self.get_fitness_position(X_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_new, fit]
        return sorted(pop, key=lambda item: item[self.ID_FIT])

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        for epoch in range(self.epoch):
            pop = self._update_business_1__(pop, epoch+1)
            pop = self._update_business_2__(pop, epoch+1)
            pop = self._update_business_3__(pop, g_best)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedQSA(OppoQSA, LevyQSA):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseQSA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)
        LevyQSA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            pop = self._update_business_1__(pop, epoch+1)
            pop = self._update_business_2__(pop, epoch+1)
            pop = self._update_business_3__(pop, g_best)
            pop = self._opposition_based__(pop, g_best)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalQSA(BaseQSA):
    """
    The original version of: Queuing search algorithm (QSA)
        (Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseQSA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def _update_business_3__(self, pop, g_best):
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
            fit = self.get_fitness_position(position=X_new, minmax=self.ID_MIN_PROB)
            if fit < pop[i][self.ID_FIT]:
                pop[i] = [X_new, fit]
        return pop

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        for epoch in range(self.epoch):
            pop = self._update_business_1__(pop, epoch)
            pop = self._update_business_2__(pop)
            pop = self._update_business_3__(pop, g_best)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
