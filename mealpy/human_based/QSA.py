#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseQSA(Optimizer):
    """
    My version of: Queuing search algorithm (QSA)
        (Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890
    Notes:
        + Remove all third loop
        + Using g_best solution in business 3 instead of random solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

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

    def _update_business_1(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT][self.ID_TAR], pop[1][self.ID_FIT][self.ID_TAR], pop[2][self.ID_FIT][self.ID_TAR]
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
            beta = np.power(current_epoch, np.power(current_epoch / self.epoch, 0.5))
            alpha = np.random.uniform(-1, 1)
            E = np.random.exponential(0.5, self.problem.n_dims)
            F1 = beta * alpha * (E * np.abs(A - pop[i][self.ID_POS])) + np.random.exponential(0.5) * (A - pop[i][self.ID_POS])
            F2 = beta * alpha * (E * np.abs(A - pop[i][self.ID_POS]))
            if case == 1:
                pos_new = A + F1
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if self.compare_agent([pos_new, fit_new], pop[i]):
                    pop[i] = [pos_new, fit_new]
                else:
                    case = 2
            else:
                pos_new = pop[i][self.ID_POS] + F2
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if self.compare_agent([pos_new, fit_new], pop[i]):
                    pop[i] = [pos_new, fit_new]
                else:
                    case = 1
        pop, _ = self.get_global_best_solution(pop)
        return pop

    def _update_business_2(self, pop=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT][self.ID_TAR], pop[1][self.ID_FIT][self.ID_TAR], pop[2][self.ID_FIT][self.ID_TAR]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        if t1 > 1.0e-005:
            cv = t1 / (t2 + t3)
        else:
            cv = 1.0 / 2
        pop_new = []
        for i in range(self.pop_size):
            if i < q1:
                A = deepcopy(A1)
            elif q1 <= i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if np.random.random() < pr[i]:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                if np.random.random() < cv:
                    X_new = pop[i][self.ID_POS] + np.random.exponential(0.5) * (pop[i1][self.ID_POS] - pop[i2][self.ID_POS])
                else:
                    X_new = pop[i][self.ID_POS] + np.random.exponential(0.5) * (A - pop[i1][self.ID_POS])
                pos_new = self.amend_position_faster(X_new)
            else:
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        pop = self.greedy_selection_population(pop, pop_new)
        pop, _ = self.get_global_best_solution(pop)
        return pop

    def _update_business_3(self, pop, g_best):
        pr = np.array([i / self.pop_size for i in range(1, self.pop_size + 1)])
        pop_new = []
        for i in range(self.pop_size):
            X_new = deepcopy(pop[i][self.ID_POS])
            id1 = np.random.choice(self.pop_size)
            temp = g_best[self.ID_POS] + np.random.exponential(0.5, self.problem.n_dims) * (pop[id1][self.ID_POS] - pop[i][self.ID_POS])
            X_new = np.where(np.random.random(self.problem.n_dims) > pr[i], temp, X_new)
            pos_new = self.amend_position_faster(X_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(pop, pop_new)
        return pop_new

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = self._update_business_1(self.pop, epoch + 1)
        pop = self._update_business_2(pop)
        self.pop = self._update_business_3(pop, self.g_best)


class OppoQSA(BaseQSA):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 4 * pop_size
        self.sort_flag = True

    def _opposition_based(self, pop=None, g_best=None):
        pop, _ = self.get_global_best_solution(pop)
        pop_new = []
        for i in range(0, self.pop_size):
            X_new = self.create_opposition_position(pop[i], g_best)
            pos_new = self.amend_position_faster(X_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        return self.greedy_selection_population(pop, pop_new)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = self._update_business_1(self.pop, epoch+1)
        pop = self._update_business_2(pop)
        pop = self._update_business_3(pop, self.g_best)
        self.pop = self._opposition_based(pop, self.g_best)


class LevyQSA(BaseQSA):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

    def _update_business_2(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0][self.ID_POS], pop[1][self.ID_POS], pop[2][self.ID_POS]
        t1, t2, t3 = pop[0][self.ID_FIT][self.ID_TAR], pop[1][self.ID_FIT][self.ID_TAR], pop[2][self.ID_FIT][self.ID_TAR]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        if t1 > 1.0e-6:
            cv = t1 / (t2 + t3)
        else:
            cv = 1 / 2
        pop_new = []
        for i in range(self.pop_size):
            if i < q1:
                A = deepcopy(A1)
            elif q1 <= i < q1 + q2:
                A = deepcopy(A2)
            else:
                A = deepcopy(A3)
            if np.random.random() < pr[i]:
                id1= np.random.choice(self.pop_size)
                if np.random.random() < cv:
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                    X_new = pop[i][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * levy_step
                else:
                    X_new = pop[i][self.ID_POS] + np.random.exponential(0.5) * (A - pop[id1][self.ID_POS])
                pos_new = self.amend_position_faster(X_new)
            else:
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pop_new.append([pos_new , None])
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(pop, pop_new)
        pop_new, _ = self.get_global_best_solution(pop_new)
        return pop_new

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = self._update_business_1(self.pop, epoch + 1)
        pop = self._update_business_2(pop, epoch + 1)
        self.pop = self._update_business_3(pop, self.g_best)


class ImprovedQSA(OppoQSA, LevyQSA):

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 4 * pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = self._update_business_1(self.pop, epoch + 1)
        pop = self._update_business_2(pop, epoch + 1)
        pop = self._update_business_3(pop, self.g_best)
        self.pop = self._opposition_based(pop, self.g_best)


class OriginalQSA(BaseQSA):
    """
    The original version of: Queuing search algorithm (QSA)
        (Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """

        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 3 * pop_size
        self.sort_flag = True

    def _update_business_3(self, pop, g_best):
        pr = [i / self.pop_size for i in range(1, self.pop_size + 1)]
        pop_new = []
        for i in range(self.pop_size):
            pos_new = deepcopy(pop[i][self.ID_POS])
            for j in range(self.problem.n_dims):
                if np.random.random() > pr[i]:
                    i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                    e = np.random.exponential(0.5)
                    X1 = pop[i1][self.ID_POS]
                    X2 = pop[i2][self.ID_POS]
                    pos_new[j] = X1[j] + e * (X2[j] - pop[i][self.ID_POS][j])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        return self.greedy_selection_population(pop, pop_new)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop = self._update_business_1(self.pop, epoch)
        pop = self._update_business_2(pop)
        self.pop = self._update_business_3(pop, self.g_best)