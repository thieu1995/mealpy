#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:24, 09/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, choice
from numpy import array, zeros, where, argmin, mean
from copy import deepcopy
from mealpy.root import Root


class OriginalCHIO(Root):
    """
    The original version of: Coronavirus Herd Immunity Optimization (CHIO)
        (Coronavirus herd immunity Optimization)
    Link:
        https://link.springer.com/article/10.1007/s00521-020-05296-6
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, brr=0.06, max_age=150, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.brr = brr              # Basic reproduction rate
        self.max_age = max_age      # Maximum infected cases age

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        immunity_type_list = randint(0, 3, self.pop_size)           # Randint [0, 1, 2]
        age_list = zeros(self.pop_size)                             # Control the age of each position
        finished = False

        for epoch in range(self.epoch):

            for i in range(0, self.pop_size):
                is_corona = False
                pos_new = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    rand = uniform()
                    if rand < (1.0/3)*self.brr:
                        idx_candidates = where(immunity_type_list == 1)     # Infected list
                        if idx_candidates[0].size == 0:
                            finished = True
                            print("Epoch: {}, i: {}, immunity_list: {}".format(epoch, i, immunity_type_list))
                            break
                        idx_selected = choice(idx_candidates[0])
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                        is_corona = True
                    elif (1.0/3)*self.brr <= rand < (2.0/3)*self.brr:
                        idx_candidates = where(immunity_type_list == 0)     # Susceptible list
                        idx_selected = choice(idx_candidates[0])
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                    elif (2.0/3)*self.brr <= rand < self.brr:
                        idx_candidates = where(immunity_type_list == 2)     # Immunity list
                        fit_list = array([pop[item][self.ID_FIT] for item in idx_candidates[0]])
                        idx_selected = idx_candidates[0][argmin(fit_list)]     # Found the index of best fitness
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                if finished:
                    break
                # Step 4: Update herd immunity population
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                else:
                    age_list[i] += 1

                ## Calculate immunity mean of population
                fit_list = array([item[self.ID_FIT] for item in pop])
                delta_fx = mean(fit_list)
                if (fit_new < delta_fx) and (immunity_type_list[i] == 0) and is_corona:
                    immunity_type_list[i] = 1
                    age_list[i] = 1
                if (fit_new > delta_fx) and (immunity_type_list[i] == 1):
                    immunity_type_list[i] = 2
                    age_list[i] = 0

                # Step 5: Fatality condition
                if (age_list[i] >= self.max_age) and (immunity_type_list[i] == 1):
                    solution_new = self.create_solution()
                    pop[i] = solution_new
                    immunity_type_list[i] = 0
                    age_list[i] = 0
            if finished:
                break
            # Needed to update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseCHIO(Root):
    """
        My version of: Coronavirus Herd Immunity Optimization (CHIO)
            (Coronavirus herd immunity Optimization)
        Noted:
            changed:
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, brr=0.06, max_age=150, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.brr = brr  # Basic reproduction rate
        self.max_age = max_age  # Maximum infected cases age

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        immunity_type_list = randint(0, 3, self.pop_size)   # Randint [0, 1, 2]
        age_list = zeros(self.pop_size)                     # Control the age of each position

        for epoch in range(self.epoch):

            for i in range(0, self.pop_size):
                is_corona = False
                pos_new = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    rand = uniform()
                    if rand < (1.0 / 3) * self.brr:
                        idx_candidates = where(immunity_type_list == 1)         # Infected list
                        if idx_candidates[0].size == 0:
                            rand_choice = choice(range(0, self.pop_size), int(0.33*self.pop_size), replace=False)
                            immunity_type_list[rand_choice] = 1
                            idx_candidates = where(immunity_type_list == 1)
                        idx_selected = choice(idx_candidates[0])
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                        is_corona = True
                    elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                        idx_candidates = where(immunity_type_list == 0)         # Susceptible list
                        if idx_candidates[0].size == 0:
                            rand_choice = choice(range(0, self.pop_size), int(0.33 * self.pop_size), replace=False)
                            immunity_type_list[rand_choice] = 0
                            idx_candidates = where(immunity_type_list == 0)
                        idx_selected = choice(idx_candidates[0])
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                    elif (2.0 / 3) * self.brr <= rand < self.brr:
                        idx_candidates = where(immunity_type_list == 2)         # Immunity list
                        fit_list = array([pop[item][self.ID_FIT] for item in idx_candidates[0]])
                        idx_selected = idx_candidates[0][argmin(fit_list)]  # Found the index of best fitness
                        pos_new[j] = pop[i][self.ID_POS][j] + uniform() * (pop[i][self.ID_POS][j] - pop[idx_selected][self.ID_POS][j])
                # Step 4: Update herd immunity population
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                else:
                    age_list[i] += 1
                ## Calculate immunity mean of population
                fit_list = array([item[self.ID_FIT] for item in pop])
                delta_fx = mean(fit_list)
                if (fit_new < delta_fx) and (immunity_type_list[i] == 0) and is_corona:
                    immunity_type_list[i] = 1
                    age_list[i] = 1
                if (fit_new > delta_fx) and (immunity_type_list[i] == 1):
                    immunity_type_list[i] = 2
                    age_list[i] = 0

                # Step 5: Fatality condition
                if (age_list[i] >= self.max_age) and (immunity_type_list[i] == 1):
                    solution_new = self.create_solution()
                    pop[i] = solution_new
                    immunity_type_list[i] = 0
                    age_list[i] = 0
            # Needed to update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
