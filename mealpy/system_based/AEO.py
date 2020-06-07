#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, randint
from numpy import abs
from copy import deepcopy
from mealpy.root import Root


class OriginalAEO(Root):
    """
    Original version of: Artificial Ecosystem-based Optimization
        (Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm)
    Link:
        https://doi.org/10.1007/s00521-019-04452-x
        https://www.mathworks.com/matlabcentral/fileexchange/72685-artificial-ecosystem-based-optimization-aeo
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        # Sorted population in the descending order of the function fitness value
        pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        g_best = deepcopy(pop[self.ID_MAX_PROB])
        pop_new = deepcopy(pop)
        for epoch in range(self.epoch):
            ## Production
            # Eq. 2, 3, 1
            a = (1.0 - epoch / self.epoch) * uniform()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]

            ## Consumption
            for i in range(2, self.pop_size):
                rand = uniform()
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)      # Consumption factor

                j = randint(1, i)
                ### Herbivore
                if rand < 1.0/3:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])    # Eq. 6
                ### Omnivore
                elif 1.0/3 <= rand and rand <= 2.0/3:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])    # Eq. 7
                ### Carnivore
                else:
                    r2 = uniform()
                    x_t1 = pop[i][self.ID_POS] + c * (r2*(pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1-r2)*(pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                pop_new[i] = [x_t1, fit_t1]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy(pop_new[i])

            ## find current best used in decomposition
            best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

            ## Decomposition
            ### Eq. 10, 11, 12, 9
            for i in range(0, self.pop_size):
                r3 = uniform()
                d = 3 * normal(0, 1)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                x_t1 = best[self.ID_POS] + d * (e*best[self.ID_POS] - h*pop[i][self.ID_POS])
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                pop_new[i] = [x_t1, fit_t1]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy(pop_new[i])

            pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
            current_best = deepcopy(pop[self.ID_MAX_PROB])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseAEO(Root):
    """
    This is my modified version of: Artificial Ecosystem-based Optimization
        (Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm)

        + Same results and sometime better than original version.
        + Original version move the population at the same time. My version move after each position move.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        # Sorted population in the descending order of the function fitness value
        pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        g_best = deepcopy(pop[self.ID_MAX_PROB])

        for epoch in range(self.epoch):
            ## Production
            # Eq. 2, 3, 1
            a = (1.0 - epoch / self.epoch) * uniform()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop[0] = [x1, fit]

            ## Consumption
            for i in range(1, self.pop_size):
                rand = uniform()
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)      # Consumption factor

                if i == 1:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
                else:
                    j = randint(1, i)
                    ### Herbivore
                    if rand < 1.0/3:
                        x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])    # Eq. 6
                    ### Omnivore
                    elif 1.0/3 <= rand and rand <= 2.0/3:
                        x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])    # Eq. 7
                    ### Carnivore
                    else:
                        r2 = uniform()
                        x_t1 = pop[i][self.ID_POS] + c * (r2*(pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1-r2)*(pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                fit_t1 = self.get_fitness_position(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

                if i % self.batch_size:
                    ## Update global best base on batch_size
                    pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
                    current_best = deepcopy(pop[self.ID_MAX_PROB])
                    if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(current_best)

            ## Decomposition
            ### Eq. 10, 11, 12, 9
            for i in range(0, self.pop_size):
                r3 = uniform()
                d = 3 * normal(0, 1)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                x_t1 = pop[self.ID_MAX_PROB][self.ID_POS] + d * (e*pop[self.ID_MAX_PROB][self.ID_POS] - h*pop[i][self.ID_POS])
                fit_t1 = self.get_fitness_position(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

                if i % self.batch_size:
                    pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
                    current_best = deepcopy(pop[self.ID_MAX_PROB])
                    if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(current_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyAEO(Root):
    """
        This is modified version of: Artificial Ecosystem-based Optimization based on Levy_flight
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=750, pop_size=100):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        # Sorted population in the descending order of the function fitness value
        pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        g_best = deepcopy(pop[self.ID_MAX_PROB])
        pop_new = deepcopy(pop)
        for epoch in range(self.epoch):
            ## Production
            # Eq. 2, 3, 1
            a = (1.0 - epoch / self.epoch) * uniform()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]              # From the best produce new one

            ## Consumption
            for i in range(2, self.pop_size):   # From others left
                if uniform() < 0.5:
                    rand = uniform()
                    # Eq. 4, 5, 6
                    c = 0.5 * normal(0, 1) / abs(normal(0, 1))  # Consumption factor

                    j = randint(1, i)
                    ### Herbivore
                    if rand < 1.0/3:
                        x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
                    ### Omnivore
                    elif 1.0/3 <= rand <= 2.0/3:
                        x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
                    ### Carnivore
                    else:
                        r2 = uniform()
                        x_t1 = pop[i][self.ID_POS] + c * (r2 * (pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r2) * (pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                else:
                    x_t1 = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                pop_new[i] = [x_t1, fit_t1]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy(pop_new[i])

            ## find current best used in decomposition
            best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

            ## Decomposition
            ### Eq. 10, 11, 12, 9
            for i in range(0, self.pop_size):
                u = normal(0, 1)
                r3 = uniform()
                d = 3 * u
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                x_t1 = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[i][self.ID_POS])
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                pop_new[i] = [x_t1, fit_t1]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy(pop_new[i])

            pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
            current_best = deepcopy(pop[self.ID_MAX_PROB])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train