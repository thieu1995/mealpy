#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, randint, random
from numpy import abs, cos, sin, pi
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

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
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
                ### Carnivore
                elif 1.0/3 <= rand and rand <= 2.0/3:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])    # Eq. 7
                ### Omnivore
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

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def __update_population_and_global_best__(self, pop, g_best):
        pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        current_best = deepcopy(pop[self.ID_MAX_PROB])
        if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
            g_best = deepcopy(current_best)
        return pop, g_best

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
                    ### Carnivore
                    elif 1.0/3 <= rand and rand <= 2.0/3:
                        x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])    # Eq. 7
                    ### Omnivore
                    else:
                        r2 = uniform()
                        x_t1 = pop[i][self.ID_POS] + c * (r2*(pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1-r2)*(pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                fit_t1 = self.get_fitness_position(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        pop, g_best = self.__update_population_and_global_best__(pop, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        pop, g_best = self.__update_population_and_global_best__(pop, g_best)

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

                if self.batch_idea:
                    if (i+1) % self.batch_size == 0:
                        pop, g_best = self.__update_population_and_global_best__(pop, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        pop, g_best = self.__update_population_and_global_best__(pop, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class AdaptiveAEO(Root):
    """
        This is Adaptive Artificial Ecosystem Optimization based on
            + Linear weight factor reduce from 2 to 0 through time
            + Levy_flight
            + Global best solution
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
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
            wf = 2 * (1 - (epoch + 1) / self.epoch)         # Weight factor
            a = (1.0 - epoch / self.epoch) * random()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]  # From the best produce new one

            ## Consumption
            for i in range(2, self.pop_size):  # From others left
                if uniform() < 0.5:
                    rand = uniform()
                    # Eq. 4, 5, 6
                    c = 0.5 * normal(0, 1) / abs(normal(0, 1))  # Consumption factor

                    j = randint(1, i)
                    ### Herbivore
                    if rand < 1.0 / 3:
                        x_t1 = pop[i][self.ID_POS] + wf * c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
                    ### Omnivore
                    elif 1.0 / 3 <= rand <= 2.0 / 3:
                        x_t1 = pop[i][self.ID_POS] + wf * c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
                    ### Carnivore
                    else:
                        r2 = uniform()
                        x_t1 = pop[i][self.ID_POS] + wf * c * (
                                    r2 * (pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r2) * (pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                else:
                    x_t1 = pop[i][self.ID_POS] + self.step_size_by_levy_flight(0.01, 1.5)*(pop[i][self.ID_POS] - g_best[self.ID_POS])
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
                if uniform() < 0.5:
                    x_t1 = best[self.ID_POS] + normal(0, 1, self.problem_size)*(best[self.ID_POS] - pop[i][self.ID_POS])
                else:
                    x_t1 = g_best[self.ID_POS] + self.step_size_by_levy_flight(0.01, 0.5) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
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


class ImprovedAEO(Root):
    """
    Original version of: Improved Artificial Ecosystem-based Optimization
        (Artificial ecosystem optimizer for parameters identification of proton exchange membrane fuel cells model)
    Link:
        https://doi.org/10.1016/j.ijhydene.2020.06.256
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
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
            # Eq. 19
            a = (1.0 - cos(0)*( 1.0 / cos(1 - (epoch +1) / self.epoch))) * uniform()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]

            ## Consumption
            for i in range(2, self.pop_size):
                rand = uniform()
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor

                j = randint(1, i)
                ### Herbivore
                if rand < 1.0 / 3:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
                ### Carnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                    x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
                ### Omnivore
                else:
                    r2 = uniform()
                    x_t1 = pop[i][self.ID_POS] + c * (r2 * (pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r2) * (pop[i][self.ID_POS] - pop[j][self.ID_POS]))
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
                x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[i][self.ID_POS])
                if uniform() < 0.5:
                    beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                    x_r = pop[randint(0, self.pop_size)][self.ID_POS]
                    if uniform() < 0.5:
                        x_new = beta * x_r + (1 - beta) * pop[i][self.ID_POS]
                    else:
                        x_new = beta * pop[i][self.ID_POS] + (1 - beta) * x_r
                else:
                    best[self.ID_POS] = best[self.ID_POS] + normal() * best[self.ID_POS]
                x_new = self.amend_position_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop_new[i] = [x_new, fit_new]

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


class EnhancedAEO(Root):
    """
    Original version of: Enhanced Artificial Ecosystem-Based Optimization
        (An Enhanced Artificial Ecosystem-Based Optimization for Optimal Allocation of Multiple Distributed Generations)
    Link:
        https://doi.org/10.1109/ACCESS.2020.3027654
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
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
            # Eq. 13
            a = 2* (1 - (epoch+1) / self.epoch)
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]

            ## Consumption
            for i in range(2, self.pop_size):
                rand = uniform()
                old_position = pop[i][self.ID_POS]
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor

                r3 = 2*pi*random()
                r4 = random()

                j = randint(1, i)
                ### Herbivore
                if rand <= 1.0 / 3:  # Eq. 15
                    if r4 <= 0.5:
                        x_t1 = old_position + sin(r3) * c * (old_position - pop[0][self.ID_POS])
                    else:
                        x_t1 = old_position + cos(r3) * c * (old_position - pop[0][self.ID_POS])
                ### Carnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 16
                    if r4 <= 0.5:
                        x_t1 = old_position + sin(r3) * c * (old_position - pop[j][self.ID_POS])
                    else:
                        x_t1 = old_position + cos(r3) * c * (old_position - pop[j][self.ID_POS])
                ### Omnivore
                else:               # Eq. 17
                    r5 = random()
                    if r4 <= 0.5:
                        x_t1 = old_position + sin(r5) * c * (r5 * (old_position - pop[0][self.ID_POS]) + (1 - r5) * (old_position - pop[j][self.ID_POS]))
                    else:
                        x_t1 = old_position + cos(r5) * c * (r5 * (old_position - pop[0][self.ID_POS]) + (1 - r5) * (old_position - pop[j][self.ID_POS]))
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
                x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[i][self.ID_POS])
                if uniform() < 0.5:
                    beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                    x_r = pop[randint(0, self.pop_size)][self.ID_POS]
                    if uniform() < 0.5:
                        x_new = beta * x_r + (1 - beta) * pop[i][self.ID_POS]
                    else:
                        x_new = beta * pop[i][self.ID_POS] + (1 - beta) * x_r
                else:
                    best[self.ID_POS] = best[self.ID_POS] + normal() * best[self.ID_POS]
                x_new = self.amend_position_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop_new[i] = [x_new, fit_new]

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


class ModifiedAEO(Root):
    """
    Original version of: Modified Artificial Ecosystem-Based Optimization
        (Effective Parameter Extraction of Different Polymer Electrolyte Membrane Fuel Cell Stack Models Using a
            Modified Artificial Ecosystem Optimization Algorithm)
    Link:
        https://doi.org/10.1109/ACCESS.2020.2973351
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
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
            # Eq. 22
            H = 2 * (1 - (epoch + 1) / self.epoch)
            a = (1 - (epoch+1) / self.epoch) * random()
            x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
            fit = self.get_fitness_position(x1)
            pop_new[0] = [x1, fit]

            ## Consumption
            for i in range(2, self.pop_size):
                rand = uniform()
                old_position = pop[i][self.ID_POS]
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor

                r3 = 2 * pi * random()
                r4 = random()

                j = randint(1, i)
                ### Herbivore
                if rand <= 1.0 / 3:                         # Eq. 23
                    x_t1 = old_position + H * c * (old_position - pop[0][self.ID_POS])
                ### Carnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:   # Eq. 24
                    x_t1 = old_position + H * c * (old_position - pop[j][self.ID_POS])
                ### Omnivore
                else:                                       # Eq. 25
                    r5 = random()
                    x_t1 = old_position + H * c * (r5 * (old_position - pop[0][self.ID_POS]) + (1 - r5) * (old_position - pop[j][self.ID_POS]))
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                pop_new[i] = [x_t1, fit_t1]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = deepcopy(pop_new[i])

            ## find current best used in decomposition
            best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

            ## Decomposition
            for i in range(0, self.pop_size):
                r3 = uniform()
                d = 3 * normal(0, 1)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[i][self.ID_POS])
                if uniform() < 0.5:
                    beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                    x_r = pop[randint(0, self.pop_size)][self.ID_POS]
                    if uniform() < 0.5:
                        x_new = beta * x_r + (1 - beta) * pop[i][self.ID_POS]
                    else:
                        x_new = beta * pop[i][self.ID_POS] + (1 - beta) * x_r
                else:
                    best[self.ID_POS] = best[self.ID_POS] + normal() * best[self.ID_POS]
                x_new = self.amend_position_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop_new[i] = [x_new, fit_new]

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

