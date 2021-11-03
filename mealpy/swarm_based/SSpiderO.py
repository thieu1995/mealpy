#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:00, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseSSpiderO(Optimizer):
    """
    The original version of: Social Spider Optimization (SSpiderO)
        (Social Spider Optimization Algorithm: Modifications, Applications, and Perspectives)
    Link:
        https://www.hindawi.com/journals/mpe/2018/6843923/
    """
    ID_POS = 0
    ID_FIT = 1
    ID_WEI = 2

    def __init__(self, problem, epoch=10000, pop_size=100, fp=(0.65, 0.9), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fp (list): (fp_min, fp_max): Female Percent
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.fp = fp

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position)
        weight = 0.0
        return [position, fitness, weight]

    def _move_females__(self, mode=None, n_f=None, pop_females=None, pop_males=None, g_best=None, pm=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        pop = pop_females + pop_males
        # Start looking for any stronger vibration
        for i in range(0, n_f):    # Move the females
            ## Find the position s
            id_min = None
            dist_min = 99999999
            for j in range(0, self.pop_size):
                if pop[j][self.ID_WEI] > pop_females[i][self.ID_WEI]:
                    dt = np.linalg.norm(pop[j][self.ID_POS] - pop_females[i][self.ID_POS]) / scale_distance
                    if dt < dist_min and dt != 0:
                        dist_min = dt
                        id_min = j
            x_s = np.zeros(self.problem.n_dims)
            vibs = 0
            if not (id_min is None):
                vibs = 2*(pop[id_min][self.ID_WEI]*np.exp(-(np.random.uniform()*dist_min**2)))  # Vib for the shortest
                x_s = pop[id_min][self.ID_POS]

            ## Find the position b
            dtb = np.linalg.norm(g_best[self.ID_POS] - pop_females[i][self.ID_POS]) / scale_distance
            vibb = 2 * (g_best[self.ID_WEI] * np.exp(-(np.random.uniform() * dtb ** 2)))

            ## Do attraction or repulsion
            beta = np.random.uniform(0, 1, self.problem.n_dims)
            gamma = np.random.uniform(0, 1, self.problem.n_dims)
            random = 2 * pm * (np.random.uniform(0, 1, self.problem.n_dims) - 0.5)
            if np.random.uniform() >= pm:       # Do an attraction
                temp = pop_females[i][self.ID_POS] + vibs * (x_s - pop_females[i][self.ID_POS]) * beta + \
                    vibb * (g_best[self.ID_POS] - pop_females[i][self.ID_POS]) * gamma + random
            else:                               # Do a repulsion
                temp = pop_females[i][self.ID_POS] - vibs * (x_s - pop_females[i][self.ID_POS]) * beta - \
                       vibb * (g_best[self.ID_POS] - pop_females[i][self.ID_POS]) * gamma + random
            temp = self.amend_position_random(temp)
            pop_females[i][self.ID_POS] = temp
            # fit = self.get_fitness_position(temp)
            # pop_females[i][self.ID_POS] = temp
            # pop_females[i][self.ID_FIT] = fit
        pop_females = self.update_fitness_population(mode, pop_females)
        return pop_females

    def _move_males__(self, mode=None, n_f=None, n_m=None, pop_females=None, pop_males=None, pm=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        my_median =np.median([it[self.ID_WEI] for it in pop_males])
        pop = pop_females + pop_males
        all_pos = np.array([it[self.ID_POS] for it in pop])
        all_wei = np.array([it[self.ID_WEI] for it in pop]).reshape((2 * self.pop_size, 1))
        mean = np.sum(all_wei * all_pos, axis=0) / np.sum(all_wei)
        for i in range(0, n_m):
            delta = 2 * np.random.uniform(0, 1, self.problem.n_dims) - 0.5
            random = 2 * pm * (np.random.uniform(0, 1, self.problem.n_dims) - 0.5)

            if pop_males[i][self.ID_WEI] >= my_median:         # Spider above the median
                # Start looking for a female with stronger vibration
                id_min = None
                dist_min = 99999999
                for j in range(0, n_f):
                    if pop_females[j][self.ID_WEI] > pop_males[i][self.ID_WEI]:
                        dt = np.linalg.norm(pop_females[j][self.ID_POS] - pop_males[i][self.ID_POS]) / scale_distance
                        if dt < dist_min and dt != 0:
                            dist_min = dt
                            id_min = j
                x_s = np.zeros(self.problem.n_dims)
                vibs = 0
                if id_min != None:
                    vibs = 2 * (pop_females[id_min][self.ID_WEI] * np.exp(-(np.random.uniform() * dist_min ** 2)))      # Vib for the shortest
                    x_s = pop_females[id_min][self.ID_POS]
                temp = pop_males[i][self.ID_POS] + vibs * (x_s - pop_males[i][self.ID_POS])*delta + random
            else:
                # Spider below median, go to weighted mean
                temp = pop_males[i][self.ID_POS] + delta * (mean - pop_males[i][self.ID_POS]) + random
            temp = self.amend_position_random(temp)
            pop_males[i][self.ID_POS] = temp
            # fit = self.get_fitness_position(temp)
            # pop_males[i][self.ID_FIT] = fit
        pop_males = self.update_fitness_population(mode, pop_males)
        return pop_males

    ### Crossover
    def _crossover__(self, mom=None, dad=None, id=0):
        child1 = np.zeros(self.problem.n_dims)
        child2 = np.zeros(self.problem.n_dims)
        if id == 0:         # arithmetic recombination
            r = np.random.uniform(0.5, 1)             # w1 = w2 when r =0.5
            child1 = np.multiply(r, mom) + np.multiply((1 - r), dad)
            child2 = np.multiply(r, dad) + np.multiply((1 - r), mom)

        elif id == 1:
            id1 = np.random.randint(1, int(self.problem.n_dims / 2))
            id2 = int(id1 + self.problem.n_dims / 2)

            child1[:id1] = mom[:id1]
            child1[id1:id2] = dad[id1:id2]
            child1[id2:] = mom[id2:]

            child2[:id1] = dad[:id1]
            child2[id1:id2] = mom[id1:id2]
            child2[id2:] = dad[id2:]
        elif id == 2:
            temp = int(self.problem.n_dims/2)
            child1[:temp] = mom[:temp]
            child1[temp:] = dad[temp:]
            child2[:temp] = dad[:temp]
            child2[temp:] = mom[temp:]

        return child1, child2

    def _mating__(self, mode, pop_females=None, pop_males=None, n_f=None, n_m=None):
        # Check whether a spider is good or not (above median)
        my_median = np.median([it[self.ID_WEI] for it in pop_males])
        pop_males_new = [pop_males[i] for i in range(n_m) if pop_males[i][self.ID_WEI] > my_median]

        # Calculate the radio
        pop = pop_females + pop_males
        all_pos = np.array([it[self.ID_POS] for it in pop])
        rad = np.max(all_pos, axis=1) - np.min(all_pos, axis=1)
        r = np.sum(rad)/(2*self.problem.n_dims)

        # Start looking if there's a good female near
        list_child = []
        couples = []
        for i in range(0, len(pop_males_new)):
            for j in range(0, n_f):
                dist = np.linalg.norm(pop_males_new[i][self.ID_POS] - pop_females[j][self.ID_POS])
                if dist < r:
                    couples.append([pop_males_new[i], pop_females[j]])
        if couples:
            n_child = len(couples)
            for k in range(n_child):
                child1, child2 = self._crossover__(couples[k][0][self.ID_POS], couples[k][1][self.ID_POS], 0)
                list_child.append([child1, None, 0.0])
                list_child.append([child2, None, 0.0])
        list_child = self.update_fitness_population(mode, list_child)
        return list_child

    def _survive__(self, pop=None, pop_child=None):
        n_child = len(pop)
        pop_child = self.get_sorted_strim_population(pop_child, n_child)
        for i in range(0, n_child):
            if self.compare_agent(pop_child[i], pop[i]):
                pop[i] = pop_child[i].copy()
        return pop

    def _recalculate_weights__(self, pop=None):
        fit_total, fit_best, fit_worst = self.get_special_fitness(pop)
        # This will automatically save weight in pop_males and pop_females because this is python.
        for i in range(len(pop)):
            pop[i][self.ID_WEI] = 0.001 + (pop[i][self.ID_FIT][self.ID_TAR] - fit_worst) / (fit_best - fit_worst)
        return pop

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        self.termination_start()
        fp = self.fp[0] + (self.fp[1] - self.fp[0]) * np.random.uniform()  # Female Aleatory Percent
        n_f = int(self.pop_size * fp)  # number of female
        n_m = self.pop_size - n_f  # number of male
        # Probabilities of attraction or repulsion Proper tuning for better results
        p_m = (self.epoch + 1 - np.array(range(1, self.epoch + 1))) / (self.epoch + 1)

        pop_males = self.create_population(mode, n_m)
        pop_females = self.create_population(mode, n_f)
        pop = pop_females + pop_males
        pop = self._recalculate_weights__(pop)
        _, g_best = self.get_global_best_solution(pop)
        self.history.save_initial_best(g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ### Movement of spiders
            pop_females = self._move_females__(mode, n_f, pop_females, pop_males, g_best, p_m[epoch])
            pop_males = self._move_males__(mode, n_f, n_m, pop_females, pop_males, p_m[epoch])

            # Recalculate weights
            pop = pop_females + pop_males
            pop = self._recalculate_weights__(pop)

            # Mating Operator
            pop_child = self._mating__(mode, pop_females, pop_males, n_f, n_m)
            pop = self._survive__(pop, pop_child)
            pop = self._recalculate_weights__(pop)

            # update global best position
            _, g_best = self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

