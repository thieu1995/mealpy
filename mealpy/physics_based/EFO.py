#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
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


class BaseEFO(Optimizer):
    """
    My version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S2210650215000528
    Notes:
        + The flow of algorithm is changed like other metaheuristics.
        + Apply levy-flight for large-scale optimization problems
        + Change equations using g_best solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (): default = 0.85    Like crossover parameter in GA
            p_field (): default = 0.1     portion of population, positive field
            n_field (): default = 0.45    portion of population, negative field
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.r_rate = r_rate
        self.ps_rate = ps_rate
        self.p_field = p_field
        self.n_field = n_field
        self.phi = (1 + np.sqrt(5)) / 2  # golden ratio

    def create_child(self, idx, pop, g_best, epoch):
        r_idx1 = np.random.randint(0, int(self.pop_size * self.p_field))  # top
        r_idx2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)  # bottom
        r_idx3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))  # middle
        if np.random.uniform() < self.ps_rate:
            # new = g_best + phi* r1 * (top - middle) + r2 (top - bottom)
            # pos_new = g_best[self.ID_POS] + \
            #            phi * np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
            #            np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx2][self.ID_POS])
            # new = top + phi * r1 * (g_best - bottom) + r2 * (g_best - middle)
            pos_new = pop[r_idx1][self.ID_POS] + self.phi * np.random.uniform() * (g_best[self.ID_POS] - pop[r_idx3][self.ID_POS]) \
                      + np.random.uniform() * (g_best[self.ID_POS] - pop[r_idx2][self.ID_POS])
        else:
            # new = top
            pos_new = self.levy_flight(epoch + 1, pop[idx][self.ID_POS], g_best[self.ID_POS])

        # replacement of one electromagnet of generated particle with a random number
        # (only for some generated particles) to bring diversity to the population
        if np.random.uniform() < self.r_rate:
            RI = np.random.randint(0, self.problem.n_dims)
            pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform(self.problem.lb[RI], self.problem.ub[RI])

        # checking whether the generated number is inside boundary or not
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

        # # batch size idea
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) % self.pop_size == 0:
        #         pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        #

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, epoch) for idx in pop_idx]
        return child


class OriginalEFO(BaseEFO):
    """
    The original version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

        https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (): default = 0.85    Like crossover parameter in GA
            p_field (): default = 0.1     portion of population, positive field
            n_field (): default = 0.45    portion of population, negative field
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, r_rate, ps_rate, p_field, n_field, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

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
        if mode != "sequential":
            print("EFO is support sequential mode only!")
            exit(0)
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history.save_initial_best(g_best)

        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        r_index1 = np.random.randint(0, int(self.pop_size * self.p_field), (self.problem.n_dims, self.epoch))
        # random particles from positive field
        r_index2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size, (self.problem.n_dims, self.epoch))
        # random particles from negative field
        r_index3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)), (self.problem.n_dims, self.epoch))
        # random particles from neutral field
        ps = np.random.uniform(0, 1, (self.problem.n_dims, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        r_force = np.random.uniform(0, 1, self.epoch)
        # random force in each generation
        rp = np.random.uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        randomization = np.random.uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        RI = 0
        # index of the electromagnet (variable) which is going to be initialized by random number

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            r = r_force[epoch]
            x_new = np.zeros(self.problem.n_dims)  # temporary array to store generated particle
            for i in range(0, self.problem.n_dims):
                if ps[i, epoch] > self.ps_rate:
                    x_new[i] = pop[r_index3[i, epoch]][self.ID_POS][i] + \
                               self.phi * r * (pop[r_index1[i, epoch]][self.ID_POS][i] - pop[r_index3[i, epoch]][self.ID_POS][i]) + \
                               r * (pop[r_index3[i, epoch]][self.ID_POS][i] - pop[r_index2[i, epoch]][self.ID_POS][i])
                else:
                    x_new[i] = pop[r_index1[i, epoch]][self.ID_POS][i]

            # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
            if rp[epoch] < self.r_rate:
                x_new[RI] = self.problem.lb[RI] + (self.problem.ub[RI] - self.problem.lb[RI]) * randomization[epoch]
                RI = RI + 1
                if RI >= self.problem.n_dims:
                    RI = 0

            # checking whether the generated number is inside boundary or not
            pos_new = self.amend_position_random(x_new)
            fit_new = self.get_fitness_position(pos_new)
            # Updating the population if the fitness of the generated particle is better than worst fitness in
            #     the population (because the population is sorted by fitness, the last particle is the worst)
            pop[-1] = [pos_new, fit_new]

            # update global best position
            pop, g_best = self.update_global_best_solution(pop)  # We sort the population

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
