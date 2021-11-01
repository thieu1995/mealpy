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
from mealpy.optimizer import Optimizer


class BasicBA(Optimizer):
    """
        The original version of: Bat-Inspired Algorithm (BA)
            A New Metaheuristic Bat-Inspired Algorithm
        Notes:
            The value of A and r are changing after each iteration
    """
    ID_VEC = 2      # Velocity
    ID_LOUD = 3     # Loudness
    ID_PRAT = 4     # Pulse Rate
    ID_PFRE = 5     # Pulse Frequency

    def __init__(self, problem, epoch=10000, pop_size=100, loudness=(1.0, 2.0),
                 pulse_rate=(0.15, 0.85), pulse_frequency=(0, 10), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness (): (A_min, A_max): loudness, default = (1.0, 2.0)
            pulse_rate (): (r_min, r_max): pulse rate / emission rate, default = (0.15, 0.85)
            pulse_frequency (): (pf_min, pf_max): pulse frequency, default = (0, 10)
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.pulse_frequency = pulse_frequency
        self.alpha = self.gamma = 0.9

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
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        loudness = np.random.uniform(self.loudness[0], self.loudness[1])
        pulse_rate = np.random.uniform(self.pulse_rate[0], self.pulse_rate[1])
        pulse_frequency = self.pulse_frequency[0] + (self.pulse_frequency[1] - self.pulse_frequency[0]) * np.random.uniform()
        return [position, fitness, velocity, loudness, pulse_rate, pulse_frequency]

    def create_child(self, idx, pop, g_best, epoch, mean_a):
        agent = pop[idx].copy()
        agent[self.ID_VEC] = agent[self.ID_VEC] + pop[idx][self.ID_PFRE] * (pop[idx][self.ID_POS] - g_best[self.ID_POS])
        x = pop[idx][self.ID_POS] + agent[self.ID_VEC]
        ## Local Search around g_best position
        if np.random.uniform() > agent[self.ID_PRAT]:
            # print(f"{epoch}, {mean_a}, {self.dyn_r}")
            x = g_best[self.ID_POS] + mean_a * np.random.normal(-1, 1)
        pos_new = self.amend_position_faster(x)
        fit_new = self.get_fitness_position(pos_new)

        ## Replace the old position by the new one when its has better fitness.
        ##  and then update loudness and emission rate
        if self.compare_agent([pos_new, fit_new], pop[idx]) and np.random.rand() < agent[self.ID_LOUD]:
            agent[self.ID_POS] = pos_new
            agent[self.ID_FIT] = fit_new
            agent[self.ID_LOUD] = self.alpha * agent[self.ID_LOUD]
            agent[self.ID_PRAT] = agent[self.ID_PRAT] * (1 - np.exp(-self.gamma * (epoch + 1)))
            return agent
        return pop[idx].copy()

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
        mean_a = np.mean([agent[self.ID_LOUD] for agent in pop])
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, mean_a=mean_a), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, mean_a=mean_a), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, epoch, mean_a) for idx in pop_idx]
        return child


class BaseBA(Optimizer):
    """
    My modified version of: Bat-inspired Algorithm (BA)
    (A little bit better than both 2 original version)
    - No need A parameter
    - I changed the process.
        + 1st: We proceed exploration phase (using frequency)
        + 2nd: If new position has better fitness we replace the old position
        + 3rd: Otherwise, we proceed exploitation phase (using finding around the best position so far)
    Link:
        https://link.springer.com/chapter/10.1007/978-3-642-12538-6_6
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pulse_rate=0.95, pulse_frequency=(0, 10), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pulse_rate (float): (r_min, r_max): pulse rate / emission rate, default = (0.15, 0.85)
            pulse_frequency (list): (pf_min, pf_max): pulse frequency, default = (0, 10)
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.pulse_rate = pulse_rate
        self.pulse_frequency = pulse_frequency
        self.alpha = self.gamma = 0.9

        self.dyn_list_velocity = np.zeros((self.pop_size, self.problem.n_dims))

    def create_child(self, idx, pop, g_best):
        pf = self.pulse_frequency[0] + (self.pulse_frequency[1] - self.pulse_frequency[0]) * np.random.uniform()  # Eq. 2
        self.dyn_list_velocity[idx] = np.random.uniform() * self.dyn_list_velocity[idx] + (g_best[self.ID_POS] - pop[idx][self.ID_POS]) * pf  # Eq. 3
        x = pop[idx][self.ID_POS] + self.dyn_list_velocity[idx]  # Eq. 4
        pos_new = self.amend_position_faster(x)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        else:
            if np.random.uniform() > self.pulse_rate:
                x = g_best[self.ID_POS] + 0.01 * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position_faster(x)
                fit_new = self.get_fitness_position(x)
                if self.compare_agent([pos_new, fit_new], pop[idx]):
                    return [pos_new, fit_new]
            return pop[idx].copy()

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
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best) for idx in pop_idx]
        return child


class OriginalBA(Optimizer):
    """
        The original version of: Bat-Inspired Algorithm (BA)
            A New Metaheuristic Bat-Inspired Algorithm
        Notes:
            The value of A and r parameters are constant
    """
    ID_VEC = 2  # Velocity
    ID_PFRE = 3  # Pulse Frequency

    def __init__(self, problem, epoch=10000, pop_size=100, loudness=0.8, pulse_rate=0.95, pulse_frequency=(0, 10), **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness (float): (A_min, A_max): loudness, default = (1.0, 2.0)
            pulse_rate (float): (r_min, r_max): pulse rate / emission rate, default = (0.15, 0.85)
            pulse_frequency (list): (pf_min, pf_max): pulse frequency, default = (0, 10)
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.pulse_frequency = pulse_frequency
        self.alpha = self.gamma = 0.9

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
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        pulse_frequency = self.pulse_frequency[0] + (self.pulse_frequency[1] - self.pulse_frequency[0]) * np.random.uniform()
        return [position, fitness, velocity, pulse_frequency]

    def create_child(self, idx, pop, g_best):
        agent = pop[idx].copy()
        agent[self.ID_VEC] = agent[self.ID_VEC] + pop[idx][self.ID_PFRE] * (pop[idx][self.ID_POS] - g_best[self.ID_POS])
        x = pop[idx][self.ID_POS] + agent[self.ID_VEC]
        ## Local Search around g_best position
        if np.random.uniform() > self.pulse_rate:
            # print(f"{epoch}, {mean_a}, {self.dyn_r}")
            x = g_best[self.ID_POS] + 0.001 * np.random.normal(self.problem.n_dims)  # gauss
        pos_new = self.amend_position_faster(x)
        fit_new = self.get_fitness_position(pos_new)

        ## Replace the old position by the new one when its has better fitness.
        ##  and then update loudness and emission rate
        if self.compare_agent([pos_new, fit_new], pop[idx]) and np.random.rand() < self.loudness:
            agent[self.ID_POS] = pos_new
            agent[self.ID_FIT] = fit_new
            return agent
        return pop[idx].copy()

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
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best) for idx in pop_idx]
        return child
