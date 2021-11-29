#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:00, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        mean_a = np.mean([agent[self.ID_LOUD] for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            agent[self.ID_VEC] = agent[self.ID_VEC] + self.pop[idx][self.ID_PFRE] * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            x = self.pop[idx][self.ID_POS] + agent[self.ID_VEC]
            ## Local Search around g_best position
            if np.random.uniform() > agent[self.ID_PRAT]:
                # print(f"{epoch}, {mean_a}, {self.dyn_r}")
                x = self.g_best[self.ID_POS] + mean_a * np.random.normal(-1, 1)
            pos_new = self.amend_position_faster(x)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
        pop_new = self.update_fitness_population(pop_new)

        for idx in range(0, self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_agent(pop_new[idx], self.pop[idx]) and np.random.rand() < pop_new[idx][self.ID_LOUD]:
                pop_new[idx][self.ID_LOUD] = self.alpha * pop_new[idx][self.ID_LOUD]
                pop_new[idx][self.ID_PRAT] = pop_new[idx][self.ID_PRAT] * (1 - np.exp(-self.gamma * (epoch + 1)))
                self.pop[idx] = deepcopy(pop_new[idx])


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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = []
        for idx in range(0, self.pop_size):
            pf = self.pulse_frequency[0] + (self.pulse_frequency[1] - self.pulse_frequency[0]) * np.random.uniform()  # Eq. 2
            self.dyn_list_velocity[idx] = np.random.uniform() * self.dyn_list_velocity[idx] + \
                                          (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * pf  # Eq. 3
            x = self.pop[idx][self.ID_POS] + self.dyn_list_velocity[idx]  # Eq. 4
            pos_new = self.amend_position_faster(x)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        nfe_epoch += self.pop_size
        pop_child_idx = []
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
            else:
                if np.random.random() > self.pulse_rate:
                    x = self.g_best[self.ID_POS] + 0.01 * np.random.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.amend_position_faster(x)
                    pop_child_idx.append(idx)
                    pop_child.append([pos_new, None])
                    nfe_epoch += 1
        pop_child = self.update_fitness_population(pop_child)
        for idx, idx_selected in enumerate(pop_child_idx):
            if self.compare_agent(pop_child[idx], pop_new[idx_selected]):
                pop_new[idx_selected] = deepcopy(pop_child[idx])
        self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch


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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            agent[self.ID_VEC] = agent[self.ID_VEC] + self.pop[idx][self.ID_PFRE] * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            x = self.pop[idx][self.ID_POS] + agent[self.ID_VEC]
            ## Local Search around g_best position
            if np.random.uniform() > self.pulse_rate:
                x = self.g_best[self.ID_POS] + 0.0001 * np.random.normal(self.problem.n_dims)  # gauss
            pos_new = self.amend_position_faster(x)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
        pop_new = self.update_fitness_population(pop_new)
        for idx in range(self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if self.compare_agent(pop_new[idx], self.pop[idx]) and np.random.rand() < self.loudness:
                self.pop[idx] = deepcopy(pop_new[idx])


