#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseASO(Optimizer):
    """
        The original version of: Atom Search Optimization (ASO)
            https://doi.org/10.1016/j.knosys.2018.08.030
            https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2          # Velocity
    ID_MAS = 3          # Mass of atom

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=50, beta=0.2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): Depth weight, default = 50
            beta (float): Multiplier weight, default = 0.2
            **kwargs ():
        """

        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], velocity, mass]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        mass = 0.0
        return [position, fitness, velocity, mass]

    def _update_mass__(self, population):
        fit_total, fit_best, fit_worst = self.get_special_fitness(population)
        for agent in population:
            agent[self.ID_MAS] = np.exp((agent[self.ID_FIT][self.ID_TAR] - fit_best)/(fit_worst - fit_best + self.EPSILON)) / fit_total
        return population

    def _find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin((iteration+1) / self.epoch * np.pi / 2)
        rsmax = 1.24
        if radius/average_dist < rsmin:
            rs = rsmin
        else:
            if radius/average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs)**(-13) - 6 * (-rs)**(-7))
        return potential

    def _acceleration__(self, population, g_best, iteration):
        eps = 2**(-52)
        pop = self._update_mass__(population)

        G = np.exp(-20.0 * (iteration+1) / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5) + 1
        if self.problem.minmax == "min":
            k_best_pop = sorted(pop, key=lambda agent: agent[self.ID_MAS], reverse=True)[:k_best].copy()
        else:
            k_best_pop = sorted(pop, key=lambda agent: agent[self.ID_MAS])[:k_best].copy()
        mk_average = np.mean([item[self.ID_POS] for item in k_best_pop])

        acc_list = np.zeros((self.pop_size, self.problem.n_dims))
        for i in range(0, self.pop_size):
            dist_average = np.linalg.norm(pop[i][self.ID_POS] - mk_average)
            temp = np.zeros((self.problem.n_dims))

            for atom in k_best_pop:
                # calculate LJ-potential
                radius = np.linalg.norm(pop[i][self.ID_POS]-atom[self.ID_POS])
                potential = self._find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * np.random.uniform(0, 1, self.problem.n_dims) * ((atom[self.ID_POS]-pop[i][self.ID_POS])/(radius + eps))
            temp = self.alpha * temp + self.beta * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            # calculate acceleration
            acc = G * temp / pop[i][self.ID_MAS]
            acc_list[i] = acc
        return acc_list

    def create_child(self, idx, pop, atom_acc_list):
        velocity_rand = np.random.uniform(self.problem.lb, self.problem.ub)
        velocity = velocity_rand * pop[idx][self.ID_VEL] + atom_acc_list[idx]
        pos_new = pop[idx][self.ID_POS] + velocity
        # Relocate atom out of range
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new, pop[idx][self.ID_VEL], pop[idx][self.ID_MAS]]
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
        # Calculate acceleration.
        atom_acc_list = self._acceleration__(pop, g_best, iteration=epoch)

        # Update velocity based on random dimensions and position of global best
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, atom_acc_list=atom_acc_list), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, atom_acc_list=atom_acc_list), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, atom_acc_list) for idx in pop_idx]

        _, current_best = self.get_global_best_solution(child)
        if self.compare_agent(g_best, current_best):
            child[np.randint(0, self.pop_size)] = g_best.copy()
        return child
