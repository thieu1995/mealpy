#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:34, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseBeesA(Optimizer):
    """
        The original version of: Bees Algorithm (BeesA)
        Link:
            https://www.sciencedirect.com/science/article/pii/B978008045157250081X
            https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540
    """

    def __init__(self, problem, epoch=10000, pop_size=100, site_ratio=(0.5, 0.4), site_bee_ratio=(0.1, 2),
                 recruited_bee_ratio=0.1, dance_radius=0.1, dance_radius_damp=0.99, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            site_ratio (list): (selected_site_ratio, elite_site_ratio)
            site_bee_ratio (list): (selected_site_bee_ratio, elite_site_bee_ratio)
            recruited_bee_ratio (float):
            dance_radius (float): Bees Dance Radius
            dance_radius_damp (float): Bees Dance Radius Damp Rate
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.site_ratio = site_ratio
        # Scout Bee Count, Selected Sites Bee Count
        self.site_bee_ratio = site_bee_ratio

        self.recruited_bee_ratio = recruited_bee_ratio
        self.dance_radius_damp = dance_radius_damp

        # Initial Value of Dance Radius
        self.dance_radius = dance_radius
        self.dyn_radius = dance_radius
        self.n_selected_bees = int(round(self.site_ratio[0] * self.pop_size))
        self.n_elite_bees = int(round(self.site_ratio[1] * self.n_selected_bees))
        self.n_selected_bees_local = int(round(self.site_bee_ratio[0] * self.pop_size))
        self.n_elite_bees_local = int(round(self.site_bee_ratio[1] * self.n_selected_bees_local))
        self.nfe_per_epoch = self.n_elite_bees * self.n_elite_bees_local + self.pop_size - self.n_selected_bees + \
                             (self.n_selected_bees - self.n_elite_bees) * self.n_selected_bees_local
        self.sort_flag = True

    def perform_dance(self, position, r):
        j = np.random.choice(list(range(0, self.problem.n_dims)))
        position[j] = position[j] + r*np.random.uniform(-1, 1)
        return self.amend_position_faster(position)

    def create_child(self, idx, pop):
        # Elite Sites
        if idx < self.n_elite_bees:
            pop_child = []
            for j in range(0, self.n_elite_bees_local):
                pos_new = self.perform_dance(pop[idx][self.ID_POS], self.dyn_radius)
                fit_new = self.get_fitness_position(pos_new)
                pop_child.append([pos_new, fit_new])
            _, local_best = self.get_global_best_solution(pop_child)
            if self.compare_agent(local_best, pop[idx]):
                return local_best
            return pop[idx].copy()
        elif self.n_elite_bees <= idx < self.n_selected_bees:
        # Selected Non-Elite Sites
            pop_child = []
            for j in range(0, self.n_selected_bees_local):
                pos_new = self.perform_dance(pop[idx][self.ID_POS], self.dyn_radius)
                fit_new = self.get_fitness_position(pos_new)
                pop_child.append([pos_new, fit_new])
            _, local_best = self.get_global_best_solution(pop_child)
            if self.compare_agent(local_best, pop[idx]):
                return local_best
            return pop[idx].copy()
        else:
        # Non-Selected Sites
            return self.create_solution()

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
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop) for idx in pop_idx]

        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
        return child


class ProbBeesA(Optimizer):
    """
        The original version of: Bees Algorithm (BeesA)
        Link:
            https://www.sciencedirect.com/science/article/pii/B978008045157250081X
            https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540
        Version:
            Probabilistic version
    """

    def __init__(self, problem, epoch=10000, pop_size=100, recruited_bee_ratio=0.1,
                 dance_radius=0.1, dance_radius_damp=0.99, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float):
            dance_radius (float): Bees Dance Radius
            dance_radius_damp (float): Bees Dance Radius Damp Rate
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.recruited_bee_ratio = recruited_bee_ratio
        self.dance_radius = dance_radius
        self.dance_radius_damp = dance_radius_damp

        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def perform_dance(self, position, r):
        j = np.random.choice(list(range(0, self.problem.n_dims)))
        position[j] = position[j] + r * np.random.uniform(-1, 1)
        return self.amend_position_faster(position)

    def create_child(self, idx, pop, d_fit):
        # Determine Rejection Probability based on Score
        if d_fit[idx] < 0.9:
            reject_prob = 0.6
        elif 0.9 <= d_fit[idx] < 0.95:
            reject_prob = 0.2
        elif 0.95 <= d_fit[idx] < 1.15:
            reject_prob = 0.05
        else:
            reject_prob = 0

        # Check for Acceptance/Rejection
        if np.random.rand() >= reject_prob:  # Acceptance
            # Calculate New Bees Count
            bee_count = int(np.ceil(d_fit[idx] * self.recruited_bee_count))
            # Create New Bees(Solutions)
            pop_child = []
            for j in range(0, bee_count):
                pos_new = self.perform_dance(pop[idx][self.ID_POS], self.dyn_radius)
                fit_new = self.get_fitness_position(pos_new)
                pop_child.append([pos_new, fit_new])
            _, local_best = self.get_global_best_solution(pop_child)
            if self.compare_agent(local_best, pop[idx]):
                return local_best
            return pop[idx].copy()
        else:
            return self.create_solution()

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
        # Calculate Scores
        fit_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in pop])
        fit_list = 1.0 / fit_list
        d_fit = fit_list / np.mean(fit_list)

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, d_fit=d_fit), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, d_fit=d_fit), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, d_fit) for idx in pop_idx]

        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
        return child

