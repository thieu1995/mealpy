#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:34, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from copy import deepcopy


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
        j = np.random.choice(range(0, self.problem.n_dims))
        position[j] = position[j] + r*np.random.uniform(-1, 1)
        return self.amend_position_faster(position)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = deepcopy(self.pop)
        for idx in range(0, self.pop_size):
            # Elite Sites
            if idx < self.n_elite_bees:
                nfe_epoch += self.n_elite_bees_local
                pop_child = []
                for j in range(0, self.n_elite_bees_local):
                    pos_new = self.perform_dance(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                pop_child = self.update_fitness_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    pop_new[idx] = local_best
            elif self.n_elite_bees <= idx < self.n_selected_bees:
                # Selected Non-Elite Sites
                nfe_epoch += self.n_selected_bees_local
                pop_child = []
                for j in range(0, self.n_selected_bees_local):
                    pos_new = self.perform_dance(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                pop_child = self.update_fitness_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    pop_new[idx] = local_best
            else:
                # Non-Selected Sites
                nfe_epoch += 1
                pop_new[idx] = self.create_solution()
        self.pop = pop_new
        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
        self.nfe_per_epoch = nfe_epoch


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
        self.nfe_per_epoch = pop_size
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Calculate Scores
        fit_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in self.pop])
        fit_list = 1.0 / fit_list
        d_fit = fit_list / np.mean(fit_list)

        nfe_epoch = 0
        for idx in range(0, self.pop_size):
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
                if bee_count < 2: bee_count = 2
                if bee_count > self.pop_size: bee_count = self.pop_size
                # Create New Bees(Solutions)
                pop_child = []
                nfe_epoch += bee_count
                for j in range(0, bee_count):
                    pos_new = self.perform_dance(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                pop_child = self.update_fitness_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    self.pop[idx] = local_best
            else:
                nfe_epoch += 1
                self.pop[idx] = self.create_solution()
        self.nfe_per_epoch = nfe_epoch
        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
