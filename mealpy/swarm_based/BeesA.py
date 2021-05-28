#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:34, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import array, mean, ceil
from mealpy.root import Root


class BaseBeesA(Root):
    """
        The original version of: Bees Algorithm (BeesA)
        Link:
            https://www.sciencedirect.com/science/article/pii/B978008045157250081X
            https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, site_ratio=(0.5, 0.4),
                 site_bee_ratio=(0.1, 2), recruited_bee_ratio=0.1, dance_radius=0.1, dance_radius_damp=0.99, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.site_ratio = site_ratio            # (selected_site_ratio, elite_site_ratio)

        # Scout Bee Count, Selected Sites Bee Count
        self.site_bee_ratio = site_bee_ratio    # (selected_site_bee_ratio, elite_site_bee_ratio)

        self.recruited_bee_ratio = recruited_bee_ratio
        self.dance_radius = dance_radius                # Bees Dance Radius
        self.dance_radius_damp = dance_radius_damp      # Bees Dance Radius Damp Rate

    def perform_dance(self, position, r):
        j = choice(list(range(0, self.problem_size)))
        position[j] = position[j] + r*uniform(-1, 1)
        return self.amend_position_faster(position)


    def train(self):
        # Create Initial Population (Sorted)
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Initial Value of Dance Radius
        r = self.dance_radius

        selected_site_count = int(round(self.site_ratio[0] * self.pop_size))
        elite_site_count = int(round(self.site_ratio[1] * selected_site_count))

        selected_site_bee_count = int(round(self.site_bee_ratio[0] * self.pop_size))
        elite_site_bee_count = int(round(self.site_bee_ratio[1] * selected_site_bee_count))

        for epoch in range(0, self.epoch):
            # Elite Sites
            for i in range(0, elite_site_count):
                pop_child = []
                # Create New Bees (Solutions)
                for j in range(0, elite_site_bee_count):
                    pos_new = self.perform_dance(pop[i][self.ID_POS], r)
                    fit_new = self.get_fitness_position(pos_new)
                    pop_child.append([pos_new, fit_new])
                local_best = self.get_global_best_solution(pop_child, self.ID_FIT, self.ID_MIN_PROB)
                if local_best[self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = local_best

            # Selected Non-Elite Sites
            for i in range(elite_site_count, selected_site_count):
                # Create New Bees (Solutions)
                pop_child = []
                for j in range(0, selected_site_bee_count):
                    pos_new = self.perform_dance(pop[i][self.ID_POS], r)
                    fit_new = self.get_fitness_position(pos_new)
                    pop_child.append([pos_new, fit_new])
                local_best = self.get_global_best_solution(pop_child, self.ID_FIT, self.ID_MIN_PROB)
                if local_best[self.ID_FIT] < pop[i][self.ID_FIT]:
                    pop[i] = local_best

            # Non-Selected Sites
            for i in range(selected_site_count, self.pop_size):
                pop[i] = self.create_solution()

            # Sort Population
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Damp Dance Radius
            r = self.dance_radius_damp * r

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ProbBeesA(Root):
    """
        The original version of: Bees Algorithm (BeesA)
        Link:
            https://www.sciencedirect.com/science/article/pii/B978008045157250081X
            https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540
        Version:
            Probabilistic version
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 recruited_bee_ratio=0.1, dance_radius=0.1, dance_radius_damp=0.99, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.recruited_bee_ratio = recruited_bee_ratio
        self.dance_radius = dance_radius  # Bees Dance Radius
        self.dance_radius_damp = dance_radius_damp  # Bees Dance Radius Damp Rate

    def perform_dance(self, position, r):
        j = choice(list(range(0, self.problem_size)))
        position[j] = position[j] + r * uniform(-1, 1)
        return self.amend_position_faster(position)

    def train(self):
        # Create Initial Population (Sorted)
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Initial Value of Dance Radius
        r = self.dance_radius
        recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

        for epoch in range(0, self.epoch):

            # Calculate Scores
            fit_list = array([solution[self.ID_FIT] for solution in pop])
            fit_list = 1.0 / fit_list
            d = fit_list / mean(fit_list)

            # Iterate on Bees
            for i in range(0, self.pop_size):
                # Determine Rejection Probability based on Score
                if d[i] < 0.9:
                    reject_prob = 0.6
                elif 0.9 <= d[i] < 0.95:
                    reject_prob = 0.2
                elif 0.95 <= d[i] < 1.15:
                    reject_prob = 0.05
                else:
                    reject_prob = 0

                # Check for Acceptance/Rejection
                if uniform() >= reject_prob:  # Acceptance
                    # Calculate New Bees Count
                    bee_count = int(ceil(d[i] * recruited_bee_count))
                    # Create New Bees(Solutions)
                    pop_child = []
                    for j in range(0, bee_count):
                        pos_new = self.perform_dance(pop[i][self.ID_POS], r)
                        fit_new = self.get_fitness_position(pos_new)
                        pop_child.append([pos_new, fit_new])
                    local_best = self.get_global_best_solution(pop_child, self.ID_FIT, self.ID_MIN_PROB)
                    if local_best[self.ID_FIT] < pop[i][self.ID_FIT]:
                        pop[i] = local_best
                else:
                    pop[i] = self.create_solution()

            # Sort Population
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Damp Dance Radius
            r = self.dance_radius_damp * r

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

