#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:59, 24/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseCOA(Optimizer):
    """
        The original version of: Coyote Optimization Algorithm (COA)
            (Coyote Optimization Algorithm: A new metaheuristic for global optimization problems)
        Link:
            https://ieeexplore.ieee.org/document/8477769
            Old version (Mealpy < 1.2.2) use this Ref code: https://github.com/jkpir/COA/blob/master/COA.py
    """
    ID_AGE = 2

    def __init__(self, problem, epoch=10000, pop_size=100, n_coyotes=5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_coyotes (int): number of coyotes per group
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size + 1
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.n_coyotes = n_coyotes
        self.n_packs = int(pop_size/self.n_coyotes)
        self.ps = 1 / self.problem.n_dims
        self.p_leave = 0.005 * (self.n_coyotes**2)  # Probability of leaving a pack

    def create_solution(self):
        pos = np.random.uniform(self.problem.lb, self.problem.ub)
        fit = self.get_fitness_position(pos)
        age = 1
        return [pos, fit, age]

    def _create_pop_group(self, pop):
        pop_group = []
        for i in range(0, self.n_packs):
            group = pop[i*self.n_coyotes:(i+1)*self.n_coyotes]
            pop_group.append(group)
        return pop_group

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.pop_group = self._create_pop_group(self.pop)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Execute the operations inside each pack
        for p in range(self.n_packs):
            # Get the coyotes that belong to each pack

            self.pop_group[p], local_best = self.get_global_best_solution(self.pop_group[p])
            # Detect alphas according to the costs (Eq. 5)

            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.mean([agent[self.ID_POS] for agent in self.pop_group[p]])

            #  Update coyotes' social condition
            pop_new = []
            for i in range(self.n_coyotes):
                rc1, rc2 = np.random.choice(list(set(range(0, self.n_coyotes)) - {i}), 2, replace=False)

                # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
                pos_new = self.pop_group[p][i][self.ID_POS] + np.random.rand() * \
                          (self.pop_group[p][0][self.ID_POS] - self.pop_group[p][rc1][self.ID_POS]) + \
                          np.random.rand() * (tendency - self.pop_group[p][rc2][self.ID_POS])
                # Keep the coyotes in the search space (optimization problem constraint)
                pos_new = self.amend_position_faster(pos_new)
                pop_new.append([pos_new, None, self.pop_group[p][i][self.ID_AGE]])
            # Evaluate the new social condition (Eq. 13)
            pop_new = self.update_fitness_population(pop_new)
            # Adaptation (Eq. 14)
            self.pop_group[p] = self.greedy_selection_population(self.pop_group[p], pop_new)

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            id_dad, id_mom = np.random.choice(list(range(0, self.n_coyotes)), 2, replace=False)
            prob1 = (1 - self.ps) / 2
            # Generate the pup considering intrinsic and extrinsic influence
            pup = np.where(np.random.uniform(0, 1, self.problem.n_dims) < prob1,
                           self.pop_group[p][id_dad][self.ID_POS], self.pop_group[p][id_mom][self.ID_POS])
            # Eventual noise
            pos_new = np.random.normal(0, 1) * pup
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)

            # Verify if the pup will survive
            packs, local_best = self.get_global_best_solution(self.pop_group[p])
            # Find index of element has fitness larger than new child
            # If existed a element like that, new child is good
            if self.compare_agent([pos_new, fit_new], packs[-1]):
                packs = sorted(packs, key=lambda agent: agent[self.ID_AGE])
                # Replace worst element by new child
                # New born child with age = 0
                packs[-1] = [pos_new, fit_new, 0]
                self.pop_group[p] = deepcopy(packs)

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        if self.n_packs > 1:
            if np.random.rand() < self.p_leave:
                id_pack1, id_pack2 = np.random.choice(list(range(0, self.n_packs)), 2, replace=False)
                id1, id2 = np.random.choice(list(range(0, self.n_coyotes)), 2, replace=False)
                self.pop_group[id_pack1][id1], self.pop_group[id_pack2][id2] = self.pop_group[id_pack2][id2], self.pop_group[id_pack1][id1]

        # Update coyotes ages
        for id_pack in range(0, self.n_packs):
            for id_coy in range(0, self.n_coyotes):
                self.pop_group[id_pack][id_coy][self.ID_AGE] += 1
        self.pop = [agent for pack in self.pop_group for agent in pack]