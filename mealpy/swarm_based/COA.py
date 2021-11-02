#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:59, 24/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
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
        self.nfe_per_epoch = pop_size
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

    def _create_population__(self):
        pop = []
        for i in range(0, self.n_packs):
            group = [self.create_solution() for _ in range(0, self.n_coyotes)]
            pop.append(group)
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
        if mode != "sequential":
            print("COA is supported sequential mode only!")
            exit(0)
        self.termination_start()
        pop = self._create_population__()
        pop_new = [agent for pack in pop for agent in pack]
        _, g_best = self.get_global_best_solution(pop_new)
        self.history.save_initial_best(g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            # Execute the operations inside each pack
            for p in range(self.n_packs):
                # Get the coyotes that belong to each pack

                pop[p], local_best = self.get_global_best_solution(pop[p])
                # Detect alphas according to the costs (Eq. 5)

                # Compute the social tendency of the pack (Eq. 6)
                tendency = np.mean([agent[self.ID_POS] for agent in pop[p]])

                #  Update coyotes' social condition
                for i in range(self.n_coyotes):
                    rc1, rc2 = np.random.choice(list(set(range(0, self.n_coyotes)) - {i}), 2, replace=False)

                    # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
                    pos_new = pop[p][i][self.ID_POS] + np.random.rand() * (pop[p][0][self.ID_POS] - pop[p][rc1][self.ID_POS]) + \
                              np.random.rand() * (tendency - pop[p][rc2][self.ID_POS])
                    # Keep the coyotes in the search space (optimization problem constraint)
                    pos_new = self.amend_position_faster(pos_new)
                    # Evaluate the new social condition (Eq. 13)
                    fit_new = self.get_fitness_position(pos_new)

                    # Adaptation (Eq. 14)
                    if self.compare_agent([pos_new, fit_new], pop[p][i]):
                        pop[p][i] = [pos_new, fit_new, pop[p][i][self.ID_AGE]]

                # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
                id_dad, id_mom = np.random.choice(list(range(0, self.n_coyotes)), 2, replace=False)
                prob1 = (1 - self.ps) / 2
                # Generate the pup considering intrinsic and extrinsic influence
                pup = np.where(np.random.uniform(0, 1, self.problem.n_dims) < prob1, pop[p][id_dad][self.ID_POS], pop[p][id_mom][self.ID_POS])
                # Eventual noise
                pos_new = np.random.normal(0, 1) * pup
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)

                # Verify if the pup will survive
                packs, local_best = self.get_global_best_solution(pop[p])
                # Find index of element has fitness larger than new child
                # If existed a element like that, new child is good
                if self.compare_agent([pos_new, fit_new], packs[-1]):
                    packs = sorted(packs, key=lambda agent: agent[self.ID_AGE])
                    # Replace worst element by new child
                    # New born child with age = 0
                    packs[-1] = [pos_new, fit_new, 0]
                    pop[p] = packs.copy()

            # A coyote can leave a pack and enter in another pack (Eq. 4)
            if self.n_packs > 1:
                if np.random.rand() < self.p_leave:
                    id_pack1, id_pack2 = np.random.choice(list(range(0, self.n_packs)), 2, replace=False)
                    id1, id2 = np.random.choice(list(range(0, self.n_coyotes)), 2, replace=False)
                    pop[id_pack1][id1], pop[id_pack2][id2] = pop[id_pack2][id2].copy(), pop[id_pack1][id1].copy()

            # Update coyotes ages
            for id_pack in range(0, self.n_packs):
                for id_coy in range(0, self.n_coyotes):
                    pop[id_pack][id_coy][self.ID_AGE] += 1

            ## Update the global best
            pop_new = [agent for pack in pop for agent in pack]
            _, g_best = self.update_global_best_solution(pop_new)

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop_new.copy())
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


    # def train(self):
    #     pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
    #     fits = np.ones((1, self.pop_size))
    #     for i in range(0, self.pop_size):   # Calculate coyotes adaptation (fitness) Eq.3
    #         fits[0, i] = self.get_fitness_position(pop[i, :])
    #     ages = np.zeros((1, self.pop_size))
    #     pack_indexes = np.random.permutation(self.pop_size).reshape(self.n_packs, self.n_coyotes)       # 20 packs - 5 coyetes
    #
    #     # Find global best
    #     id_best = np.argmin(fits[0, :])
    #     g_best = [pop[id_best], fits[0, id_best]]
    #
    #     for epoch in range(self.epoch):     # epoch: year
    #
    #         # Execute the operations inside each pack
    #         for p in range(self.n_packs):
    #             # Get the coyotes that belong to each pack
    #             pack_pos = pop[pack_indexes[p, :], :]   # Get the 2D position of packs
    #             pack_fit = fits[0, pack_indexes[p, :]]
    #             pack_age = ages[0, pack_indexes[p, :]]
    #
    #             # Detect alphas according to the costs (Eq. 5)
    #             ind = np.argsort(pack_fit)
    #             pack_fit = pack_fit[ind]
    #             pack_pos = pack_pos[ind, :]
    #             pack_age = pack_age[ind]
    #             c_alpha = pack_pos[0, :]
    #
    #             # Compute the social tendency of the pack (Eq. 6)
    #             tendency = np.median(pack_pos, 0)
    #
    #             #  Update coyotes' social condition
    #             pack_pos_new = np.zeros((self.n_coyotes, self.problem_size))
    #             for i in range(self.n_coyotes):
    #                 rc1, rc2 = np.random.choice(list(set(range(0, self.n_coyotes)) - {i}), 2, replace=False)
    #
    #                 # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
    #                 child = pack_pos[i, :] + rand() * (c_alpha - pack_pos[rc1, :]) + rand() * (tendency - pack_pos[rc2, :])
    #                 # Keep the coyotes in the search space (optimization problem constraint)
    #                 pack_pos_new[i, :] = self.amend_position_faster(child)
    #                 # Evaluate the new social condition (Eq. 13)
    #                 fit_child = self.get_fitness_position(pack_pos_new[i, :])
    #
    #                 # Adaptation (Eq. 14)
    #                 if fit_child < pack_fit[i]:
    #                     pack_fit[i] = fit_child
    #                     pack_pos[i, :] = pack_pos_new[i, :]
    #
    #         # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
    #         parents = np.random.permutation(self.n_coyotes)[:2]
    #         prob1 = (1 - self.ps) / 2
    #         prob2 = prob1
    #         pdr = np.random.permutation(self.problem_size)
    #         p1 = np.zeros((1, self.problem_size))
    #         p2 = np.zeros((1, self.problem_size))
    #
    #         p1[0, pdr[0]] = 1  # Guarantee 1 charac. per individual
    #         p2[0, pdr[1]] = 1  # Guarantee 1 charac. per individual
    #         r = rand(1, self.problem_size - 2)
    #         p1[0, pdr[2:]] = r < prob1
    #         p2[0, pdr[2:]] = r > 1 - prob2
    #
    #         # Eventual noise
    #         n = np.logical_not(np.logical_or(p1, p2))
    #
    #         # Generate the pup considering intrinsic and extrinsic influence
    #         pup = p1 * pack_pos[parents[0], :] + p2 * pack_pos[parents[1], :] + n * np.random.uniform(self.lb, self.ub, (1, self.problem_size))
    #         pup_fit = self.get_fitness_position(pup)
    #
    #         # Verify if the pup will survive
    #         worst = np.flatnonzero(pack_fit > pup_fit)     # Find index of element has fitness larger than new child
    #         if len(worst) > 0:      # If existed a element like that, new child is good
    #             older = np.argsort(pack_age[worst])
    #             which = worst[older[::-1]]
    #             pack_pos[which[0], :] = pup     # Replace worst element by new child
    #             pack_fit[which[0]] = pup_fit
    #             pack_age[which[0]] = 0          # New born child with age = 0
    #
    #         # Update the pack information
    #         pop[pack_indexes[p], :] = pack_pos
    #         fits[0, pack_indexes[p]] = pack_fit
    #         ages[0, pack_indexes[p]] = pack_age
    #
    #         # A coyote can leave a pack and enter in another pack (Eq. 4)
    #         if self.n_packs > 1:
    #             if rand() < self.p_leave:
    #                 rp = np.random.permutation(self.n_packs)[:2]
    #                 rc = [np.random.randint(0, self.n_coyotes), np.random.randint(0, self.n_coyotes)]
    #                 aux = pack_indexes[rp[0], rc[0]]
    #                 pack_indexes[rp[0], rc[0]] = pack_indexes[rp[1], rc[1]]
    #                 pack_indexes[rp[1], rc[1]] = aux
    #
    #         # Update coyotes ages
    #         pack_age += 1
    #
    #         ## Update the global best
    #         id_best = np.argmin(pack_fit)
    #         if pack_fit[id_best] < g_best[self.ID_FIT]:
    #             g_best = [pack_pos[id_best, :], pack_fit[id_best]]
    #
    #         self.loss_train.append(g_best[self.ID_FIT])
    #         if self.verbose:
    #             print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
    #     self.solution = g_best
    #     return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
    #
