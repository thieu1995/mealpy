#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:59, 24/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import choice, uniform, permutation, rand, randint
from numpy import ones, zeros, argmin, argsort, median, logical_not, logical_or, flatnonzero
from mealpy.optimizer import Root


class BaseCOA(Root):
    """
        The original version of: Coyote Optimization Algorithm (COA)
            (Coyote Optimization Algorithm: A new metaheuristic for global optimization problems)
        Link:
            https://ieeexplore.ieee.org/document/8477769
            Ref code: https://github.com/jkpir/COA/blob/master/COA.py
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_coyotes = 5
        self.n_packs = int(pop_size/self.n_coyotes)
        self.ps = 1 / self.problem_size
        self.p_leave = 0.005 * (self.n_coyotes**2)  # Probability of leaving a pack

    def train(self):
        pop = uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        fits = ones((1, self.pop_size))
        for i in range(0, self.pop_size):   # Calculate coyotes adaptation (fitness) Eq.3
            fits[0, i] = self.get_fitness_position(pop[i, :])
        ages = zeros((1, self.pop_size))
        pack_indexes = permutation(self.pop_size).reshape(self.n_packs, self.n_coyotes)

        # Find global best
        id_best = argmin(fits[0, :])
        g_best = [pop[id_best], fits[0, id_best]]

        for epoch in range(self.epoch):     # epoch: year

            # Execute the operations inside each pack
            for p in range(self.n_packs):
                # Get the coyotes that belong to each pack
                pack_pos = pop[pack_indexes[p, :], :]   # Get the 2D position of packs
                pack_fit = fits[0, pack_indexes[p, :]]
                pack_age = ages[0, pack_indexes[p, :]]

                # Detect alphas according to the costs (Eq. 5)
                ind = argsort(pack_fit)
                pack_fit = pack_fit[ind]
                pack_pos = pack_pos[ind, :]
                pack_age = pack_age[ind]
                c_alpha = pack_pos[0, :]

                # Compute the social tendency of the pack (Eq. 6)
                tendency = median(pack_pos, 0)

                #  Update coyotes' social condition
                pack_pos_new = zeros((self.n_coyotes, self.problem_size))
                for i in range(self.n_coyotes):
                    rc1, rc2 = choice(list(set(range(0, self.n_coyotes)) - {i}), 2, replace=False)

                    # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
                    child = pack_pos[i, :] + rand() * (c_alpha - pack_pos[rc1, :]) + rand() * (tendency - pack_pos[rc2, :])
                    # Keep the coyotes in the search space (optimization problem constraint)
                    pack_pos_new[i, :] = self.amend_position_faster(child)
                    # Evaluate the new social condition (Eq. 13)
                    fit_child = self.get_fitness_position(pack_pos_new[i, :])

                    # Adaptation (Eq. 14)
                    if fit_child < pack_fit[i]:
                        pack_fit[i] = fit_child
                        pack_pos[i, :] = pack_pos_new[i, :]

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = permutation(self.n_coyotes)[:2]
            prob1 = (1 - self.ps) / 2
            prob2 = prob1
            pdr = permutation(self.problem_size)
            p1 = zeros((1, self.problem_size))
            p2 = zeros((1, self.problem_size))

            p1[0, pdr[0]] = 1  # Guarantee 1 charac. per individual
            p2[0, pdr[1]] = 1  # Guarantee 1 charac. per individual
            r = rand(1, self.problem_size - 2)
            p1[0, pdr[2:]] = r < prob1
            p2[0, pdr[2:]] = r > 1 - prob2

            # Eventual noise
            n = logical_not(logical_or(p1, p2))

            # Generate the pup considering intrinsic and extrinsic influence
            pup = p1 * pack_pos[parents[0], :] + p2 * pack_pos[parents[1], :] + n * uniform(self.lb, self.ub, (1, self.problem_size))
            pup_fit = self.get_fitness_position(pup)

            # Verify if the pup will survive
            worst = flatnonzero(pack_fit > pup_fit)     # Find index of element has fitness larger than new child
            if len(worst) > 0:      # If existed a element like that, new child is good
                older = argsort(pack_age[worst])
                which = worst[older[::-1]]
                pack_pos[which[0], :] = pup     # Replace worst element by new child
                pack_fit[which[0]] = pup_fit
                pack_age[which[0]] = 0          # New born child with age = 0

            # Update the pack information
            pop[pack_indexes[p], :] = pack_pos
            fits[0, pack_indexes[p]] = pack_fit
            ages[0, pack_indexes[p]] = pack_age

            # A coyote can leave a pack and enter in another pack (Eq. 4)
            if self.n_packs > 1:
                if rand() < self.p_leave:
                    rp = permutation(self.n_packs)[:2]
                    rc = [randint(0, self.n_coyotes), randint(0, self.n_coyotes)]
                    aux = pack_indexes[rp[0], rc[0]]
                    pack_indexes[rp[0], rc[0]] = pack_indexes[rp[1], rc[1]]
                    pack_indexes[rp[1], rc[1]] = aux

            # Update coyotes ages
            pack_age += 1

            ## Update the global best
            id_best = argmin(pack_fit)
            if pack_fit[id_best] < g_best[self.ID_FIT]:
                g_best = [pack_pos[id_best, :], pack_fit[id_best]]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

