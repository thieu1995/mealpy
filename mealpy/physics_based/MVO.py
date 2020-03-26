#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import cumsum, array, max, reshape
from copy import deepcopy
from sklearn.preprocessing import normalize
from mealpy.root import Root


class BaseMVO(Root):
    """
    Original: Multi-Verse Optimizer (MVO)
        http://dx.doi.org/10.1007/s00521-015-1870-7
        https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, wep_minmax=(1.0, 0.2)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.wep_minmax = wep_minmax       # Wormhole Existence Probability (min and max in Eq.(3.3) paper

    # sorted_Inflation_rates
    def _roulette_wheel_selection__(self, weights=None):
        accumulation = cumsum(weights)
        p = uniform() * accumulation[-1]
        chosen_idx = None
        for idx in range(len(accumulation)):
            if accumulation[idx] > p:
                chosen_idx = idx
                break
        return chosen_idx

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Eq. (3.3) in the paper
            wep = self.wep_minmax[0] + (epoch+1) * ( (self.wep_minmax[1] - self.wep_minmax[0]) / self.epoch )

            # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
            tdr = 1 - (epoch+1)**(1.0/6) / self.epoch ** (1.0/6)

            list_fitness_raw = array([item[self.ID_FIT] for item in pop])
            maxx = max(list_fitness_raw)
            if maxx > (2**64-1):
                list_fitness_normalized = uniform(0, 0.1, self.pop_size)
                # print("Fitness value too large for dtype('float64')")
            else:
                ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
                list_fitness_normalized = reshape(normalize(array([list_fitness_raw])), self.pop_size)  # Matrix

            # Update the position of universes
            for i in range(1, self.pop_size):           # Starting from 1 since 0 is the elite
                black_hole_pos = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    r1 = uniform()
                    if r1 < list_fitness_normalized[i]:
                        white_hole_id = self._roulette_wheel_selection__((-1 * list_fitness_raw))
                        if white_hole_id == None or white_hole_id == -1:
                            white_hole_id = 0
                        # Eq. (3.1) in the paper
                        black_hole_pos[j] = pop[white_hole_id][self.ID_POS][j]

                    # Eq. (3.2) in the paper if the boundaries are all the same
                    r2 = uniform()
                    if r2 < wep:
                        r3 = uniform()
                        if r3 < 0.5:
                            black_hole_pos[j] = g_best[self.ID_POS][j] + tdr * uniform(self.domain_range[0], self.domain_range[1])
                        else:
                            black_hole_pos[j] = g_best[self.ID_POS][j] - tdr * uniform(self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(black_hole_pos)
                pop[i] = [black_hole_pos, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
