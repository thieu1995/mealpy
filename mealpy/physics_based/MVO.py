#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
from numpy import cumsum, array, max, reshape, where, min, sum, ptp
from copy import deepcopy
from mealpy.root import Root


class BaseMVO(Root):
    """
        My version of: Multi-Verse Optimizer (MVO)
            http://dx.doi.org/10.1007/s00521-015-1870-7
        Notes:
            + Using my routtele wheel selection which can handle negative values
            + No need condition when normalize fitness. So the chance to choose while whole higher --> better
            + Change equation 3.3 to match the name of parameter wep_minmax
            + Using levy-flight to adapt large-scale dimensions
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, wep_minmax=(0.2, 1.0), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.wep_minmax = wep_minmax  # Wormhole Existence Probability (min and max in Eq.(3.3) paper

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Eq. (3.3) in the paper
            wep = self.wep_minmax[1] - (epoch + 1) * ((self.wep_minmax[1] - self.wep_minmax[0]) / self.epoch)

            # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
            tdr = 1 - (epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

            # Update the position of universes
            for i in range(1, self.pop_size):  # Starting from 1 since 0 is the elite

                if uniform() < wep:
                    if uniform() < 0.5:
                        list_fitness = array([item[self.ID_FIT] for item in pop])

                        white_hole_id = self.get_index_roulette_wheel_selection(list_fitness)

                        black_hole_pos_1 = pop[i][self.ID_POS] + tdr * normal(0, 1) * (pop[white_hole_id][self.ID_POS] - pop[i][self.ID_POS])

                        black_hole_pos_2 = g_best[self.ID_POS] + tdr * normal(0, 1) * (g_best[self.ID_POS] - pop[i][self.ID_POS])

                        black_hole_pos = where(uniform(0, 1, self.problem_size) < 0.5, black_hole_pos_1, black_hole_pos_2)
                    else:
                        black_hole_pos = self.levy_flight(epoch + 1, pop[i][self.ID_POS], g_best[self.ID_POS])
                else:
                    black_hole_pos = uniform(self.lb, self.ub)

                # black_hole_pos = self.amend_position_faster(black_hole_pos)
                fit = self.get_fitness_position(black_hole_pos)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [black_hole_pos, fit]

                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalMVO(Root):
    """
    Original: Multi-Verse Optimizer (MVO)
        http://dx.doi.org/10.1007/s00521-015-1870-7
        https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, wep_minmax=(0.2, 1.0), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
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

    def normalize(self, d, to_sum=True):
        # d is a (n x dimension) np array
        d -= min(d, axis=0)
        d /= (sum(d, axis=0) if to_sum else ptp(d, axis=0))
        return d

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Eq. (3.3) in the paper
            wep = self.wep_minmax[0] + (epoch+1) * ((self.wep_minmax[1] - self.wep_minmax[0]) / self.epoch)

            # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
            tdr = 1 - (epoch+1)**(1.0/6) / self.epoch ** (1.0/6)

            list_fitness_raw = array([item[self.ID_FIT] for item in pop])
            maxx = max(list_fitness_raw)
            if maxx > (2**64-1):
                list_fitness_normalized = uniform(0, 0.1, self.pop_size)
                # print("Fitness value too large for dtype('float64')")
            else:
                ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
                list_fitness_normalized = reshape(self.normalize(array([list_fitness_raw])), self.pop_size)  # Matrix

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
                            black_hole_pos[j] = g_best[self.ID_POS][j] + tdr * uniform(self.lb[j], self.ub[j])
                        else:
                            black_hole_pos[j] = g_best[self.ID_POS][j] - tdr * uniform(self.lb[j], self.ub[j])
                fit = self.get_fitness_position(black_hole_pos)
                pop[i] = [black_hole_pos, fit]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
