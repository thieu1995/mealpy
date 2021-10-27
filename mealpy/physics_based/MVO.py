#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseMVO(Optimizer):
    """
        My version of: Multi-Verse Optimizer (MVO)
            http://dx.doi.org/10.1007/s00521-015-1870-7
        Notes:
            + Using my routtele wheel selection which can handle negative values
            + No need condition when np.random.normalize fitness. So the chance to choose while whole higher --> better
            + Change equation 3.3 to match the name of parameter wep_minmax
            + Using levy-flight to adapt large-scale dimensions
    """

    def __init__(self, problem, epoch=10000, pop_size=100, wep_min=0.2, wep_max=1.0, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.wep_min = wep_min
        self.wep_max = wep_max

    def create_child(self, idx, pop, g_best, epoch, wep, tdr):
        if np.random.uniform() < wep:
            if np.random.rand() < 0.5:
                list_fitness = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop])

                white_hole_id = self.get_index_roulette_wheel_selection(list_fitness)

                black_hole_pos_1 = pop[idx][self.ID_POS] + tdr * np.random.normal(0, 1) * (pop[white_hole_id][self.ID_POS] - pop[idx][self.ID_POS])

                black_hole_pos_2 = g_best[self.ID_POS] + tdr * np.random.normal(0, 1) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])

                black_hole_pos = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, black_hole_pos_1, black_hole_pos_2)
            else:
                black_hole_pos = self.levy_flight(epoch + 1, pop[idx][self.ID_POS], g_best[self.ID_POS])
        else:
            black_hole_pos = np.random.uniform(self.problem.lb, self.problem.ub)

        pos_new = self.amend_position_faster(black_hole_pos)
        fit_new = self.get_fitness_position(black_hole_pos)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
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
        # Eq. (3.3) in the paper
        wep = self.wep_max - (epoch + 1) * ((self.wep_max - self.wep_min) / self.epoch)

        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - (epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

        # Update the position of universes
        pop_idx = np.array(range(0, self.pop_size))  # Starting from 1 since 0 is the elite
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, wep=wep, tdr=tdr), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, wep=wep, tdr=tdr), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best, epoch, wep, tdr) for idx in pop_idx]
        return child


class OriginalMVO(BaseMVO):
    """
    Original: Multi-Verse Optimizer (MVO)
        http://dx.doi.org/10.1007/s00521-015-1870-7
        https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo
    """

    def __init__(self, problem, epoch=10000, pop_size=100, wep_min=0.2, wep_max=1.0, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, wep_min, wep_max, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

    # sorted_inflation_rates
    def _roulette_wheel_selection__(self, weights=None):
        accumulation = np.cumsum(weights)
        p = np.random.uniform() * accumulation[-1]
        chosen_idx = None
        for idx in range(len(accumulation)):
            if accumulation[idx] > p:
                chosen_idx = idx
                break
        return chosen_idx

    def normalize(self, d, to_sum=True):
        # d is a (n x dimension) np np.array
        d -= np.min(d, axis=0)
        d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
        return d

    def create_child2(self, idx, pop, g_best, wep, tdr, list_fitness_normalized, list_fitness_raw):
        black_hole_pos = pop[idx][self.ID_POS].copy()
        for j in range(0, self.problem.n_dims):
            r1 = np.random.uniform()
            if r1 < list_fitness_normalized[idx]:
                white_hole_id = self._roulette_wheel_selection__((-1 * list_fitness_raw))
                if white_hole_id == None or white_hole_id == -1:
                    white_hole_id = 0
                # Eq. (3.1) in the paper
                black_hole_pos[j] = pop[white_hole_id][self.ID_POS][j]

            # Eq. (3.2) in the paper if the boundaries are all the same
            r2 = np.random.uniform()
            if r2 < wep:
                r3 = np.random.uniform()
                if r3 < 0.5:
                    black_hole_pos[j] = g_best[self.ID_POS][j] + tdr * np.random.uniform(self.problem.lb[j], self.problem.ub[j])
                else:
                    black_hole_pos[j] = g_best[self.ID_POS][j] - tdr * np.random.uniform(self.problem.lb[j], self.problem.ub[j])
        pos_new = self.amend_position_faster(black_hole_pos)
        fit_new = self.get_fitness_position(black_hole_pos)
        return [pos_new, fit_new]

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
        # Eq. (3.3) in the paper
        wep = self.wep_min + (epoch + 1) * ((self.wep_max - self.wep_min) / self.epoch)

        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - (epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

        list_fitness_raw = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop])
        maxx = max(list_fitness_raw)
        if maxx > (2 ** 64 - 1):
            list_fitness_normalized = np.random.uniform(0, 0.1, self.pop_size)
        else:
            ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
            list_fitness_normalized = np.reshape(self.normalize(np.array([list_fitness_raw])), self.pop_size)  # Matrix

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, wep=wep, tdr=tdr,
                                                 list_fitness_normalized=list_fitness_normalized, list_fitness_raw=list_fitness_raw), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop, g_best=g_best, wep=wep, tdr=tdr,
                                                 list_fitness_normalized=list_fitness_normalized, list_fitness_raw=list_fitness_raw), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child2(idx, pop, g_best, wep, tdr, list_fitness_normalized, list_fitness_raw) for idx in pop_idx]
        return child