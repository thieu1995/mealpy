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


class BaseEO(Optimizer):
    """
        The original version of: Equilibrium Optimizer (EO)
            (Equilibrium Optimizer: A Novel Optimization Algorithm)
        Link:
            https://doi.org/10.1016/j.knosys.2019.105190
            https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        ## Fixed parameter proposed by authors
        self.V = 1
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def make_equilibrium_pool(self, list_equilibrium=None):
        pos_list = [item[self.ID_POS] for item in list_equilibrium]
        pos_mean = np.mean(pos_list, axis=0)
        fit = self.get_fitness_position(pos_mean)
        list_equilibrium.append([pos_mean, fit])
        return list_equilibrium

    def create_child(self, idx, pop_copy, c_pool, t):
        current_agent = pop_copy[idx].copy()
        lamda = np.random.uniform(0, 1, self.problem.n_dims)                # lambda in Eq. 11
        r = np.random.uniform(0, 1, self.problem.n_dims)                    # r in Eq. 11
        c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
        f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)         # Eq. 11
        r1 = np.random.uniform()
        r2 = np.random.uniform()                                            # r1, r2 in Eq. 15
        gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)     # Eq. 15
        g0 = gcp * (c_eq - lamda * current_agent[self.ID_POS])              # Eq. 14
        g = g0 * f                                                          # Eq. 13
        pos_new = c_eq + (current_agent[self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
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

        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        # ---------------- Memory saving-------------------  make equilibrium pool
        pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT][self.ID_TAR])
        c_eq_list = pop_sorted[:4].copy()
        c_pool = self.make_equilibrium_pool(c_eq_list)

        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child(idx, pop_copy, c_pool, t) for idx in pop_idx]
        return pop


class ModifiedEO(BaseEO):
    """
        Original version of: Modified Equilibrium Optimizer (MEO)
            (An efficient equilibrium optimizer with mutation strategy for numerical optimization)
    Link:
        https://doi.org/10.1016/j.asoc.2020.106542
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 2*pop_size
        self.sort_flag = False

        self.pop_len = int(self.pop_size / 3)

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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        # ---------------- Memory saving-------------------  make equilibrium pool
        pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT][self.ID_TAR])
        c_eq_list = pop_sorted[:4].copy()
        c_pool = self.make_equilibrium_pool(c_eq_list)

        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child(idx, pop_copy, c_pool, t) for idx in pop_idx]

        ## Sort the updated population based on fitness
        pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT][self.ID_TAR])
        pop_s1 = pop_sorted[:self.pop_len]
        pop_s2 = pop_s1.copy()
        pop_s3 = pop_s1.copy()

        ## Mutation scheme
        for i in range(0, self.pop_len):
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))  # Eq. 12
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            pop_s2[i] = [pos_new, fit_new]

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        for i in range(0, self.pop_len):
            pos_new = (c_pool[0][self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            pop_s3[i] = [pos_new, fit_new]

        ## Construct a new population
        pop = pop_s1 + pop_s2 + pop_s3
        temp = self.pop_size - len(pop)
        idx_selected = np.random.choice(range(0, len(c_pool)), temp, replace=False)
        for i in range(0, temp):
            pop.append(c_pool[idx_selected[i]])
        return pop


class AdaptiveEO(BaseEO):
    """
        Original version of: Adaptive Equilibrium Optimization (AEO)
            (A novel interdependence based multilevel thresholding technique using adaptive equilibrium optimizer)
    Link:
        https://doi.org/10.1016/j.engappai.2020.103836
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.pop_len = int(self.pop_size / 3)

    def create_child(self, idx, pop_copy, c_pool, t):
        lamda = np.random.uniform(0, 1, self.problem.n_dims)
        r = np.random.uniform(0, 1, self.problem.n_dims)
        c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
        f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14

        r1 = np.random.uniform()
        r2 = np.random.uniform()
        gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)
        g0 = gcp * (c_eq - lamda * pop_copy[idx][self.ID_POS])
        g = g0 * f

        fit_average = np.mean([item[self.ID_FIT][self.ID_TAR] for item in pop_copy])  # Eq. 19
        pos_new = c_eq + (pop_copy[idx][self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 9
        if pop_copy[idx][self.ID_FIT][self.ID_TAR] >= fit_average:
            pos_new = np.multiply(pos_new, (0.5 + np.random.uniform(0, 1, self.problem.n_dims)))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        # ---------------- Memory saving-------------------  make equilibrium pool
        pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT][self.ID_TAR])
        c_eq_list = pop_sorted[:4].copy()
        c_pool = self.make_equilibrium_pool(c_eq_list)

        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        ## Memory saving, Eq 20, 21
        if epoch != 0:
            for i in range(0, self.pop_size):
                pop_copy[i] = self.get_better_solution(pop[i], pop_copy[i])
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop_copy = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, c_pool=c_pool, t=t), pop_idx)
            pop_copy = [x for x in pop_child]
        else:
            pop_copy = [self.create_child(idx, pop_copy, c_pool, t) for idx in pop_idx]
        return pop_copy


class LevyEO(BaseEO):
    """
        My modified version of: Equilibrium Optimizer (EO)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def create_child_new(self, idx, pop_copy, c_pool, t, epoch, g_best):
        if np.random.uniform() < 0.5:
            lamda = np.random.uniform(0, 1, self.problem.n_dims)  # lambda in Eq. 11
            r = np.random.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = np.random.uniform()
            r2 = np.random.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * pop_copy[idx][self.ID_POS])  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = c_eq + (pop_copy[idx][self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
        else:
            ## Idea: Sometimes, an unpredictable event happens, It make the status of equilibrium change.
            step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
            pos_new = pop_copy[idx][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) \
                      * step * (pop_copy[idx][self.ID_POS] - g_best[self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
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

        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        # ---------------- Memory saving-------------------  make equilibrium pool
        pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT][self.ID_TAR])
        c_eq_list = pop_sorted[:4].copy()
        c_pool = self.make_equilibrium_pool(c_eq_list)

        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child_new, pop_copy=pop_copy, c_pool=c_pool, t=t, epoch=epoch, g_best=g_best), pop_idx)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child_new, pop_copy=pop_copy, c_pool=c_pool, t=t, epoch=epoch, g_best=g_best), pop_idx)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child_new(idx, pop_copy, c_pool, t, epoch, g_best) for idx in pop_idx]
        return pop

