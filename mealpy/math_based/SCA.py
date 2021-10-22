#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseSCA(Optimizer):
    """
    The original version of: Sine Cosine Algorithm (SCA)
        A Sine Cosine Algorithm for solving optimization problems
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
    Notes:
        + I changed the flow as well as the equations.
        + Removed third loop for faster computational time
        + Batch size ideas
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best, epoch):
        # Eq 3.4, r1 decreases linearly from a to 0
        a = 2.0
        r1 = a - (epoch + 1) * (a / self.epoch)
        # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
        r2 = 2 * np.pi * np.random.uniform(0, 1, self.problem.n_dims)
        r3 = 2 * np.random.uniform(0, 1, self.problem.n_dims)
        # Eq. 3.3, 3.1 and 3.2
        pos_new1 = pop[idx][self.ID_POS] + r1 * np.sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new2 = pop[idx][self.ID_POS] + r1 * np.cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, pos_new1, pos_new2)
        # Check the bound
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

        # ## Update the global best
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) * self.pop_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        #

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

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, g_best, epoch) for idx in pop_idx]
        return pop_new


class OriginalSCA(BaseSCA):
    """
    Original version of: Sine Cosine Algorithm (SCA)
        A Sine Cosine Algorithm for solving optimization problems
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
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

    def create_child(self, idx, pop, g_best, epoch):
        # Eq 3.4, r1 decreases linearly from a to 0
        a = 2.0
        r1 = a - (epoch + 1) * (a / self.epoch)
        pos_new = pop[idx][self.ID_POS].copy()
        for j in range(self.problem.n_dims):  # j-th dimension
            # Update r2, r3, and r4 for Eq. (3.3)
            r2 = 2 * np.pi * np.random.uniform()
            r3 = 2 * np.random.uniform()
            r4 = np.random.uniform()
            # Eq. 3.3, 3.1 and 3.2
            if r4 < 0.5:
                pos_new[j] = pos_new[j] + r1 * np.sin(r2) * abs(r3 * g_best[self.ID_POS][j] - pos_new[j])
            else:
                pos_new[j] = pos_new[j] + r1 * np.cos(r2) * abs(r3 * g_best[self.ID_POS][j] - pos_new[j])
        # Check the bound
        pos_new = self.amend_position_random(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

#
# class FasterSCA(BaseSCA):
#     """
#         A Sine Cosine Algorithm for solving optimization problems (SCA)
#     Link:
#         https://doi.org/10.1016/j.knosys.2015.12.022
#         https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
#
#     This is my version of SCA. The original version of SCA is not working. So I changed the flow of algorithm
#     """
#     def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
#         """
#         Args:
#             epoch (int): maximum number of iterations, default = 10000
#             pop_size (int): number of population size, default = 100
#         """
#         super().__init__(problem, epoch, pop_size, **kwargs)
#         self.nfe_per_epoch = pop_size
#         self.sort_flag = False
#
#     def create_solution(self, minmax=0):
#         position = uniform(self.lb, self.ub)
#         fitness = self.get_fitness_position(position=position, minmax=minmax)
#         return {0: position, 1: fitness}
#
#     def train(self):
#         pop = {i: self.create_solution() for i in range(self.pop_size)}
#         pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
#         g_best = next(iter(pop_sorted.values()))
#
#         for epoch in range(self.epoch):
#             # Update the position of solutions with respect to destination
#             for i, (idx, item) in enumerate(pop.items()):  # i-th position
#                 # Eq 3.4, r1 decreases linearly from a to 0
#                 a = 2.0
#                 r1 = a - (epoch + 1) * (a / self.epoch)
#                 # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
#                 r2 = 2 * pi * uniform(0, 1, self.problem_size)
#                 r3 = 2 * uniform(0, 1, self.problem_size)
#                 # Eq. 3.3, 3.1 and 3.2
#                 pos_new1 = pop[i][self.ID_POS] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
#                 pos_new2 = pop[i][self.ID_POS] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
#                 pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
#                 # Check the bound
#                 pos_new = self.amend_position_random(pos_new)
#                 fit = self.get_fitness_position(pos_new)
#                 if fit < item[self.ID_FIT]:  # My improved part
#                     pop[idx] = {0: pos_new, 1: fit}
#
#                 ## Update the global best
#                 if self.batch_idea:
#                     if (i + 1) % self.batch_size == 0:
#                         pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
#                         current_best = next(iter(pop_sorted.values()))
#                         if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
#                             g_best = deepcopy(current_best)
#                 else:
#                     if (i + 1) * self.pop_size == 0:
#                         pop_sorted = {k: v for k, v in sorted(pop.items(), key=lambda encoded: encoded[1][self.ID_FIT])}
#                         current_best = next(iter(pop_sorted.values()))
#                         if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
#                             g_best = deepcopy(current_best)
#
#             self.loss_train.append(g_best[self.ID_FIT])
#             if self.verbose:
#                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
#         self.solution = g_best
#         return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
#
#
# class FastestSCA(Root):
#     """
#         A Sine Cosine Algorithm for solving optimization problems (SCA)
#     Link:
#         https://doi.org/10.1016/j.knosys.2015.12.022
#         https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
#
#     This is my version of SCA. The original version of SCA is not working. So I changed the flow of algorithm
#     """
#
#     def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
#         super().__init__(obj_func, lb, ub, verbose, kwargs)
#         self.epoch = epoch
#         self.pop_size = pop_size
#
#     def create_solution(self, minmax=0):
#         position = uniform(self.lb, self.ub)
#         fitness = self.get_fitness_position(position=position, minmax=minmax)
#         return {0: position, 1: fitness}
#
#     def train(self):
#         pop = [self.create_solution() for i in range(self.pop_size)]
#         g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
#
#         for epoch in range(self.epoch):
#             # Update the position of solutions with respect to destination
#             for i in range(self.pop_size):  # i-th position
#                 # Eq 3.4, r1 decreases linearly from a to 0
#                 a = 2.0
#                 r1 = a - (epoch + 1) * (a / self.epoch)
#                 # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
#                 r2 = 2 * pi * uniform(0, 1, self.problem_size)
#                 r3 = 2 * uniform(0, 1, self.problem_size)
#                 # Eq. 3.3, 3.1 and 3.2
#                 pos_new1 = pop[i][self.ID_POS] + r1 * sin(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
#                 pos_new2 = pop[i][self.ID_POS] + r1 * cos(r2) * abs(r3 * g_best[self.ID_POS] - pop[i][self.ID_POS])
#                 pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
#                 # Check the bound
#                 pos_new = self.amend_position_random(pos_new)
#                 fit = self.get_fitness_position(pos_new)
#                 if fit < pop[i][self.ID_FIT]:  # My improved part
#                     pop[i] = {0: pos_new, 1: fit}
#
#                 ## Update the global best
#                 if self.batch_idea:
#                     if (i + 1) % self.batch_size == 0:
#                         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
#                 else:
#                     if (i + 1) * self.pop_size == 0:
#                         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
#
#             self.loss_train.append(g_best[self.ID_FIT])
#             if self.verbose:
#                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
#         self.solution = g_best
#         return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
#
