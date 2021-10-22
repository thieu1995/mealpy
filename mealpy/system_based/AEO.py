#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAEO(Optimizer):
    """
    Original version of: Artificial Ecosystem-based Optimization (AEO)
        (Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm)
    Link:
        https://doi.org/10.1007/s00521-019-04452-x
        https://www.mathworks.com/matlabcentral/fileexchange/72685-artificial-ecosystem-based-optimization-aeo
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop):
        rand = np.random.random()
        # Eq. 4, 5, 6
        v1 = np.random.normal(0, 1)
        v2 = np.random.normal(0, 1)
        c = 0.5 * v1 / abs(v2)  # Consumption factor

        if idx == 0:
            j = 1
        else:
            j = np.random.randint(0, idx)

        ### Herbivore
        if rand < 1.0 / 3:
            x_t1 = pop[idx][self.ID_POS] + c * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
        ### Carnivore
        elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
            x_t1 = pop[idx][self.ID_POS] + c * (pop[idx][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
        ### Omnivore
        else:
            r2 = np.random.uniform()
            x_t1 = pop[idx][self.ID_POS] + c * (r2 * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])
                                                + (1 - r2) * (pop[idx][self.ID_POS] - pop[j][self.ID_POS]))
        pos_new = self.amend_position_faster(x_t1)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, agent_i, best):
        r3 = np.random.uniform()
        d = 3 * np.random.normal(0, 1)
        e = r3 * np.random.randint(1, 3) - 1
        h = 2 * r3 - 1
        x_t1 = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
        pos_new = self.amend_position_faster(x_t1)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], agent_i):
            return [pos_new, fit_new]
        return agent_i.copy()

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
        pop_idx = np.array(range(0, self.pop_size-1))

        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * np.random.uniform()
        x1 = (1 - a) * pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position_faster(x1)
        fit_new = self.get_fitness_position(x1)
        pop[-1] = [pos_new, fit_new]

        ## Consumption - Update the whole population left
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy) for idx in pop_idx]
        pop_new.append(pop[-1])

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(pop_new)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best), pop_new)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best), pop_new)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child2(agent, best) for agent in pop_new]
        return pop_new


class ImprovedAEO(OriginalAEO):
    """
    Original version of: Improved Artificial Ecosystem-based Optimization
        (Artificial ecosystem optimizer for parameters identification of proton exchange membrane fuel cells model)
    Link:
        https://doi.org/10.1016/j.ijhydene.2020.06.256
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)

    def create_child3(self, agent_i, pop, best, epoch):
        r3 = np.random.uniform()
        d = 3 * np.random.normal(0, 1)
        e = r3 * np.random.randint(1, 3) - 1
        h = 2 * r3 - 1

        x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
        if np.random.random() < 0.5:
            beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
            x_r = pop[np.random.randint(0, self.pop_size-1)][self.ID_POS]
            if np.random.random() < 0.5:
                x_new = beta * x_r + (1 - beta) * agent_i[self.ID_POS]
            else:
                x_new = beta * agent_i[self.ID_POS] + (1 - beta) * x_r
        else:
            best[self.ID_POS] = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
        pos_new = self.amend_position_faster(x_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], agent_i):
            return [pos_new, fit_new]
        return agent_i.copy()

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
        pop_idx = np.array(range(0, self.pop_size - 1))

        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * np.random.uniform()
        x1 = (1 - a) * pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position_faster(x1)
        fit_new = self.get_fitness_position(x1)
        pop[-1] = [pos_new, fit_new]

        ## Consumption - Update the whole population left
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy) for idx in pop_idx]
        pop_new.append(pop[-1])

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(pop_new)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child3, pop=pop_new, best=best, epoch=epoch), pop_new)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child3, pop=pop_new, best=best, epoch=epoch), pop_new)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child3(agent, pop_new, best, epoch) for agent in pop_new]
        return pop_new


class EnhancedAEO(Optimizer):
    """
    Original version of: Enhanced Artificial Ecosystem-Based Optimization
        (An Enhanced Artificial Ecosystem-Based Optimization for Optimal Allocation of Multiple Distributed Generations)
    Link:
        https://doi.org/10.1109/ACCESS.2020.3027654
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop):
        rand = np.random.random()
        # Eq. 4, 5, 6
        v1 = np.random.normal(0, 1)
        v2 = np.random.normal(0, 1)
        c = 0.5 * v1 / abs(v2)  # Consumption factor

        r3 = 2 * np.pi * np.random.random()
        r4 = np.random.random()

        if idx == 0:
            j = 1
        else:
            j = np.random.randint(0, idx)
        ### Herbivore
        if rand <= 1.0 / 3:  # Eq. 15
            if r4 <= 0.5:
                x_t1 = pop[idx][self.ID_POS] + np.sin(r3) * c * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])
            else:
                x_t1 = pop[idx][self.ID_POS] + np.cos(r3) * c * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])
        ### Carnivore
        elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 16
            if r4 <= 0.5:
                x_t1 = pop[idx][self.ID_POS] + np.sin(r3) * c * (pop[idx][self.ID_POS] - pop[j][self.ID_POS])
            else:
                x_t1 = pop[idx][self.ID_POS] + np.cos(r3) * c * (pop[idx][self.ID_POS] - pop[j][self.ID_POS])
        ### Omnivore
        else:  # Eq. 17
            r5 = np.random.random()
            if r4 <= 0.5:
                x_t1 = pop[idx][self.ID_POS] + np.sin(r5) * c * \
                       (r5 * (pop[idx][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r5) * (pop[idx][self.ID_POS] - pop[j][self.ID_POS]))
            else:
                x_t1 = pop[idx][self.ID_POS] + np.cos(r5) * c * \
                       (r5 * (pop[idx][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r5) * (pop[idx][self.ID_POS] - pop[j][self.ID_POS]))
        pos_new = self.amend_position_faster(x_t1)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, best, pop, epoch):
        r3 = np.random.uniform()
        d = 3 * np.random.normal(0, 1)
        e = r3 * np.random.randint(1, 3) - 1
        h = 2 * r3 - 1
        # x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
        if np.random.random() < 0.5:
            beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
            r_idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            x_r = pop[r_idx][self.ID_POS]
            # x_r = pop[np.random.randint(0, self.pop_size-1)][self.ID_POS]
            if np.random.random() < 0.5:
                x_new = beta * x_r + (1 - beta) * pop[idx][self.ID_POS]
            else:
                x_new = (1 - beta) * x_r + beta * pop[idx][self.ID_POS]
        else:
            x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[idx][self.ID_POS])
            # x_new = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
        pos_new = self.amend_position_faster(x_new)
        fit_new = self.get_fitness_position(pos_new)
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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size - 1))

        ## Production - Update the worst agent
        # Eq. 13
        a = 2 * (1 - (epoch + 1) / self.epoch)
        x1 = (1 - a) * pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position_faster(x1)
        fit_new = self.get_fitness_position(x1)
        pop[-1] = [pos_new, fit_new]

        ## Consumption - Update the whole population left
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy) for idx in pop_idx]
        pop_new.append(pop[-1])

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(pop_new)
        pop_idx = np.array(range(0, self.pop_size))

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best, pop=pop_new, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best, pop=pop_new, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child2(idx, best, pop_new, epoch) for idx in pop_idx]
        return pop_new


class ModifiedAEO(Optimizer):
    """
    Original version of: Modified Artificial Ecosystem-Based Optimization
        (Effective Parameter Extraction of Different Polymer Electrolyte Membrane Fuel Cell Stack Models Using a
            Modified Artificial Ecosystem Optimization Algorithm)
    Link:
        https://doi.org/10.1109/ACCESS.2020.2973351
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, H):
        rand = np.random.random()
        # Eq. 4, 5, 6
        v1 = np.random.normal(0, 1)
        v2 = np.random.normal(0, 1)
        c = 0.5 * v1 / abs(v2)  # Consumption factor
        if idx == 0:
            j = 1
        else:
            j = np.random.randint(0, idx)
        ### Herbivore
        if rand <= 1.0 / 3:  # Eq. 23
            pos_new = pop[idx][self.ID_POS] + H * c * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])
        ### Carnivore
        elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 24
            pos_new = pop[idx][self.ID_POS] + H * c * (pop[idx][self.ID_POS] - pop[j][self.ID_POS])
        ### Omnivore
        else:  # Eq. 25
            r5 = np.random.random()
            pos_new = pop[idx][self.ID_POS] + H * c * (r5 * (pop[idx][self.ID_POS] - pop[0][self.ID_POS]) +
                                           (1 - r5) * (pop[idx][self.ID_POS] - pop[j][self.ID_POS]))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, best, pop, epoch):
        r3 = np.random.uniform()
        d = 3 * np.random.normal(0, 1)
        e = r3 * np.random.randint(1, 3) - 1
        h = 2 * r3 - 1
        # x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
        if np.random.random() < 0.5:
            beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
            r_idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            x_r = pop[r_idx][self.ID_POS]
            # x_r = pop[np.random.randint(0, self.pop_size-1)][self.ID_POS]
            if np.random.random() < 0.5:
                x_new = beta * x_r + (1 - beta) * pop[idx][self.ID_POS]
            else:
                x_new = (1 - beta) * x_r + beta * pop[idx][self.ID_POS]
        else:
            x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * pop[idx][self.ID_POS])
            # x_new = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
        pos_new = self.amend_position_faster(x_new)
        fit_new = self.get_fitness_position(pos_new)
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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size - 1))

        ## Production
        # Eq. 22
        H = 2 * (1 - (epoch + 1) / self.epoch)
        a = (1 - (epoch + 1) / self.epoch) * np.random.random()
        x1 = (1 - a) * pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position_faster(x1)
        fit_new = self.get_fitness_position(pos_new)
        pop[-1] = [pos_new, fit_new]

        ## Consumption - Update the whole population left
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, H=H), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, H=H), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, H) for idx in pop_idx]
        pop_new.append(pop[-1])

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(pop_new)
        pop_idx = np.array(range(0, self.pop_size))

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best, pop=pop_new, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, best=best, pop=pop_new, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child2(idx, best, pop_new, epoch) for idx in pop_idx]
        return pop_new


class AdaptiveAEO(Optimizer):
    """
        This is Adaptive Artificial Ecosystem Optimization based on
            + Linear weight factor reduce from 2 to 0 through time
            + Levy_flight
            + Global best solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best, epoch, wf):
        if np.random.random() < 0.5:
            rand = np.random.random()
            # Eq. 4, 5, 6
            c = 0.5 * np.random.normal(0, 1) / abs(np.random.normal(0, 1))  # Consumption factor

            if idx == 0:
                j = 1
            else:
                j = np.random.randint(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                pos_new = pop[idx][self.ID_POS] + wf * c * (pop[idx][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
            ### Omnivore
            elif 1.0 / 3 <= rand <= 2.0 / 3:
                pos_new = pop[idx][self.ID_POS] + wf * c * (pop[idx][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
            ### Carnivore
            else:
                r2 = np.random.uniform()
                pos_new = pop[idx][self.ID_POS] + wf * c * (r2 * (pop[idx][self.ID_POS] - pop[0][self.ID_POS]) +
                                                         (1 - r2) * (pop[idx][self.ID_POS] - pop[j][self.ID_POS]))
        else:
            pos_new = pop[idx][self.ID_POS] + self.get_levy_flight_step(0.001, 1., case=-1) * \
                      (1.0 / np.sqrt(epoch + 1)) * np.sign(np.random.random() - 0.5) * (pop[idx][self.ID_POS] - g_best[self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, pop, g_best, local_best, epoch):
        if np.random.random() < 0.5:
            pos_new = local_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * (local_best[self.ID_POS] - pop[idx][self.ID_POS])
        else:
            pos_new = g_best[self.ID_POS] + self.get_levy_flight_step(0.001, 0.75, case=-1) * \
                      1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
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
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size - 1))

        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - (epoch + 1) / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * np.random.random()
        x1 = (1 - a) * pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position_faster(x1)
        fit_new = self.get_fitness_position(x1)
        pop[-1] = [pos_new, fit_new]

        ## Consumption - Update the whole population left
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, epoch=epoch, wf=wf), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best, epoch=epoch, wf=wf), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child(idx, pop_copy, g_best, epoch, wf) for idx in pop_idx]
        pop_new.append(pop[-1])

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(pop_new)
        pop_idx = np.array(range(0, self.pop_size))

        ## Decomposition
        ### Eq. 10, 11, 12, 9   idx, pop, g_best, local_best
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop_new, g_best=g_best, local_best=best, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop_new, g_best=g_best, local_best=best, epoch=epoch), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            pop_new = [self.create_child2(idx, pop_new, g_best, best, epoch) for idx in pop_idx]
        return pop_new

#
# class BaseAEO(Root):
#     """
#     This is my modified version of: Artificial Ecosystem-based Optimization
#         (Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm)
#
#         + Same results and sometime better than original version.
#         + Original version move the population at the same time. My version move after each position move.
#     """
#
#     def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
#         super().__init__(obj_func, lb, ub, verbose, kwargs)
#         self.epoch = epoch
#         self.pop_size = pop_size
#
#     def __update_population_and_global_best__(self, pop, g_best):
#         pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
#         current_best = deepcopy(pop[self.ID_MAX_PROB])
#         if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
#             g_best = deepcopy(current_best)
#         return pop, g_best
#
#     def train(self):
#         pop = [self.create_solution() for _ in range(self.pop_size)]
#         # Sorted population in the descending order of the function fitness value
#         pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
#         g_best = deepcopy(pop[self.ID_MAX_PROB])
#
#         for epoch in range(self.epoch):
#             ## Production
#             # Eq. 2, 3, 1
#             a = (1.0 - epoch / self.epoch) * uniform()
#             x1 = (1 - a) * pop[self.ID_MAX_PROB][self.ID_POS] + a * uniform(self.lb, self.ub)
#             fit = self.get_fitness_position(x1)
#             pop[0] = [x1, fit]
#
#             ## Consumption
#             for i in range(1, self.pop_size):
#                 rand = uniform()
#                 # Eq. 4, 5, 6
#                 v1 = normal(0, 1)
#                 v2 = normal(0, 1)
#                 c = 0.5 * v1 / abs(v2)      # Consumption factor
#
#                 if i == 1:
#                     x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
#                 else:
#                     j = randint(1, i)
#                     ### Herbivore
#                     if rand < 1.0/3:
#                         x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])    # Eq. 6
#                     ### Carnivore
#                     elif 1.0/3 <= rand and rand <= 2.0/3:
#                         x_t1 = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])    # Eq. 7
#                     ### Omnivore
#                     else:
#                         r2 = uniform()
#                         x_t1 = pop[i][self.ID_POS] + c * (r2*(pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1-r2)*(pop[i][self.ID_POS] - pop[j][self.ID_POS]))
#                 fit_t1 = self.get_fitness_position(x_t1)
#                 if fit_t1 < pop[i][self.ID_FIT]:
#                     pop[i] = [x_t1, fit_t1]
#
#                 if self.batch_idea:
#                     if (i + 1) % self.batch_size == 0:
#                         pop, g_best = self.__update_population_and_global_best__(pop, g_best)
#                 else:
#                     if (i + 1) % self.pop_size == 0:
#                         pop, g_best = self.__update_population_and_global_best__(pop, g_best)
#
#             ## Decomposition
#             ### Eq. 10, 11, 12, 9
#             for i in range(0, self.pop_size):
#                 r3 = uniform()
#                 d = 3 * normal(0, 1)
#                 e = r3 * randint(1, 3) - 1
#                 h = 2 * r3 - 1
#                 x_t1 = pop[self.ID_MAX_PROB][self.ID_POS] + d * (e*pop[self.ID_MAX_PROB][self.ID_POS] - h*pop[i][self.ID_POS])
#                 fit_t1 = self.get_fitness_position(x_t1)
#                 if fit_t1 < pop[i][self.ID_FIT]:
#                     pop[i] = [x_t1, fit_t1]
#
#                 if self.batch_idea:
#                     if (i+1) % self.batch_size == 0:
#                         pop, g_best = self.__update_population_and_global_best__(pop, g_best)
#                 else:
#                     if (i + 1) % self.pop_size == 0:
#                         pop, g_best = self.__update_population_and_global_best__(pop, g_best)
#
#             self.loss_train.append(g_best[self.ID_FIT])
#             if self.verbose:
#                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
#
#         self.solution = g_best
#         return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
