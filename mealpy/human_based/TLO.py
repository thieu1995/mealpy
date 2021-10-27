#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:14, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial, reduce
import numpy as np
from mealpy.optimizer import Optimizer
import time


class BaseTLO(Optimizer):
    """
        Teaching-Learning-based Optimization (TLO)
    An elitist teaching-learning-based optimization algorithm for solving complex constrained optimization problems(TLO)
        This is my version taken the advantages of numpy np.array to faster handler operations.
    Notes:
        + Remove all third loop
        + Using global best solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, g_best):
        ## Teaching Phrase
        TF = np.random.randint(1, 3)  # 1 or 2 (never 3)
        list_pos = np.array([item[self.ID_POS] for item in pop])
        DIFF_MEAN = np.random.rand(self.problem.n_dims) * (g_best[self.ID_POS] - TF * np.mean(list_pos, axis=0))
        temp = pop[idx][self.ID_POS] + DIFF_MEAN
        pos_new = self.amend_position_faster(temp)
        fit_new = self.get_fitness_position(temp)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            pop[idx] = [pos_new, fit_new]

        ## Learning Phrase
        temp = pop[idx][self.ID_POS].copy()
        id_partner = np.random.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))
        # arr_random = np.random.rand(self.problem.n_dims)
        if pop[idx][self.ID_FIT] < pop[id_partner][self.ID_FIT]:
            temp += np.random.rand(self.problem.n_dims) * (pop[idx][self.ID_POS] - pop[id_partner][self.ID_POS])
        else:
            temp += np.random.rand(self.problem.n_dims) * (pop[id_partner][self.ID_POS] - pop[idx][self.ID_POS])
        pos_new = self.amend_position_faster(temp)
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
        pop_idx = np.array(range(0, self.pop_size))

        ## Reproduction
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop_copy, g_best=g_best), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop_copy, g_best=g_best) for idx in pop_idx]
        return child


class OriginalTLO(BaseTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)
        Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems
    This is slower version which inspired from this version:
        https://github.com/andaviaco/tblo
    Notes:
        + Removed the third loop to make it faster
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = False

    def create_child(self, idx, pop, g_best):
        ## Teaching Phrase
        TF = np.random.randint(1, 3)  # 1 or 2 (never 3)
        #### Remove third loop here
        list_pos = np.array([item[self.ID_POS] for item in pop])
        pos_new = pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (g_best[self.ID_POS] - TF * np.mean(list_pos, axis=0))
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            pop[idx] = [pos_new, fit_new]

        ## Learning Phrase
        id_partner = np.random.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))

        #### Remove third loop here
        if self.compare_agent(pop[idx], pop[id_partner]):
            diff = pop[idx][self.ID_POS] - pop[id_partner][self.ID_POS]
        else:
            diff = pop[id_partner][self.ID_POS] - pop[idx][self.ID_POS]
        pos_new = pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * diff
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()


class ITLO(BaseTLO):
    """
    My version of: Improved Teaching-Learning-based Optimization (ITLO)
    Link:
        An improved teaching-learning-based optimization algorithm for solving unconstrained optimization problems
    Notes:
        + Kinda similar to the paper, but the pseudo-code in the paper is not clear.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_teachers=5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_teachers (int): number of teachers in class
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.n_teachers = n_teachers  # Number of teams / group
        self.n_students = pop_size - n_teachers
        self.n_students_in_team = int(self.n_students / self.n_teachers)

    def classify(self, pop):
        sorted_pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        best = sorted_pop[0].copy()
        teachers = sorted_pop[:self.n_teachers]
        sorted_pop = sorted_pop[self.n_teachers:]
        idx_list = np.random.permutation(range(0, self.n_students))
        teams = []
        for id_teacher in range(0, self.n_teachers):
            group = []
            for idx in range(0, self.n_students_in_team):
                start_index = id_teacher * self.n_students_in_team + idx
                group.append(sorted_pop[idx_list[start_index]])
            teams.append(group)
        return teachers, teams, best

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
            print("ImprovedTLO is support sequential mode only!")
            exit(0)
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        teachers, teams, g_best = self.classify(pop)
        self.history.save_initial_best(g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            for id_teach, teacher in enumerate(teachers):
                team = teams[id_teach]
                list_pos = np.array([student[self.ID_POS] for student in teams[id_teach]])  # Step 7
                mean_team = np.mean(list_pos, axis=0)
                for id_stud, student in enumerate(team):
                    if teacher[self.ID_FIT][self.ID_TAR] == 0:
                        TF = 1
                    else:
                        TF = student[self.ID_FIT][self.ID_TAR] / teacher[self.ID_FIT][self.ID_TAR]
                    diff_mean = np.random.rand() * (teacher[self.ID_POS] - TF * mean_team)  # Step 8

                    id2 = np.random.choice(list(set(range(0, self.n_teachers)) - {id_teach}))
                    if self.compare_agent(teacher, team[id2]):
                        pos_new = (student[self.ID_POS] + diff_mean) + np.random.rand() * (team[id2][self.ID_POS] - student[self.ID_POS])
                    else:
                        pos_new = (student[self.ID_POS] + diff_mean) + np.random.rand() * (student[self.ID_POS] - team[id2][self.ID_POS])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if self.compare_agent([pos_new, fit_new], student):
                        teams[id_teach][id_stud] = [pos_new, fit_new]

            for id_teach, teacher in enumerate(teachers):
                ef = round(1 + np.random.rand())
                team = teams[id_teach]
                for id_stud, student in enumerate(team):
                    id2 = np.random.choice(list(set(range(0, self.n_students_in_team)) - {id_stud}))
                    if self.compare_agent(student, team[id2]):
                        pos_new = student[self.ID_POS] + np.random.rand() * (student[self.ID_POS] - team[id2][self.ID_POS]) + \
                                  np.random.rand() * (teacher[self.ID_POS] - ef * team[id2][self.ID_POS])
                    else:
                        pos_new = student[self.ID_POS] + np.random.rand() * (team[id2][self.ID_POS] - student[self.ID_POS]) + \
                                  np.random.rand() * (teacher[self.ID_POS] - ef * student[self.ID_POS])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if self.compare_agent([pos_new, fit_new], student):
                        teams[id_teach][id_stud] = [pos_new, fit_new]

            for id_teach, teacher in enumerate(teachers):
                team = teams[id_teach] + [teacher]
                team = sorted(team, key=lambda item: item[self.ID_FIT])
                teachers[id_teach] = team[0]
                teams[id_teach] = team[1:]

            pop = teachers + reduce(lambda x, y: x + y, teams)
            # update global best position
            _, g_best = self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
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
