#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:58, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy
from mealpy.utils.history import History
from mealpy.problem import Problem
from mealpy.utils.termination import Termination
import concurrent.futures as parallel
import time


class Optimizer:
    """ This is base class of all Algorithms """

    ## Assumption the A solution with format: [position, [target, [obj1, obj2, ...]]]
    ID_POS = 0  # Index of position/location of solution/agent
    ID_FIT = 1  # Index of fitness value of solution/agent

    ID_TAR = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in fitness

    EPSILON = 10E-10

    def __init__(self, problem, kwargs):
        """
        Args:
            problem: Design your problem based on the format of the Problem class

        Examples:
            problem = {
                "obj_func": your objective function,
                "lb": list of value
                "ub": list of value
                "minmax": "min" or "max"
                "verbose": True or False
                "n_dims": int (Optional)
                "batch_idea": True or False (Optional)
                "batch_size": int (Optional, smaller than population size)
                "obj_weight": list weights for all your objectives (Optional, default = [1, 1, ...1])
             }
        """
        super(Optimizer, self).__init__()
        self.epoch, self.pop_size, self.solution = None, None, None
        self.mode = "sequential"
        self.pop, self.g_best = None, None
        self.history = History()
        if not isinstance(problem, Problem):
            problem = Problem(problem)
        self.problem = problem
        self.verbose = problem.verbose
        self.termination_flag = False       # Check if exist object or not
        if "termination" in kwargs:
            termination = kwargs["termination"]
            if not isinstance(termination, Termination):
                print("Please create and input your Termination object!")
                exit(0)
            else:
                self.termination = termination
            self.termination_flag = True
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def termination_start(self):
        if self.termination_flag:
            if self.termination.mode == 'TB':
                self.count_terminate = time.time()
            elif self.termination.mode == 'ES':
                self.count_terminate = 0
            elif self.termination.mode == 'MG':
                self.count_terminate = self.epoch
            else:                       # number of function evaluation (NFE)
                self.count_terminate = self.pop_size        # First out of loop
        else:
            pass

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        if self.sort_flag:
            self.pop, self.g_best = self.get_global_best_solution(self.pop)  # We sort the population
        else:
            _, self.g_best = self.get_global_best_solution(self.pop)  # We don't sort the population

    def before_evolve(self, epoch):
        pass

    def after_evolve(self, epoch):
        pass

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
        self.mode = mode
        self.termination_start()
        self.initialization()
        self.history.save_initial_best(self.g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Call before evolve function
            self.before_evolve(epoch)

            ## Evolve method will be called in child class
            self.evolve(epoch)

            ## Call after evolve function
            self.after_evolve(epoch)

            # update global best position
            if self.sort_flag:
                self.pop, self.g_best = self.update_global_best_solution(self.pop)  # We sort the population
            else:
                _, self.g_best = self.update_global_best_solution(self.pop)  # We don't sort the population
            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(deepcopy(self.pop))
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
                else:                       # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

    def evolve(self, epoch):
        pass

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def create_population(self, pop_size=None):
        """
        Args:
            mode (str): processing mode, it can be "sequential", "thread" or "process"
            pop_size (int): number of solutions

        Returns:
            population: list of solutions/agents
        """
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                list_executors = [executor.submit(self.create_solution) for _ in range(pop_size)]
                # This method yield the result everytime a thread finished their job (not by order)
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                list_executors = [executor.submit(self.create_solution) for _ in range(pop_size)]
                # This method yield the result everytime a cpu finished their job (not by order).
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        else:
            pop = [self.create_solution() for _ in range(0, pop_size)]
        return pop

    def update_fitness_population(self, pop=None):
        """
        Args:
            mode (str): processing mode, it can be "sequential", "thread" or "process"
            pop (list): the population

        Returns:
            population: with updated fitness value
        """
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                list_results = executor.map(self.get_fitness_solution, pop)  # Return result not the future object
                for idx, fit in enumerate(list_results):
                    pop[idx][self.ID_FIT] = fit
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                list_results = executor.map(self.get_fitness_solution, pop)  # Return result not the future object
                for idx, fit in enumerate(list_results):
                    pop[idx][self.ID_FIT] = fit
        else:
            for idx, agent in enumerate(pop):
                pop[idx][self.ID_FIT] = self.get_fitness_solution(agent)
        return pop

    def get_fitness_position(self, position=None):
        """
        Args:
            position (nd.array): 1-D numpy array

        Returns:
            [target, [obj1, obj2, ...]]
        """
        objs = self.problem.obj_func(position)
        if not self.problem.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.problem.obj_weight)
        return [fit, objs]

    def get_fitness_solution(self, solution=None):
        """
        Args:
            solution (list): A solution with format [position, [target, [obj1, obj2, ...]]]

        Returns:
            [target, [obj1, obj2, ...]]
        """
        return self.get_fitness_position(solution[self.ID_POS])

    def get_global_best_solution(self, pop: list):
        """
        Sort population and return the sorted population and the best solution

        Args:
            pop (list): The population of pop_size individuals

        Returns:
            Sorted population and global best solution
        """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])  # Already returned a new sorted list
        if self.problem.minmax == "min":
            return sorted_pop, deepcopy(sorted_pop[0])
        else:
            return sorted_pop, deepcopy(sorted_pop[-1])

    def get_better_solution(self, agent1: list, agent2: list):
        """
        Args:
            agent1 (list): A solution
            agent2 (list): Another solution

        Returns:
            The better solution between them
        """
        if self.problem.minmax == "min":
            if agent1[self.ID_FIT][self.ID_TAR] < agent2[self.ID_FIT][self.ID_TAR]:
                return deepcopy(agent1)
            return deepcopy(agent2)
        else:
            if agent1[self.ID_FIT][self.ID_TAR] < agent2[self.ID_FIT][self.ID_TAR]:
                return deepcopy(agent2)
            return deepcopy(agent1)

    def compare_agent(self, agent_a: list, agent_b: list):
        """
        Args:
            agent_a (list): Solution a
            agent_b (list): Solution b

        Returns:
            boolean: Return True if solution a better than solution b and otherwise
        """
        if self.problem.minmax == "min":
            if agent_a[self.ID_FIT][self.ID_TAR] < agent_b[self.ID_FIT][self.ID_TAR]:
                return True
            return False
        else:
            if agent_a[self.ID_FIT][self.ID_TAR] < agent_b[self.ID_FIT][self.ID_TAR]:
                return False
            return True

    def get_special_solutions(self, pop=None, best=3, worst=3):
        """
        Args:
            pop (list): The population
            best (int): Top k1 best solutions, default k1=3, it can be None
            worst (int): Top k2 worst solutions, default k2=3, it can be None

        Returns:
            sorted_population, k1 best solutions and k2 worst solutions
        """
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=True)
        if best is None:
            if worst is None:
                exit(0)
            else:
                return pop, None, deepcopy(pop[:-worst])
        else:
            if worst is None:
                return pop, deepcopy(pop[:best]), None
            else:
                return pop, deepcopy(pop[:best]), deepcopy(pop[:-worst])

    def get_special_fitness(self, pop=None):
        """
        Args:
            pop (list): The population

        Returns:
            Total fitness, best fitness, worst fitness
        """
        total_fitness = np.sum([agent[self.ID_FIT][self.ID_TAR] for agent in pop])
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=True)
        return total_fitness, pop[0][self.ID_FIT][self.ID_TAR], pop[-1][self.ID_FIT][self.ID_TAR]

    def update_global_best_solution(self, pop=None, save=True):
        """
        Update the global best solution saved in variable named: self.history_list_g_best
        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current global best and False if you just want update the current one.

        Returns:
            Sorted population and the global best solution
        """
        if self.problem.minmax == "min":
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        else:
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=True)
        current_best = sorted_pop[0]
        # self.history_list_c_best.append(current_best)
        # better = self.get_better_solution(current_best, self.history_list_g_best[-1])
        # self.history_list_g_best.append(better)
        if save:
            self.history.list_current_best.append(current_best)
            better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best.append(better)
            return deepcopy(sorted_pop), deepcopy(better)
        else:
            local_better = self.get_better_solution(current_best, self.history.list_current_best[-1])
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best[-1] = global_better
            return deepcopy(sorted_pop), deepcopy(global_better)

    def print_epoch(self, epoch, runtime):
        """
        Print out the detailed information of training process
        Args:
            epoch (int): current iteration
            runtime (float): the runtime for current iteration
        """
        if self.verbose:
            print(f"> Epoch: {epoch}, Current best: {self.history.list_current_best[-1][self.ID_FIT][self.ID_TAR]}, "
                  f"Global best: {self.history.list_global_best[-1][self.ID_FIT][self.ID_TAR]}, Runtime: {runtime:.5f} seconds")

    def save_optimization_process(self):
        """
        Detail: Save important data for later use such as:
            + history_list_g_best_fit
            + history_list_c_best_fit
            + history_list_div
            + history_list_explore
            + history_list_exploit
        """
        # self.history_list_g_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history_list_g_best]
        # self.history_list_c_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history_list_c_best]
        #
        # # Draw the exploration and exploitation line with this data
        # self.history_list_div = np.ones(self.epoch)
        # for idx, pop in enumerate(self.history_list_pop):
        #     pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
        #     div = np.mean(abs((np.median(pos_matrix, axis=0) - pos_matrix)), axis=0)
        #     self.history_list_div[idx] = np.mean(div, axis=0)
        # div_max = np.max(self.history_list_div)
        # self.history_list_explore = 100 * (self.history_list_div / div_max)
        # self.history_list_exploit = 100 - self.history_list_explore

        self.history.epoch = len(self.history.list_global_best)
        self.history.list_global_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history.list_global_best]
        self.history.list_current_best_fit = [agent[self.ID_FIT][self.ID_TAR] for agent in self.history.list_current_best]

        # Draw the exploration and exploitation line with this data
        self.history.list_diversity = np.ones(self.history.epoch)
        for idx, pop in enumerate(self.history.list_population):
            pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
            div = np.mean(abs((np.median(pos_matrix, axis=0) - pos_matrix)), axis=0)
            self.history.list_diversity[idx] = np.mean(div, axis=0)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (self.history.list_diversity / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.solution = self.history.list_global_best[-1]

    ## Crossover techniques
    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        """
        This method can handle min/max problem, and negative or positive fitness value.
        Args:
            list_fitness (nd.array): 1-D numpy array

        Returns:
            Index of selected solution
        """
        scaled_fitness = (list_fitness - np.min(list_fitness)) / (np.ptp(list_fitness) + self.EPSILON)
        if self.problem.minmax == "min":
            final_fitness = 1.0 - scaled_fitness
        else:
            final_fitness = scaled_fitness
        total_sum = sum(final_fitness)
        r = np.random.uniform(low=0, high=total_sum)
        for idx, f in enumerate(final_fitness):
            r = r + f
            if r > total_sum:
                return idx
        return np.random.choice(range(0, len(list_fitness)))

    def get_solution_kway_tournament_selection(self, pop: list, k_way=0.2, output=2):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        k_way = round(k_way)
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        if self.problem.minmax == "min":
            return list_parents[:output]
        else:
            return list_parents[-output:]

    def get_levy_flight_step(self, beta=1.0, multiplier=0.001, case=0):
        """
        Parameters
        ----------
        multiplier (float, optional): 0.01
        beta: [0-2]
            + 0-1: small range --> exploit
            + 1-2: large range --> explore
        case: 0, 1, -1
            + 0: return multiplier * s * np.random.uniform()
            + 1: return multiplier * s * np.random.normal(0, 1)
            + -1: return multiplier * s
        """
        # u and v are two random variables which follow np.random.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        u = np.random.normal(0, sigma_u ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = u / np.power(abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * np.random.uniform()
        elif case == 1:
            step = multiplier * s * np.random.normal(0, 1)
        else:
            step = multiplier * s
        return step

    def levy_flight(self, epoch=None, position=None, g_best_position=None, step=0.001, case=0):
        """
        Parameters
        ----------
        epoch (int): current iteration
        position : 1-D numpy np.array
        g_best_position : 1-D numpy np.array
        step (float, optional): 0.001
        case (int, optional): 0, 1, 2

        """
        beta = 1
        # muy and v are two random variables which follow np.random.normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = muy / np.power(abs(v), 1 / beta)
        levy = np.random.uniform(self.problem.lb, self.problem.ub) * step * s * (position - g_best_position)

        if case == 0:
            return levy
        elif case == 1:
            return position + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * levy
        elif case == 2:
            return position + np.random.normal(0, 1, len(self.problem.lb)) * levy
        elif case == 3:
            return position + 0.01 * levy

    def amend_position(self, position=None):
        """
        Args:
            position (): vector position (location) of the solution.

        Returns:
            Amended position (make the position is in bound)
        """
        return np.maximum(self.problem.lb, np.minimum(self.problem.ub, position))

    def amend_position_faster(self, position=None):
        """
        This is method is faster than "amend_position" in most cases.
        Args:
            position (): vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.clip(position, self.problem.lb, self.problem.ub)

    def amend_position_random(self, position=None):
        """
        If solution out of bound at dimension x, then it will re-arrange to random location in the range of domain
        Args:
            position (): vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.where(np.logical_and(self.problem.lb <= position, position <= self.problem.ub),
                        position, np.random.uniform(self.problem.lb, self.problem.ub))

    def get_global_best_global_worst_solution(self, pop=None):
        """
        Args:
            pop (): The population

        Returns:
            The global best and the global worst solution
        """
        # Already returned a new sorted list
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        if self.problem.minmax == "min":
            return deepcopy(sorted_pop[0]), deepcopy(sorted_pop[-1])
        else:
            return deepcopy(sorted_pop[-1]), deepcopy(sorted_pop[0])

    ### Survivor Selection
    def greedy_selection_population(self, pop_old=None, pop_new=None):
        """
        Args:
            pop_old (): The current population
            pop_new (): The next population

        Returns:
            The new population with better solutions
        """
        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            print("Pop old and Pop new should be the same length!")
            exit(0)
        if self.problem.minmax == "min":
            return [pop_new[i] if pop_new[i][self.ID_FIT][self.ID_TAR] < pop_old[i][self.ID_FIT][self.ID_TAR]
                    else pop_old[i] for i in range(len_old)]
        else:
            return [pop_new[i] if pop_new[i][self.ID_FIT] > pop_old[i][self.ID_FIT]
                    else pop_old[i] for i in range(len_old)]

    def get_sorted_strim_population(self, pop=None, pop_size=None, reverse=False):
        """
        Args:
            pop (list): The population
            pop_size (int): The number of population

        Returns:
            The sorted population with pop_size size
        """
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=reverse)
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=reverse)
        return pop[:pop_size]

    def create_opposition_position(self, agent=None, g_best=None):
        """
        Args:
            agent (): The current agent
            g_best (): the global best solution

        Returns:
            The opposite solution
        """
        return self.problem.lb + self.problem.ub - g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - agent[self.ID_POS])

    def get_parent_kway_tournament_selection(self, pop=None, k_way=0.2, output=2):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[:output]

    ### Crossover
    def crossover_arthmetic_recombination(self, dad_pos=None, mom_pos=None):
        r = np.random.uniform()           # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
        return w1, w2


    #### Improved techniques can be used in any algorithms: 1
    ## Based on this paper: An efficient equilibrium optimizer with mutation strategy for numerical optimization (but still different)
    ## This scheme used after the original and including 4 step:
    ##  s1: sort population, take p1 = 1/2 best population for next round
    ##  s2: do the mutation for p1, using greedy method to select the better solution
    ##  s3: do the search mechanism for p1 (based on global best solution and the updated p1 above), to make p2 population
    ##  s4: construct the new population for next generation
    def improved_ms(self, pop=None, g_best=None):    ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]

        ## Mutation scheme
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s1[i])
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))
            agent[self.ID_POS] = self.amend_position_faster(pos_new)
            pop_new.append(agent)
        pop_new = self.update_fitness_population(pop_new)
        pop_s1 = self.greedy_selection_population(pop_s1, pop_new)  ## Greedy method --> improved exploitation

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s2[i])
            pos_new = (g_best[self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            agent[self.ID_POS] = self.amend_position_faster(pos_new)
            pop_new.append(agent)
        ## Keep the diversity of populatoin and still improved the exploration
        pop_s2 = self.update_fitness_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new)

        ## Construct a new population
        pop = pop_s1 + pop_s2
        return pop
