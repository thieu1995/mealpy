#!/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from typing import List, Union, Tuple, Dict
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem
from math import gamma
from mealpy.utils.history import History
from mealpy.utils.target import Target
from mealpy.utils.termination import Termination
from mealpy.utils.logger import Logger
from mealpy.utils.validator import Validator
import concurrent.futures as parallel
from functools import partial
import os
import time


class Optimizer:
    """
    The base class of all algorithms. All methods in this class will be inherited

    Notes
    ~~~~~
    + The function solve() is the most important method, trained the model
    + The parallel (multithreading or multiprocessing) is used in method: generate_population(), update_target_for_population()
    + The general format of:
        + population = [agent_1, agent_2, ..., agent_N]
        + agent = [solution, target]
        + target = [fitness value, objective_list]
        + objective_list = [obj_1, obj_2, ..., obj_M]
    """

    EPSILON = 10E-10
    SUPPORTED_MODES = ["process", "thread", "swarm", "single"]
    AVAILABLE_MODES = ["process", "thread", "swarm"]
    PARALLEL_MODES = ["process", "thread"]
    SUPPORTED_ARRAYS = [list, tuple, np.ndarray]

    def __init__(self, **kwargs):
        super(Optimizer, self).__init__()
        self.epoch, self.pop_size = None, None
        self.mode, self.n_workers, self.name = None, None, None
        self.pop, self.g_best, self.g_worst = None, Agent(), None
        self.problem, self.logger, self.history = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)

        if self.name is None: self.name = self.__class__.__name__
        self.sort_flag = False
        self.nfe_counter = -1  # The first one is tested in Problem class
        self.parameters, self.params_name_ordered = {}, None
        self.is_parallelizable = True

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_parameters(self, parameters: Union[List, Tuple, Dict]) -> None:
        """
        Set the parameters for current optimizer.

        if paras is a list of parameter's name, then it will set the default value in optimizer as current parameters
        if paras is a dict of parameter's name and value, then it will override the current parameters

        Args:
            parameters: The parameters
        """
        if type(parameters) in (list, tuple):
            self.params_name_ordered = tuple(parameters)
            self.parameters = {}
            for name in parameters:
                self.parameters[name] = self.__dict__[name]

        if type(parameters) is dict:
            valid_para_names = set(self.parameters.keys())
            new_para_names = set(parameters.keys())
            if new_para_names.issubset(valid_para_names):
                for key, value in parameters.items():
                    setattr(self, key, value)
                    self.parameters[key] = value
            else:
                raise ValueError(f"Invalid input parameters: {new_para_names} for {self.get_name()} optimizer. "
                                 f"Valid parameters are: {valid_para_names}.")

    def get_parameters(self) -> Dict:
        """
        Get parameters of optimizer.
        """
        return self.parameters

    def get_attributes(self) -> Dict:
        """
        Get all attributes in optimizer.
        """
        return self.__dict__

    def get_name(self) -> str:
        """
        Get name of the optimizer
        """
        return self.name

    def __str__(self):
        temp = ""
        for key in self.params_name_ordered:
            temp += f"{key}={self.parameters[key]}, "
        temp = temp[:-2]
        return f"{self.__class__.__name__}({temp})"

    def initialize_variables(self):
        pass

    def before_initialization(self, starting_solutions: Union[List, Tuple, np.ndarray] = None) -> None:
        """
        Args:
            starting_solutions: The starting solutions (not recommended)
        """
        if starting_solutions is None:
            pass
        elif type(starting_solutions) in self.SUPPORTED_ARRAYS and len(starting_solutions) == self.pop_size:
            if type(starting_solutions[0]) in self.SUPPORTED_ARRAYS and len(starting_solutions[0]) == self.problem.n_dims:
                self.pop = [self.generate_agent(solution) for solution in starting_solutions]
            else:
                raise ValueError("Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only.")
        else:
            raise ValueError("Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size.")

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

    def after_initialization(self) -> None:
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        self.g_best, self.g_worst = best[0], worst[0]
        if self.sort_flag: self.pop = pop_temp
        ## Store initial best and worst solutions
        self.history.store_initial_best_worst(self.g_best, self.g_worst)

    def before_main_loop(self):
        pass

    def evolve(self, epoch: int) -> None:
        pass

    def check_problem(self, problem, seed) -> None:
        if isinstance(problem, Problem):
            problem.set_seed(seed)
            self.problem = problem
        elif type(problem) == dict:
            problem["seed"] = seed
            self.problem = Problem(**problem)
        else:
            raise ValueError("problem needs to be a dict or an instance of Problem class.")
        self.generator = np.random.default_rng(seed)
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self.problem.msg)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)
        self.pop, self.g_best, self.g_worst = None, None, None

    def check_mode_and_workers(self, mode, n_workers):
        self.mode = self.validator.check_str("mode", mode, self.SUPPORTED_MODES)
        if self.mode in self.PARALLEL_MODES:
            if not self.is_parallelizable:
                self.logger.warning(f"{self.get_name()} doesn't support parallelization. The default mode 'single' is activated.")
                self.mode = "single"
            elif n_workers is not None:
                if self.mode == "process":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
                if self.mode == "thread":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(32, os.cpu_count() + 4)])
            else:
                self.logger.warning(f"The parallel mode: {self.mode} is selected. But n_workers is not set. The default n_workers = 4 is used.")
                self.n_workers = 4

    def check_termination(self, mode="start", termination=None, epoch=None):
        if mode == "start":
            self.termination = termination
            if termination is not None:
                if isinstance(termination, Termination):
                    self.termination = termination
                elif type(termination) == dict:
                    self.termination = Termination(log_to=self.problem.log_to, log_file=self.problem.log_file, **termination)
                else:
                    raise ValueError("Termination needs to be a dict or an instance of Termination class.")
                self.nfe_counter = 0
                self.termination.set_start_values(0, self.nfe_counter, time.perf_counter(), 0)
        else:
            finished = False
            if self.termination is not None:
                es = self.history.get_global_repeated_times(self.termination.epsilon)
                finished = self.termination.should_terminate(epoch, self.nfe_counter, time.perf_counter(), es)
                if finished:
                    self.logger.warning(self.termination.message)
            return finished

    def solve(self, problem: Union[Dict, Problem] = None, mode: str = 'single', n_workers: int = None,
              termination: Union[Dict, Termination] = None, starting_solutions: Union[List, np.ndarray, Tuple] = None,
              seed: int = None) -> Agent:
        """
        Args:
            problem: an instance of Problem class or a dictionary
            mode: Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, this is default mode

            n_workers: The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination: The termination dictionary or an instance of Termination class
            starting_solutions: List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            seed: seed for random number generation needed to be *explicitly* set to int value

        Returns:
            g_best: g_best, the best found agent, that hold the best solution and the best target. Access by: .g_best.solution, .g_best.target
        """
        self.check_problem(problem, seed)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()
        for epoch in range(1, self.epoch + 1):
            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch)

            # Update global best solution, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_agent(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch, time_epoch)
            if self.check_termination("end", None, epoch):
                break
        self.track_optimize_process()
        return self.g_best

    def track_optimize_step(self, pop: List[Agent] = None, epoch: int = None, runtime: float = None) -> None:
        """
        Save some historical data and print out the detailed information of training process in each epoch

        Args:
            pop: the current population
            epoch: current iteration
            runtime: the runtime for current iteration
        """
        ## Save history data
        if self.problem.save_population:
            self.history.list_population.append(Optimizer.duplicate_pop(pop))
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1].target.fitness)
        self.history.list_current_best_fit.append(self.history.list_current_best[-1].target.fitness)
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent.solution for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">>>Problem: {self.problem.name}, Epoch: {epoch}, Current best: {self.history.list_current_best[-1].target.fitness}, "
                         f"Global best: {self.history.list_global_best[-1].target.fitness}, Runtime: {runtime:.5f} seconds")

    def track_optimize_process(self) -> None:
        """
        Save some historical data after training process finished
        """
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.history.list_global_worst = self.history.list_global_worst[1:]
        self.history.list_current_worst = self.history.list_current_worst[1:]

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with solution

        Args:
            solution (np.ndarray): The solution
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return Agent(solution=solution)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with full information

        Args:
            solution (np.ndarray): The solution
        """
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        return agent

    def generate_population(self, pop_size: int = None) -> List[Agent]:
        """
        Args:
            pop_size (int): number of solutions

        Returns:
            list: population or list of solutions/agents
        """
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.generate_agent) for _ in range(pop_size)]
                # This method yield the result everytime a thread finished their job (not by order)
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.generate_agent) for _ in range(pop_size)]
                # This method yield the result everytime a cpu finished their job (not by order).
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        else:
            pop = [self.generate_agent() for _ in range(0, pop_size)]
        return pop

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        This function is based on optimizer's strategy.
        In each optimizer, this function can be overridden

        Args:
            solution: The position

        Returns:
            The valid solution based on optimizer's strategy
        """
        return solution

    def correct_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        This function is based on optimizer's strategy and problem-specific condition
        DO NOT override this function

        Args:
            solution: The position

        Returns:
            The correct solution that can be used to calculate target
        """
        solution = self.amend_solution(solution)
        return self.problem.correct_solution(solution)

    def update_target_for_population(self, pop: List[Agent] = None) -> List[Agent]:
        """
        Update target for the input population

        Args:
            pop: the population of agents

        Returns:
            list: population with updated target value
        """
        pos_list = [agent.solution for agent in pop]
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                # Return result as original order, not the future object
                list_results = executor.map(partial(self.get_target, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx].target = target
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                # Return result as original order, not the future object
                list_results = executor.map(partial(self.get_target, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx].target = target
        elif self.mode == "swarm":
            for idx, pos in enumerate(pos_list):
                pop[idx].target = self.get_target(pos, counted=False)
        else:
            return pop
        self.nfe_counter += len(pop)
        return pop

    def get_target(self, solution: np.ndarray, counted: bool = True) -> Target:
        """
        Get target value

        Args:
            solution: The real-value solution
            counted: Indicating the number of function evaluations is increasing or not

        Returns:
            The target value
        """
        if counted:
            self.nfe_counter += 1
        return self.problem.get_target(solution)

    @staticmethod
    def compare_target(target_x: Target, target_y: Target, minmax: str = "min") -> bool:
        if minmax == "min":
            return True if target_x.fitness < target_y.fitness else False
        else:
            return False if target_x.fitness < target_y.fitness else True

    @staticmethod
    def compare_fitness(fitness_x: Union[float, int], fitness_y: Union[float, int], minmax: str = "min") -> bool:
        if minmax == "min":
            return True if fitness_x < fitness_y else False
        else:
            return False if fitness_x < fitness_y else True

    @staticmethod
    def duplicate_pop(pop: List[Agent]) -> List[Agent]:
        return [agent.copy() for agent in pop]

    @staticmethod
    def get_sorted_population(pop: List[Agent], minmax: str = "min") -> List[Agent]:
        """
        Get sorted population based on type (minmax) of problem

        Args:
            pop: The population
            minmax: The type of the problem

        Returns:
            Sorted population (1st agent is the best, last agent is the worst
        """
        if minmax == "min":
            return sorted(pop, key=lambda agent: agent.target.fitness)
        else:
            return sorted(pop, key=lambda agent: agent.target.fitness, reverse=True)

    @staticmethod
    def get_best_agent(pop: List[Agent], minmax: str = "min") -> Agent:
        """
        Args:
            pop: The population of agents
            minmax: The type of problem

        Returns:
            The best agent
        """
        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[0].copy()

    @staticmethod
    def get_index_best(pop: List[Agent], minmax: str = "min") -> int:
        fit_list = np.array([agent.target.fitness for agent in pop])
        if minmax == "min":
            return np.argmin(fit_list)
        else:
            return np.argmax(fit_list)

    @staticmethod
    def get_worst_agent(pop: List[Agent], minmax: str = "min") -> Agent:
        """
        Args:
            pop: The population of agents
            minmax: The type of problem

        Returns:
            The worst agent
        """
        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[-1].copy()

    @staticmethod
    def get_special_agents(pop: List[Agent] = None, n_best: int = 3, n_worst: int = 3,
                           minmax: str = "min") -> Tuple[List[Agent], Union[List[Agent], None], Union[List[Agent], None]]:
        """
        Get special agents include sorted population, n1 best agents, n2 worst agents

        Args:
            pop: The population
            n_best: Top n1 best agents, default n1=3, good level reduction
            n_worst: Top n2 worst agents, default n2=3, worst level reduction
            minmax: The problem type

        Returns:
            The sorted_population, n1 best agents and n2 worst agents
        """
        pop = Optimizer.get_sorted_population(pop, minmax)
        if n_best is None:
            if n_worst is None:
                return pop, None, None
            else:
                return pop, None, [agent.copy() for agent in pop[::-1][:n_worst]]
        else:
            if n_worst is None:
                return pop, [agent.copy() for agent in pop[:n_best]], None
            else:
                return pop, [agent.copy() for agent in pop[:n_best]], [agent.copy() for agent in pop[::-1][:n_worst]]

    @staticmethod
    def get_special_fitness(pop: List[Agent] = None, minmax: str = "min") -> Tuple[Union[float, np.ndarray], float, float]:
        """
        Get special target include the total fitness, the best fitness, and the worst fitness

        Args:
            pop: The population
            minmax: The problem type

        Returns:
            The total fitness, the best fitness, and the worst fitness
        """
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        pop = Optimizer.get_sorted_population(pop, minmax)
        return total_fitness, pop[0].target.fitness, pop[-1].target.fitness

    @staticmethod
    def get_better_agent(agent_x: Agent, agent_y: Agent, minmax: str = "min", reverse: bool = False) -> Agent:
        """
        Args:
            agent_x: First agent
            agent_y: Second agent
            minmax: The problem type
            reverse: Reverse the minmax

        Returns:
            The better agent based on fitness
        """
        minmax_dict = {"min": 0, "max": 1}
        idx = minmax_dict[minmax]
        if reverse:
            idx = 1 - idx
        if idx == 0:
            return agent_x.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_y.copy()
        else:
            return agent_y.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_x.copy()

    ### Survivor Selection
    @staticmethod
    def greedy_selection_population(pop_old: List[Agent] = None, pop_new: List[Agent] = None, minmax: str = "min") -> List[Agent]:
        """
        Args:
            pop_old: The current population
            pop_new: The next population
            minmax: The problem type

        Returns:
            The new population with better solutions
        """
        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            raise ValueError("Greedy selection of two population with different length.")
        if minmax == "min":
            return [pop_new[idx] if pop_new[idx].target.fitness < pop_old[idx].target.fitness else pop_old[idx] for idx in range(len_old)]
        else:
            return [pop_new[idx] if pop_new[idx].target.fitness > pop_old[idx].target.fitness else pop_old[idx] for idx in range(len_old)]

    @staticmethod
    def get_sorted_and_trimmed_population(pop: List[Agent] = None, pop_size: int = None, minmax: str = "min") -> List[Agent]:
        """
        Args:
            pop: The population
            pop_size: The number of selected agents
            minmax: The problem type

        Returns:
            The sorted and trimmed population with pop_size size
        """
        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[:pop_size]

    def update_global_best_agent(self, pop: List[Agent], save: bool = True) -> Union[List, Tuple]:
        """
        Update global best and current best solutions in history object.
        Also update global worst and current worst solutions in history object.

        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current/global best to history, False if you just want to update current/global best

        Returns:
            list: Sorted population and the global best solution
        """
        sorted_pop = self.get_sorted_population(pop, self.problem.minmax)
        c_best, c_worst = sorted_pop[0], sorted_pop[-1]
        if save:
            ## Save current best
            self.history.list_current_best.append(c_best)
            better = self.get_better_agent(c_best, self.history.list_global_best[-1], self.problem.minmax)
            self.history.list_global_best.append(better)
            ## Save current worst
            self.history.list_current_worst.append(c_worst)
            worse = self.get_better_agent(c_worst, self.history.list_global_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_global_worst.append(worse)
            return sorted_pop, better
        else:
            ## Handle current best
            local_better = self.get_better_agent(c_best, self.history.list_current_best[-1], self.problem.minmax)
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_agent(c_best, self.history.list_global_best[-1], self.problem.minmax)
            self.history.list_global_best[-1] = global_better
            ## Handle current worst
            local_worst = self.get_better_agent(c_worst, self.history.list_current_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_current_worst[-1] = local_worst
            global_worst = self.get_better_agent(c_worst, self.history.list_global_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_global_worst[-1] = global_worst
            return sorted_pop, global_better

    ## Selection techniques
    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        """
        This method can handle min/max problem, and negative or positive fitness value.

        Args:
            list_fitness (nd.array): 1-D numpy array

        Returns:
            int: Index of selected solution
        """
        if type(list_fitness) in [list, tuple, np.ndarray]:
            list_fitness = np.array(list_fitness).flatten()
        if list_fitness.ptp() == 0:
            return int(self.generator.integers(0, len(list_fitness)))
        if np.any(list_fitness) < 0:
            list_fitness = list_fitness - np.min(list_fitness)
        final_fitness = list_fitness
        if self.problem.minmax == "min":
            final_fitness = np.max(list_fitness) - list_fitness
        prob = final_fitness / np.sum(final_fitness)
        return int(self.generator.choice(range(0, len(list_fitness)), p=prob))

    def get_index_kway_tournament_selection(self, pop: List = None, k_way: float = 0.2, output: int = 2, reverse: bool = False) -> List:
        """
        Args:
            pop: The population
            k_way (float/int): The percent or number of solutions are randomized pick
            output (int): The number of outputs
            reverse (bool): set True when finding the worst fitness

        Returns:
            list: List of the selected indexes
        """
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = self.generator.choice(range(len(pop)), k_way, replace=False)
        list_parents = [[idx, pop[idx].target.fitness] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def get_levy_flight_step(self, beta: float = 1.0, multiplier: float = 0.001, 
                             size: Union[List, Tuple, np.ndarray] = None, case: int = 0) -> Union[float, List, np.ndarray]:
        """
        Get the Levy-flight step size

        Args:
            beta (float): Should be in range [0, 2].

                * 0-1: small range --> exploit
                * 1-2: large range --> explore

            multiplier (float): default = 0.001
            size (tuple, list): size of levy-flight steps, for example: (3, 2), 5, (4, )
            case (int): Should be one of these value [0, 1, -1].

                * 0: return multiplier * s * self.generator.uniform()
                * 1: return multiplier * s * self.generator.normal(0, 1)
                * -1: return multiplier * s

        Returns:
            float, list, np.ndarray: The step size of Levy-flight trajectory
        """
        # u and v are two random variables which follow self.generator.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(gamma(1. + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2.) * beta * np.power(2., (beta - 1) / 2)), 1. / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        size = 1 if size is None else size
        u = self.generator.normal(0, sigma_u ** 2, size)
        v = self.generator.normal(0, sigma_v ** 2, size)
        s = u / np.power(np.abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * self.generator.uniform()
        elif case == 1:
            step = multiplier * s * self.generator.normal(0, 1)
        else:
            step = multiplier * s
        return step[0] if size == 1 else step

    def generate_opposition_solution(self, agent: Agent = None, g_best: Agent = None) -> np.ndarray:
        """
        Args:
            agent: The current agent
            g_best: the global best agent

        Returns:
            The opposite solution
        """
        pos_new = self.problem.lb + self.problem.ub - g_best.solution + self.generator.uniform() * (g_best.solution - agent.solution)
        return self.correct_solution(pos_new)

    def generate_group_population(self, pop: List[Agent], n_groups: int, m_agents: int) -> List:
        """
        Generate a list of group population from pop

        Args:
            pop: The current population
            n_groups: The n groups
            m_agents: The m agents in each group

        Returns:
            A list of group population
        """
        pop_group = []
        for idx in range(0, n_groups):
            group = pop[idx * m_agents: (idx + 1) * m_agents]
            pop_group.append([agent.copy() for agent in group])
        return pop_group

    ### Crossover
    def crossover_arithmetic(self, dad_pos=None, mom_pos=None):
        """
        Args:
            dad_pos: position of dad
            mom_pos: position of mom

        Returns:
            list: position of 1st and 2nd child
        """
        r = self.generator.uniform()  # w1 = w2 when r =0.5
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
    def improved_ms(self, pop=None, g_best=None):  ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda agent: agent.target.fitness)
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]

        ## Mutation scheme
        pop_new = []
        for idx in range(0, pop_len):
            agent = pop_s1[idx].copy()
            pos_new = pop_s1[idx].solution * (1 + self.generator.normal(0, 1, self.problem.n_dims))
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)
        pop_new = self.update_target_for_population(pop_new)
        pop_s1 = self.greedy_selection_population(pop_s1, pop_new, self.problem.minmax)  ## Greedy method --> improved exploitation

        ## Search Mechanism
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []
        for idx in range(0, pop_len):
            agent = pop_s2[idx].copy()
            pos_new = (g_best.solution - pos_s1_mean) - self.generator.random() * \
                      (self.problem.lb + self.generator.random() * (self.problem.ub - self.problem.lb))
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)
        ## Keep the diversity of populatoin and still improved the exploration
        pop_s2 = self.update_target_for_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new, self.problem.minmax)

        ## Construct a new population
        pop = pop_s1 + pop_s2
        return pop
