#!/usr/bin/env python
# Created by "Thieu" at 22:21, 06/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from pathlib import Path
from mealpy.optimizer import Optimizer
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination
from mealpy.utils.validator import Validator
from functools import partial
import concurrent.futures as parallel
from copy import deepcopy
import os


class Multitask:
    """Multitask utility class.

    This feature enables the execution of multiple algorithms across multiple problems and trials.
    Additionally, it allows for exporting results in various formats such as Pandas DataFrame, JSON, and CSV.

    Args:
        algorithms (list, tuple): List of algorithms to run
        problems (list, tuple): List of problems to run
        terminations (list, tuple): List of terminations to apply on algorithm/problem
        modes (list, tuple): List of modes to apply on algorithm/problem
    """
    def __init__(self, algorithms=(), problems=(), terminations=None, modes=None, **kwargs):
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)
        self.algorithms = self.validator.check_list_tuple("algorithms", algorithms, "Optimizer")
        self.problems = self.validator.check_list_tuple("problems", problems, "Problem")
        self.n_algorithms = len(self.algorithms)
        self.m_problems = len(self.problems)
        self.terminations = self.check_input("terminations", terminations, "Termination")
        self.modes = self.check_input("modes", modes, "str (thread, process, single, swarm)")

    def check_input(self, name=None, values=None, kind=None):
        if values is None:
            return None
        elif type(values) in (list, tuple):
            if len(values) == 1:
                values_final = [[deepcopy(values[0]) for _ in range(0, self.m_problems)] for _ in range(0, self.n_algorithms)]
            elif len(values) == self.n_algorithms:
                values_final = [deepcopy(values[idx] for _ in range(0, self.m_problems)) for idx in range(0, self.n_algorithms)]
            elif len(values) == self.m_problems:
                values_final = [deepcopy(values) for _ in range(0, self.n_algorithms)]
            elif len(values) == (self.n_algorithms * self.m_problems):
                values_final = values
            else:
                raise ValueError(f"{name} should be list of {kind} instances with size (1) or (n) or (m) or (n*m), n: #algorithms, m: #problems.")
            return values_final
        else:
            raise ValueError(f"{name} should be list of {kind} instances.")

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def export_to_dataframe(result: pd.DataFrame, save_path: str):
        result.to_pickle(f"{save_path}.pkl")

    @staticmethod
    def export_to_json(result: pd.DataFrame, save_path: str):
        result.to_json(f"{save_path}.json")

    @staticmethod
    def export_to_csv(result: pd.DataFrame, save_path: str):
        result.to_csv(f"{save_path}.csv", header=True, index=False)

    def __run__(self, id_trial, model, problem, termination=None, mode="single"):
        _, best_fitness = model.solve(problem, mode=mode, termination=termination)
        return {
            "id_trial": id_trial,
            "best_fitness": best_fitness,
            "convergence": model.history.list_global_best_fit
        }

    def execute(self, n_trials=2, n_jobs=None, save_path="history", save_as="csv", save_convergence=False, verbose=False):
        """Execute multitask utility.

        Args:
            n_trials (int): Number of repetitions
            n_jobs (int, None): Number of processes will be used to speed up the computation (<=1 or None: sequential, >=2: parallel)
            save_path (str): The path to the folder that hold results
            save_as (str): Saved file type (e.g. dataframe, json, csv) (default: "csv")
            save_convergence (bool): Save the error (convergence/fitness) during generations (default: False)
            verbose (bool): Switch for verbose logging (default: False)

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        n_trials = self.validator.check_int("n_trials", n_trials, [1, 100000])
        n_workers = None
        if (n_jobs is not None) and (n_jobs >= 1):
            n_workers = self.validator.check_int("n_jobs", n_jobs, [2, min(61, os.cpu_count() - 1)])

        ## Get export function
        save_as = self.validator.check_str("save_as", save_as, ["csv", "json", "dataframe"])
        export_function = getattr(self, f"export_to_{save_as}")

        for id_model, model in enumerate(self.algorithms):
            if not isinstance(model, Optimizer):
                print(f"Model: {id_model+1} is not an instance of Optimizer class.")
                continue

            ## Check parent directories
            path_best_fit = f"{save_path}/best_fit"
            path_convergence = f"{save_path}/convergence/{model.get_name()}"
            Path(path_best_fit).mkdir(parents=True, exist_ok=True)
            Path(path_convergence).mkdir(parents=True, exist_ok=True)

            best_fit_model_results = {}
            for id_prob, problem in enumerate(self.problems):
                if not isinstance(problem, Problem):
                    if not type(problem) is dict:
                        print(f"Problem: {id_prob+1} is not an instance of Problem class or a Python dict.")
                        continue
                    else:
                        problem = Problem(**problem)

                term = None
                if self.terminations is not None:
                    term = self.terminations[id_model][id_prob]
                    if not isinstance(term, Termination):
                        if not type(term) is dict:
                            print(f"Termination: {id_prob + 1} is not an instance of Termination class or a Python dict.")
                            continue
                        else:
                            term = Termination(**term)

                mode = "single"
                if self.modes is not None:
                    mode = self.modes[id_model][id_prob]
                    if mode not in ("process", "thread", "single", "swarm"):
                        mode = "single"
                        print(f"Mode: {id_prob + 1} is fall back on 'single'")

                convergence_trials = {}
                best_fit_trials = []

                trial_list = list(range(1, n_trials+1))

                if n_workers is not None:
                    with parallel.ProcessPoolExecutor(n_workers) as executor:
                        list_results = executor.map(partial(self.__run__, model=model, problem=problem, termination=term, mode=mode), trial_list)
                        for result in list_results:
                            convergence_trials[f"trial_{result['id_trial']}"] = result['convergence']
                            best_fit_trials.append(result['best_fitness'])
                            if verbose:
                                print(f"Solving problem: {problem.get_name()} using algorithm: {model.get_name()}, on the: {result['id_trial']} trial")
                else:
                    for idx in trial_list:
                        result = self.__run__(idx, model, problem, termination=term, mode=mode)
                        convergence_trials[f"trial_{result['id_trial']}"] = result['convergence']
                        best_fit_trials.append(result['best_fitness'])
                        if verbose:
                            print(f"Solving problem: {problem.get_name()} using algorithm: {model.get_name()}, on the: {result['id_trial']} trial")

                best_fit_model_results[problem.get_name()] = best_fit_trials
                if save_convergence:
                    df1 = pd.DataFrame(convergence_trials)
                    export_function(df1, f"{path_convergence}/{problem.get_name()}_convergence")

            df2 = pd.DataFrame(best_fit_model_results)
            export_function(df2, f"{path_best_fit}/{model.get_name()}_best_fit")
