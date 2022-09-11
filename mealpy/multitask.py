#!/usr/bin/env python
# Created by "Thieu" at 22:21, 06/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from pathlib import Path
from mealpy.optimizer import Optimizer
from mealpy.utils.problem import Problem
from mealpy.utils.validator import Validator
from functools import partial
import concurrent.futures as parallel
import os


class Multitask:
    r"""Multitask utility feature.

    Feature which enables running multiple algorithms with multiple problems, and multiple trials.
    It also supports exporting results in various formats (e.g. Pandas DataFrame, JSON, CSV)

    Attributes:
        algorithms (list, tuple): List of algorithms to run
        problems (list, tuple): List of problems to run

    """
    def __init__(self, algorithms=(), problems=(), **kwargs):
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)
        self.algorithms = self.validator.check_list_tuple("algorithms", algorithms, "Optimizer")
        self.problems = self.validator.check_list_tuple("problems", problems, "Problem")

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

    def __run__(self, id_trial, model, problem):
        _, best_fitness = model.solve(problem)
        return {
            "id_trial": id_trial,
            "best_fitness": best_fitness,
            "convergence": model.history.list_global_best_fit
        }

    def execute(self, n_trials=2, mode="sequential", n_workers=2, save_path="history", save_as="csv", save_convergence=False, verbose=False):
        """Execute multitask utility.

        Args:
            n_trials (int): Number of repetitions
            mode (str): Execute problem using "sequential" or "parallel" mode, default = "sequential"
            n_workers (int): Number of processes if mode is "parallel"
            save_path (str): The path to the folder that hold results
            save_as (str): Saved file type (e.g. dataframe, json, csv) (default: "csv")
            save_convergence (bool): Save the error (convergence/fitness) during generations (default: False)
            verbose (bool): Switch for verbose logging (default: False)

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        n_trials = self.validator.check_int("n_trials", n_trials, [1, 100000])
        mode = self.validator.check_str("mode", mode, ["parallel", "sequential"])
        if mode == "process":
            n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
        else:
            n_workers = None
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

                convergence_trials = {}
                best_fit_trials = []

                trial_list = list(range(1, n_trials+1))

                if mode == "parallel":
                    with parallel.ProcessPoolExecutor(n_workers) as executor:
                        list_results = executor.map(partial(self.__run__, model=model, problem=problem), trial_list)
                        for result in list_results:
                            convergence_trials[f"trial_{result['id_trial']}"] = result['convergence']
                            best_fit_trials.append(result['best_fitness'])
                            if verbose:
                                print(f"Solving problem: {problem.get_name()} using algorithm: {model.get_name()}, on the: {result['id_trial']} trial")
                else:
                    for idx in trial_list:
                        result = self.__run__(idx, model, problem)
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
