#!/usr/bin/env python
# Created by "Thieu" at 22:21, 06/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from pathlib import Path
from mealpy.optimizer import Optimizer
from mealpy.utils.problem import Problem


class Multitask:
    r"""Multitask utility feature.

    Feature which enables running multiple algorithms with multiple problems.
    It also supports exporting results in various formats (e.g. Pandas DataFrame, JSON, CSV)

    Attributes:
        dimension (int): Dimension of problem
        max_evals (int): Number of function evaluations
        runs (int): Number of repetitions
        algorithms (Union[List[str], List[Algorithm]]): List of algorithms to run
        problems (List[Union[str, Problem]]): List of problems to run

    """

    def __init__(self, algorithms=(), problems=(), n_trials=2):
        self.algorithms = algorithms
        self.problems = problems
        self.n_trials = n_trials

    def export_to_dataframe(self, result: pd.DataFrame, save_path: str):
        result.to_pickle(f"{save_path}.pkl")

    def export_to_json(self, result: pd.DataFrame, save_path: str):
        result.to_json(f"{save_path}.json")

    def export_to_csv(self, result: pd.DataFrame, save_path: str):
        result.to_csv(f"{save_path}.csv", header=True, index=False)

    def get_export_function(self, export_type="csv"):
        func = getattr(self, f"export_to_{export_type}")
        if func is None:
            raise TypeError(f"Passed export type: {export_type} is not supported!")
        return func

    def execute(self, save_path="", save_as="csv", save_convergence=False, verbose=False):
        """Execute multitask utility.

        Args:

            save_path (str): The path to the folder that hold results
            save_as (str): Saved file type (e.g. dataframe, json, csv) (default: "csv")
            save_convergence (bool): Save the error (convergence/fitness) during generations (default: False)
            verbose (bool): Switch for verbose logging (default: False)

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        ## Get export function
        export_function = self.get_export_function(save_as)


        for model in self.algorithms:
            if not isinstance(model, Optimizer):
                print(f"{model} is not an instance of Optimizer class.")
                continue

            ## Check parent directories
            path_best_fit = f"{save_path}/best_fit"
            path_convergence = f"{save_path}/convergence/{model.get_name()}"
            Path(path_best_fit).mkdir(parents=True, exist_ok=True)
            Path(path_convergence).mkdir(parents=True, exist_ok=True)

            best_fit_model_results = {}
            for problem in self.problems:
                if not isinstance(problem, Problem):
                    if not type(problem) is dict:
                        print(f"{problem} is not an instance of Problem class or a Python dict.")
                        continue
                    else:
                        problem = Problem(**problem)

                convergence_trials = {}
                best_fit_trials = []
                for id_trial in range(1, self.n_trials + 1):

                    if verbose:
                        print(f"Solving problem: {problem.get_name()} using algorithm: {model.get_name()}, on the: {id_trial} trial")

                    _, best_fitness = model.solve(problem)

                    convergence_trials[f"trial_{id_trial}"] = model.history.list_global_best_fit
                    best_fit_trials.append(best_fitness)

                best_fit_model_results[problem.get_name()] = best_fit_trials
                if save_convergence:
                    df1 = pd.DataFrame(convergence_trials)
                    export_function(df1, f"{path_convergence}/{problem.get_name()}_convergence")

            df2 = pd.DataFrame(best_fit_model_results)
            export_function(df2, f"{path_best_fit}/{model.get_name()}_best_fit")
