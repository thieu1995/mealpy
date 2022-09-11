#!/usr/bin/env python
# Created by "Thieu" at 10:49, 11/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from pathlib import Path
from mealpy.optimizer import Optimizer
from mealpy.utils.problem import Problem
from mealpy.utils.validator import Validator
from collections import abc
from functools import partial, reduce
from itertools import product
import concurrent.futures as parallel
import operator
import os


class ParameterGrid:
    """
    Check out this class from scikit-learn library.

    Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is useful to avoid exploring
        parameter combinations that make no sense or have no effect. See the examples below.

    Examples
    --------
    >>> from mealpy.tuner import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == ([{'a': 1, 'b': True}, {'a': 1, 'b': False}, {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'}, {'kernel': 'rbf', 'gamma': 1}, {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (abc.Mapping, abc.Iterable)):
            raise TypeError(f"Parameter grid should be a dict or a list, got: {param_grid!r} of type {type(param_grid).__name__}")

        if isinstance(param_grid, abc.Mapping):
            # wrap dictionary in a singleton list to support either dict or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f"Parameter array for {key!r} should be one-dimensional, got: {value!r} with shape {value.shape}")
                if isinstance(value, str) or not isinstance(value, (np.ndarray, abc.Sequence)):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type "
                        f"{type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element.")
                if len(value) == 0:
                    raise ValueError(f"Parameter grid for parameter {key!r} need to be a non-empty sequence, got: {value!r}")
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1 for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError("ParameterGrid index out of range")


class Tuner:
    r"""Tuner utility feature. (it is better than using GridSearchCV / ParameterGrid in scikit-learn library)

    Feature which enables tuning the hyper-parameter of an algorithm.
    It also supports exporting results in various formats (e.g. Pandas DataFrame, JSON, CSV)

    Important functions are execute() and

    Attributes:
        algorithm (Optimizer): the algorithm/optimizer to tune
        param_grid (dict, list): dict or list of dictionaries
        n_trials (int): number of repetitions
        mode (str): set the mode to run (sequential, thread, process), default="sequential"
        n_workers (int): effected only when mode is "thread" or "process".

    """
    def __init__(self, algorithm=None, param_grid=None, **kwargs):
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)
        self.algorithm = self.validator.check_is_instance("algorithm", algorithm, Optimizer)
        self.param_grid = self.validator.check_is_instance("param_grid", param_grid, dict)
        self.results, self._best_row, self._best_params, self._best_score, self._best_algorithm = None, None, None, None, None

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def best_params(self):
        return self._best_params

    @best_params.setter
    def best_params(self, x):
        self._best_params = x

    @property
    def best_row(self):
        return self._best_row

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_algorithm(self):
        self.algorithm.set_parameters(self._best_params)
        return self.algorithm

    def export_results(self, save_path="history/results", save_as="csv"):
        """Export results to various file type

        Args:
            save_path (str): The path to the folder with the file name that hold results
            save_as (str): Saved file type (e.g. dataframe, json, csv) (default: "csv")

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        ## Check parent directories
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_as = self.validator.check_str("save_as", save_as, ["csv", "json", "dataframe"])
        if save_as == "json":
            self.results.to_json(f"{save_path}.json")
        elif save_as == "dataframe":
            self.results.to_pickle(f"{save_path}.pkl")
        else:
            self.results.to_csv(f"{save_path}.csv", header=True, index=False)

    def __run__(self, id_trial):
        _, best_fitness = self.algorithm.solve(self.problem)
        return id_trial, best_fitness

    def execute(self, problem=None, n_trials=2, mode="sequential", n_workers=2, verbose=True):
        """Execute Tuner utility.

        Args:
            problem (dict, Problem): A instance of Problem class or problem dictionary
            n_trials (int): Number of trials on the Problem
            mode (str): Execute problem using "sequential" or "parallel" mode, default = "sequential"
            n_workers (int): Number of processes if mode is "parallel"
            verbose (bool): Switch for verbose logging (default: False)

        Raises:
            TypeError: Raises TypeError if problem type is not dictionary or an instance Problem class

        """
        if not isinstance(problem, Problem):
            if type(problem) is dict:
                self.problem = Problem(**problem)
            else:
                raise TypeError(f"Problem is not an instance of Problem class or a Python dict.")
        n_trials = self.validator.check_int("n_trials", n_trials, [1, 100000])
        mode = self.validator.check_str("mode", mode, ["parallel", "sequential"])
        if mode == "process":
            n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
        else:
            n_workers = None
        list_params_grid = list(ParameterGrid(self.param_grid))
        trial_columns = [f"trial_{id_trial}" for id_trial in range(1, n_trials + 1)]
        ascending = True if self.problem.minmax == "min" else False

        best_fit_results = []
        for id_params, params in enumerate(list_params_grid):

            self.algorithm.set_parameters(params)
            best_fit_results.append({"params": params})

            trial_list = list(range(0, n_trials))
            if mode == "parallel":
                with parallel.ProcessPoolExecutor(n_workers) as executor:
                    list_results = executor.map(self.__run__, trial_list)  # Return results as original order not the future object
                    for (idx, best_fitness) in list_results:
                        best_fit_results[-1][trial_columns[idx]] = best_fitness
                        if verbose:
                            print(f"Algorithm: {self.algorithm.get_name()}, with params: {params}, trial: {idx + 1}, best fitness: {best_fitness}")
            else:
                for idx in trial_list:
                    idx, best_fitness = self.__run__(idx)
                    best_fit_results[-1][trial_columns[idx]] = best_fitness
                    if verbose:
                        print(f"Algorithm: {self.algorithm.get_name()}, with params: {params}, trial: {idx+1}, best fitness: {best_fitness}")

        df = pd.DataFrame(best_fit_results)
        df["trial_mean"] = df[trial_columns].mean(axis=1)
        df["trial_std"] = df[trial_columns].std(axis=1)
        df["rank_mean"] = df["trial_mean"].rank(ascending=ascending)
        df["rank_std"] = df["trial_std"].rank(ascending=ascending)
        df["rank_mean_std"] = df[["rank_mean", "rank_std"]].apply(tuple, axis=1).rank(method='dense', ascending=ascending)

        self._best_row = df[df["rank_mean_std"] == df["rank_mean_std"].min()]
        self._best_params = self._best_row["params"].values[0]
        self._best_score = self._best_row["trial_mean"].values[0]
        self.results = df

    def resolve(self, mode='single', starting_positions=None, n_workers=None, termination=None):
        """
        Resolving the problem with the best parameters

        Args:
            mode (str): Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, default

            starting_positions(list, np.ndarray): List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            n_workers (int): The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination (dict, None): The termination dictionary or an instance of Termination class

        Returns:
            list: [position, fitness value]
        """
        self.algorithm.set_parameters(self.best_params)
        return self.algorithm.solve(problem=self.problem, mode=mode, n_workers=n_workers,
                                    starting_positions=starting_positions, termination=termination)
