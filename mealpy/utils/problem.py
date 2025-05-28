#!/usr/bin/env python
# Created by "Thieu" at 17:28, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numbers
import numpy as np
from typing import Union, List, Tuple, Dict
from mealpy.utils.space import (BaseVar, IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar,
                                PermutationVar, CategoricalVar, SequenceVar, TransferBinaryVar, TransferBoolVar)
from mealpy.utils.logger import Logger
from mealpy.utils.target import Target


class Problem:
    SUPPORTED_VARS = (IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar,
                      PermutationVar, CategoricalVar, SequenceVar, TransferBinaryVar, TransferBoolVar)
    SUPPORTED_ARRAYS = (list, tuple, np.ndarray)

    def __init__(self, bounds: Union[List, Tuple, np.ndarray, BaseVar], minmax: str = "min", **kwargs) -> None:
        self._bounds, self.lb, self.ub = None, None, None
        self.minmax = minmax
        self.seed = None
        self._n_objs, self.obj_weights = None, None
        self.n_dims, self.save_population = None, False
        self.name, self.log_to, self.log_file = "P", "console", "history.txt"
        self.__set_keyword_arguments(kwargs)
        self.set_bounds(bounds)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
                                    format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')

    @property
    def bounds(self):
        return self._bounds

    @property
    def n_objs(self):
        if self._n_objs is None:
            x = self.generate_solution(encoded=True)
            result = self.obj_func(x)
            if isinstance(result, self.SUPPORTED_ARRAYS):
                self._n_objs = len(np.asarray(result).ravel())
            elif isinstance(result, numbers.Number):
                self._n_objs = 1
            else:
                raise ValueError("`obj_func` must return a number, list, tuple or numpy array.")
            if self.obj_weights is None:
                if self._n_objs > 1:
                    self.logger.warning(
                        f"[Warning] Multi-objective problem detected (n_objs={self._n_objs}), "
                        f"but `obj_weights` not provided. Defaulting to equal weights."
                    )
                self.obj_weights = np.ones(self._n_objs)
            elif len(np.array(self.obj_weights).ravel()) != self._n_objs:
                raise ValueError(f"`obj_weights` length {len(self.obj_weights)} does not match number of objectives {self._n_objs}.")
        return self._n_objs

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_bounds(self, bounds):
        if isinstance(bounds, BaseVar):
            bounds.seed = self.seed
            self._bounds = [bounds, ]
        elif type(bounds) in self.SUPPORTED_ARRAYS:
            self._bounds = []
            for bound in bounds:
                if isinstance(bound, BaseVar):
                    bound.seed = self.seed
                else:
                    raise ValueError(f"Invalid bounds. All variables in bounds should be an instance of {self.SUPPORTED_VARS}")
                self._bounds.append(bound)
        else:
            raise TypeError(f"Invalid bounds. It should be type of {self.SUPPORTED_ARRAYS} or an instance of {self.SUPPORTED_VARS}")
        self.lb = np.concatenate([bound.lb for bound in self._bounds])
        self.ub = np.concatenate([bound.ub for bound in self._bounds])
        self.n_dims = len(self.lb)

    def set_seed(self, seed: int = None) -> None:
        self.seed = seed
        for idx in range(len(self._bounds)):
            self._bounds[idx].seed = seed

    def obj_func(self, x: np.ndarray) -> Union[List, Tuple, np.ndarray, int, float]:
        """Objective function

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """
        Returns:
            string: The name of the problem
        """
        return self.name

    def get_class_name(self) -> str:
        """Get class name."""
        return self.__class__.__name__

    @staticmethod
    def encode_solution_with_bounds(x, bounds):
        x_new = []
        for idx, var in enumerate(bounds):
            x_new += list(var.encode(x[idx]))
        return np.array(x_new)

    @staticmethod
    def decode_solution_with_bounds(x, bounds):
        x_new, n_vars = {}, 0
        for idx, var in enumerate(bounds):
            temp = var.decode(x[n_vars:n_vars + var.n_vars])
            if var.n_vars == 1:
                x_new[var.name] = temp[0]
            else:
                x_new[var.name] = temp
            n_vars += var.n_vars
        return x_new

    @staticmethod
    def correct_solution_with_bounds(x: Union[List, Tuple, np.ndarray], bounds: List) -> np.ndarray:
        x_new, n_vars = [], 0
        for idx, var in enumerate(bounds):
            x_new += list(var.correct(x[n_vars:n_vars+var.n_vars]))
            n_vars += var.n_vars
        return np.array(x_new)

    @staticmethod
    def generate_solution_with_bounds(bounds: Union[List, Tuple, np.ndarray], encoded: bool = True) -> Union[List, np.ndarray]:
        x = [var.generate() for var in bounds]
        if encoded:
            return Problem.encode_solution_with_bounds(x, bounds)
        return x

    def encode_solution(self, x: Union[List, tuple, np.ndarray]) -> np.ndarray:
        """
        Encode the real-world solution to optimized solution (real-value solution)

        Args:
            x (Union[List, tuple, np.ndarray]): The real-world solution

        Returns:
            The real-value solution
        """
        return self.encode_solution_with_bounds(x, self.bounds)

    def decode_solution(self, x: np.ndarray) -> Dict:
        """
        Decode the encoded solution to real-world solution

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The real-world (decoded) solution
        """
        return self.decode_solution_with_bounds(x, self.bounds)

    def correct_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Correct the solution to valid bounds

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The corrected solution
        """
        return self.correct_solution_with_bounds(x, self.bounds)

    def generate_solution(self, encoded: bool = True) -> Union[List, np.ndarray]:
        """
        Generate the solution.

        Args:
            encoded (bool): Encode the solution or not

        Returns:
            the encoded/non-encoded solution for the problem
        """
        return self.generate_solution_with_bounds(self.bounds, encoded)

    def get_target(self, solution: np.ndarray) -> Target:
        """
        Args:
            solution: The real-value solution

        Returns:
            The target object
        """
        objs = self.obj_func(solution)
        return Target(objectives=objs, weights=self.obj_weights)
