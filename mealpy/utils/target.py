#!/usr/bin/env python
# Created by "Thieu" at 15:51, 05/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Union, List, Tuple
import numbers
import numpy as np


class Target:
    SUPPORTED_ARRAY = [tuple, list, np.ndarray]

    def __init__(self, objectives: Union[List, Tuple, np.ndarray, int, float] = None,
                 weights: Union[List, Tuple, np.ndarray] = None) -> None:
        """
        Initialize the Target with a list of objectives and a fitness value.

        Parameters:
            objectives: The list of objective values.
            weights: The weights for calculating fitness value
        """
        self._objectives, self._weights, self._fitness = None, None, None
        self.set_objectives(objectives)
        self.set_weights(weights)
        self.calculate_fitness(self.weights)

    def copy(self) -> 'Target':
        return Target(self.objectives, self.weights)

    @property
    def objectives(self):
        """Returns the list of objective values."""
        return self._objectives

    def set_objectives(self, objs):
        if objs is None:
            raise ValueError(f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float.")
        else:
            if type(objs) not in self.SUPPORTED_ARRAY:
                if isinstance(objs, numbers.Number):
                    objs = [objs]
                else:
                    raise ValueError(f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float.")
            objs = np.array(objs).flatten()
        self._objectives = objs

    @property
    def weights(self):
        """Returns the list of weight values."""
        return self._weights

    def set_weights(self, weights):
        if weights is None:
            self._weights = len(self.objectives)
        else:
            if type(weights) not in self.SUPPORTED_ARRAY:
                if isinstance(weights, numbers.Number):
                    weights = [weights, ] * len(self.objectives)
                else:
                    raise ValueError(f"Invalid weights. It should be a list, tuple, np.ndarray.")
            weights = np.array(weights).flatten()
        self._weights = weights

    @property
    def fitness(self):
        """Returns the fitness value."""
        return self._fitness

    def calculate_fitness(self, weights: Union[List, Tuple, np.ndarray]) -> None:
        """
        Calculates the fitness value of the solution based on the provided weights.

        Parameters:
            weights (list): The weights for the objectives.

        Returns:
            float: The fitness value of the solution.
        """
        # Calculate the weighted sum of the objectives
        if not (type(weights) in self.SUPPORTED_ARRAY and len(weights) == len(self.objectives)):
            weights = len(self.objectives) * (1.,)
        self._fitness = np.dot(weights, self.objectives)

    def __str__(self):
        return f"Objectives: {self.objectives}, Fitness: {self.fitness}"
