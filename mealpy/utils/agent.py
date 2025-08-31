#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Any
import numpy as np
from mealpy.utils.target import Target


class Agent:
    ID = 0

    def __init__(self, solution: np.ndarray = None, target: Target = None, **kwargs) -> None:
        self.solution = solution
        self.target = target
        self.set_kwargs(kwargs)
        self.kwargs = kwargs
        self.id = self.increase()

    @classmethod
    def increase(cls) -> int:
        cls.ID += 1
        return cls.ID

    def __getattr__(self, name: str) -> Any:
        # return None or raise AttributeError
        return self.__dict__.get(name, None)

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self) -> 'Agent':
        agent = Agent(self.solution, self.target.copy(), **self.kwargs)
        # Copy any changes made to the attributes
        for attr, value in vars(self).items():
            if attr not in ['target', 'solution', 'id', 'kwargs']:
                setattr(agent, attr, value)
        return agent

    def update_agent(self, solution: np.ndarray, target: Target) -> None:
        self.solution = solution
        self.target = target

    def update(self, **kwargs) -> None:
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def sync_if_duplicate(self, other: "Agent") -> bool:
        """
        Check if two agents are equal (using __eq__), and if so, synchronize the target from the other agent.

        Returns:
            bool: True if duplicate (and target updated), False otherwise.
        """
        if self == other:  # use __eq__
            self.target = other.target
            return True
        return False

    def _compare_fitness(self, other: "Agent", minmax: str = "min") -> int:
        """
        Compare fitness between self and other.

        Returns:
            -1 if self is better
             0 if equal
             1 if other is better
        """
        if self.target.fitness == other.target.fitness:
            return 0
        if minmax == "min":
            return -1 if self.target.fitness < other.target.fitness else 1
        else:
            return -1 if self.target.fitness > other.target.fitness else 1

    def get_better_solution(self, other: "Agent", minmax: str = "min") -> "Agent":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "Agent", minmax: str = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self):     # represent
        return f"id: {self.id}, target: {self.target}, solution: {self.solution}"

    def __eq__(self, other):
        """ Check if two agents are equal based on their solutions with a tolerance."""
        if not isinstance(other, Agent):
            return False
        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        """ Generate a hash based on the solution of the agent.
            This is useful for using agents in sets or as dictionary keys."""
        return hash(tuple(np.round(self.solution, 6)))
