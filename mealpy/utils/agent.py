#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from typing import Union


class Agent:
    ID = 0

    def __init__(self, solution: np.ndarray = None, fitness: Union[float, int, np.ndarray] = None, **kwargs) -> None:
        self.solution = solution
        self.fitness = fitness
        self.set_kwargs(kwargs)
        self.kwargs = kwargs
        self.id = self.increase()

    @classmethod
    def increase(cls) -> int:
        cls.ID += 1
        return cls.ID

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self) -> 'Agent':
        agent = Agent(self.solution, self.fitness, **self.kwargs)
        # Copy any changes made to the attributes
        for attr, value in vars(self).items():
            if attr not in ['fitness', 'solution', 'id', 'kwargs']:
                setattr(agent, attr, value)
        return agent

    def update_agent(self, solution: np.ndarray, fitness: object) -> None:
        self.solution = solution
        self.fitness = fitness

    def get_better_solution(self, compared_agent: 'Agent', minmax: str = "min") -> 'Agent':
        if minmax == "min":
            return self if self.fitness < compared_agent.fitness else compared_agent
        else:
            return compared_agent if self.fitness < compared_agent else self

    def is_duplicate(self, compared_agent: 'Agent') -> bool:
        if np.all(self.solution - compared_agent.solution) == 0:
            return True
        return False

    def compare_duplicate(self, compared_agent: 'Agent') -> bool:
        if np.all(self.solution - compared_agent.solution) == 0:
            self.fitness = compared_agent.fitness
            return True
        return False

    def __repr__(self):     # represent
        return f"id: {self.id}, fitness: {self.fitness}, solution: {self.solution}"
