#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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

    def get_better_solution(self, compared_agent: 'Agent', minmax: str = "min") -> 'Agent':
        if minmax == "min":
            return self if self.target.fitness < compared_agent.target.fitness else compared_agent
        else:
            return compared_agent if self.target.fitness < compared_agent else self

    def is_duplicate(self, compared_agent: 'Agent') -> bool:
        if np.all(self.solution - compared_agent.solution) == 0:
            return True
        return False

    def compare_duplicate(self, compared_agent: 'Agent') -> bool:
        if np.all(self.solution - compared_agent.solution) == 0:
            self.target = compared_agent.target
            return True
        return False

    def is_better_than(self, compared_agent: 'Agent', minmax: str = "min") -> bool:
        if minmax == "min":
            return True if self.target.fitness < compared_agent.target.fitness else False
        else:
            return False if self.target.fitness < compared_agent.target.fitness else True

    def __repr__(self):     # represent
        return f"id: {self.id}, target: {self.target}, solution: {self.solution}"
