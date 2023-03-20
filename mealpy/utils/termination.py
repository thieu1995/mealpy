#!/usr/bin/env python
# Created by "Thieu" at 22:23, 17/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils.logger import Logger
from mealpy.utils.validator import Validator


class Termination:
    """
    Define custom single/multiple Stopping Conditions (termination criteria) for the Optimizer.

    Notes
    ~~~~~
    + By default, the stopping condition in the Optimizer class is based on the maximum number of generations (epochs/iterations).
    + Using this class allows you to override the default termination criteria. If multiple stopping conditions are specified, the first one that occurs will be used.

    + In general, there are four types of termination criteria: FE, MG, TB, and ES.
        + MG: Maximum Generations / Epochs / Iterations
        + FE: Maximum Number of Function Evaluations
        + TB: Time Bound - If you want your algorithm to run for a fixed amount of time (e.g., K seconds), especially when comparing different algorithms.
        + ES: Early Stopping -  Similar to the idea in training neural networks (stop the program if the global best solution has not improved by epsilon after K epochs).

    + Parameters for Termination class, set it to None if you don't want to use it
        + max_epoch (int): Indicates the maximum number of generations for the MG type.
        + max_fe (int): Indicates the maximum number of function evaluations for the FE type.
        + max_time (float): Indicates the maximum amount of time for the TB type.
        + max_early_stop (int): Indicates the maximum number of epochs for the ES type.
            + epsilon (float): (Optional) This is used for the ES termination type (default value: 1e-10).
        + termination (dict): (Optional) A dictionary of termination criteria.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import OriginalPSO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>> }
    >>>
    >>> term_dict = {
    >>>     "max_epoch": 1000,
    >>>     "max_fe": 100000,  # 100000 number of function evaluation
    >>>     "max_time": 10,     # 10 seconds to run the program
    >>>     "max_early_stop": 15    # 15 epochs if the best fitness is not getting better we stop the program
    >>> }
    >>> model1 = OriginalPSO(epoch=1000, pop_size=50)
    >>> model1.solve(problem_dict, termination=term_dict)
    """

    def __init__(self, max_epoch=None, max_fe=None, max_time=None, max_early_stop=None, **kwargs):
        self.max_epoch = max_epoch
        self.max_fe = max_fe
        self.max_time = max_time
        self.max_early_stop = max_early_stop
        self.epsilon = 1e-10
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)
        self.name, self.message, self.log_to, self.log_file = "Termination", "", None, None
        self.__set_condition(self.max_epoch, self.max_fe, self.max_time, self.max_early_stop)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
                                    format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False

    def __set_keyword_arguments(self, kwargs):
        if type(kwargs) == dict:
            if type(kwargs.get("termination")) == dict:
                for key, value in kwargs.items():
                    setattr(self, key, value)
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __set_condition(self, max_epoch, max_fe, max_time, max_early_stop):
        if (max_epoch is None) and (max_fe is None) and (max_time is None) and (max_early_stop is None):
            raise ValueError("Please set at least one stopping condition with parameter 'max_epoch' or 'max_fe' or 'max_time' or 'max_early_stop'")
        else:
            if max_epoch is not None:
                self.max_epoch = self.validator.check_int("max_epoch", max_epoch, [1, 10000000])
            if max_fe is not None:
                self.max_fe = self.validator.check_int("max_fe", max_fe, [10, 1000000000])
            if max_time is not None:
                self.max_time = self.validator.check_float("max_time", max_time, [0.1, 1000000])
            if max_early_stop is not None:
                self.max_early_stop = self.validator.check_int("max_early_stop", max_early_stop, [1, 100000])

    def get_name(self):
        return self.name

    def set_start_values(self, start_epoch, start_fe, start_time, start_threshold):
        self.start_epoch = start_epoch
        self.start_fe = start_fe
        self.start_time = start_time
        self.start_threshold = start_threshold

    def should_terminate(self, current_epoch, current_fe, current_time, current_threshold):
        # Check maximum number of generations
        if self.max_epoch is not None and current_epoch >= self.max_epoch:
            self.message = "Stopping criterion with maximum number of epochs/generations/iterations (MG) occurred. End program!"
            return True
        # Check maximum number of function evaluations
        if self.max_fe is not None and current_fe >= self.max_fe:
            self.message = "Stopping criterion with maximum number of function evaluations (FE) occurred. End program!"
            return True
        # Check maximum time
        if self.max_time is not None and current_time >= self.max_time:
            self.message = "Stopping criterion with maximum running time/time bound (TB) (seconds) occurred. End program!"
            return True
        # Check early stopping
        if self.max_early_stop is not None and current_threshold >= self.max_early_stop:
            self.message = "Stopping criterion with early stopping (ES) (fitness-based) occurred. End program!"
            return True
        return False
