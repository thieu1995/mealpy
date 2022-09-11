# !/usr/bin/env python
# Created by "Thieu" at 17:29, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import time
from mealpy.utils import validator
from mealpy.utils.logger import Logger


class Termination:
    """
    Define the Stopping Condition (Termination) for the Optimizer

    Notes
    ~~~~~
    + By default, the stopping condition is maximum generations (epochs/iterations) in Optimizer class.
    + By using this class, the default termination will be overridden
    + In general, there are 4 termination cases: FE, MG, ES, TB
        + FE: Number of Function Evaluation
        + MG: Maximum Generations / Epochs -  This is default in all algorithms
        + ES: Early Stopping - Same idea in training neural network (If the global best solution not better an epsilon after K epochs then stop the program)
        + TB: Time Bound - You just want your algorithm run in K seconds. Especially when comparing different algorithms.

    + Parameters for Termination class
        + mode (str): FE, MG, ES or TB
        + quantity (int): value for termination type
        + termination (dict): dictionary of the termination (contains at least the parameter 'mode' and 'quantity') (Optional)

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
    >>>     "mode": "FE",
    >>>     "quantity": 100000  # 100000 number of function evaluation
    >>> }
    >>> model1 = OriginalPSO(epoch=1000, pop_size=50)
    >>> model1.solve(problem_dict, termination=term_dict)
    """

    SUPPORTED_TERMINATIONS = {
        "FE": ["Function Evaluation", [10, 1000000000]],
        "ES": ["Early Stopping", [1, 1000000]],
        "TB": ["Time Bound", [1, 1000000]],
        "MG": ["Maximum Generation", [1, 1000000]],
    }

    def __init__(self, mode="FE", quantity=10000, **kwargs):
        self.mode, self.quantity, self.name = None, None, None
        self.exit_flag, self.message, self.log_to, self.log_file = False, "", None, None
        self.__set_keyword_arguments(kwargs)
        self.__set_termination(mode, quantity)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_termination(self, mode, quantity):
        if validator.is_str_in_list(mode, list(Termination.SUPPORTED_TERMINATIONS.keys())):
            self.mode = mode
            self.name = Termination.SUPPORTED_TERMINATIONS[mode][0]
            if type(quantity) in (int, float):
                qt = int(quantity)
                if validator.is_in_bound(qt, Termination.SUPPORTED_TERMINATIONS[mode][1]):
                    self.quantity = qt
                else:
                    raise ValueError(f"Mode: {mode}, 'quantity' is an integer and should be in range: {Termination.SUPPORTED_TERMINATIONS[mode][1]}.")
            else:
                raise ValueError(f"Mode: {mode}, 'quantity' is an integer and should be in range: {Termination.SUPPORTED_TERMINATIONS[mode][1]}.")
        else:
            raise ValueError("Supported termination mode: FE (function evaluation), TB (time bound), ES (early stopping), MG (maximum generation).")

    def get_name(self):
        return self.name

    def get_default_counter(self, epoch):
        if self.mode in ["ES", "FE"]:
            return 0
        elif self.mode == "TB":
            return time.perf_counter()
        else:
            return epoch

    def is_finished(self, counter):
        if counter >= self.quantity:
            return True
        return False
