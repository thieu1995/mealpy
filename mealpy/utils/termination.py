# !/usr/bin/env python
# Created by "Thieu" at 17:29, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.utils.logger import Logger
from mealpy.utils.boundary import is_in_bound, is_str_in_list


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
    >>> from mealpy.swarm_based.PSO import BasePSO
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
    >>> model1 = BasePSO(problem_dict, epoch=1000, pop_size=50, termination=term_dict)
    """

    SUPPORTED_TERMINATIONS = {
        "FE": ["Function Evaluation", [10, 1000000000]],
        "ES": ["Early Stopping", [1, 1000000]],
        "TB": ["Time Bound", [10, 1000000]],
        "MG": ["Maximum Generation", [1, 1000000]],
    }

    def __init__(self, **kwargs):
        self.exit_flag, self.message, self.log_to, self.log_file = False, "", None, None
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False
        self.__check_termination(kwargs)

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __check_termination(self, kwargs):
        if ("mode" in kwargs) and ("quantity" in kwargs):
            self.__check_mode(kwargs["mode"], kwargs["quantity"])
        elif ("termination" in kwargs) and type(kwargs["termination"] is dict):
            if ("mode" in kwargs["termination"]) and ("quantity" in kwargs["termination"]):
                self.__set_keyword_arguments(kwargs["termination"])
                self.__check_mode(kwargs["termination"]["mode"], kwargs["termination"]["quantity"])
            else:
                self.logger.error("You need to set up the termination dictionary with at least 'mode' and 'quantity'.")
                exit(0)
        else:
            self.logger.error("You need to set up the termination dictionary with at least 'mode' and 'quantity'.")
            exit(0)

    def __check_mode(self, mode, quantity):
        if is_str_in_list(mode, list(self.SUPPORTED_TERMINATIONS.keys())):
            self.mode = mode
            self.name = self.SUPPORTED_TERMINATIONS[mode][0]

            if type(quantity) in [int, float]:
                qt = int(quantity)
                if is_in_bound(qt, self.SUPPORTED_TERMINATIONS[mode][1]):
                    self.quantity = qt
                else:
                    self.logger.error(f"Mode: {mode}, 'quantity' is an integer and should be in range: {self.SUPPORTED_TERMINATIONS[mode][1]}.")
                    exit(0)
            else:
                self.logger.error(f"Mode: {mode}, 'quantity' is an integer and should be in range: {self.SUPPORTED_TERMINATIONS[mode][1]}.")
                exit(0)
        else:
            self.logger.error("Supported termination mode: FE (function evaluation), TB (time bound), ES (early stopping), MG (maximum generation).")
            exit(0)
