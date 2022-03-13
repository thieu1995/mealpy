# !/usr/bin/env python
# Created by "Thieu" at 17:29, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

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

    DEFAULT_MAX_MG = 1000  # Maximum number of epochs / generations (Default: 1000 epochs)
    DEFAULT_MAX_FE = 100000  # Maximum number of function evaluation (Default: 100000 FE)
    DEFAULT_MAX_TB = 20  # Maximum number of time bound (Default: 20 seconds)
    DEFAULT_MAX_ES = 20  # Maximum number of early stopping iterations (Default: 20 loops / generations)

    def __init__(self, **kwargs):
        self.name = "Maximum Generation"
        self.mode = "MG"
        self.quantity = self.DEFAULT_MAX_MG
        self.exit_flag, self.message = False, ""
        self.logger = Logger().create_console_logger(name=f"{__name__}.{__class__.__name__}",
                                                     format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.__set_keyword_arguments(kwargs)
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
        if type(mode) == str:
            self.mode = mode
            if mode == "FE":
                self.name = "Function Evaluation"
                self.__check_quantity(quantity, self.DEFAULT_MAX_FE)
            elif self.mode == "TB":
                self.name = "Time Bound"
                self.__check_quantity(quantity, self.DEFAULT_MAX_TB)
            elif self.mode == "ES":
                self.name = "Early Stopping"
                self.__check_quantity(quantity, self.DEFAULT_MAX_ES)
            elif self.mode == "MG":
                self.name = "Maximum Generation"
                self.__check_quantity(quantity, self.DEFAULT_MAX_MG)
            else:
                self.logger.error("Your selected mode of stopping condition is not support, please choice another one.")
                exit(0)
        else:
            self.logger.error("Please set up your termination 'mode' as 'string' and 'quantity' as integer.")
            exit(0)

    def __check_quantity(self, quantity, default_value):
        self.quantity = quantity if (type(quantity) is int and quantity > 0) else default_value
