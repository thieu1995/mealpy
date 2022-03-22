# !/usr/bin/env python
# Created by "Thieu" at 17:28, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.utils.logger import Logger


class Problem:
    """
    Define the mathematical form of optimization problem

    Notes
    ~~~~~
    + fit_func (callable): your fitness function
    + lb (list, int, float): lower bound, should be list of values
    + ub (list, int, float): upper bound, should be list of values
    + minmax (str): "min" or "max" problem (Optional, default = "min")
    + verbose (bool): print out the training process or not (Optional, default = True)
    + n_dims (int): number of dimensions / problem size (Optional)
    + obj_weights: list weights for all your objectives (Optional, default = [1, 1, ...1])
    + problem (dict): dictionary of the problem (contains at least the parameter 1, 2, 3) (Optional)
    + save_population (bool): save history of population or not, default = True (Optional). **Warning**:
        + this parameter can save you from error related to 'memory' when your model is too big (i.e, training neural network, ...)
        + when set to False, you can't use the function draw trajectory chart in history object (model.history.save_trajectory_chart)
    + amend_position(callable): Depend on your problem, may need to design an amend_position function (Optional for continuous domain, Required for discrete domain)

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
    >>>     "minmax": "min",
    >>>     "log_to": None,
    >>>     "save_population": False,
    >>> }
    >>> model1 = BasePSO(problem_dict, epoch=1000, pop_size=50)
    >>>
    >>> ## For discrete problem, you need to design an amend_position function that can (1) bring your solution back to the valid range,
    >>> ##    (2) can convert float number into integer number (combinatorial or permutation).
    >>>
    >>> def amend_position(solution, lb, ub):
    >>>     ## Bring them back to valid range
    >>>     solution = np.clip(solution, lb, ub)
    >>>     ## Convert float to integer number
    >>>     solution_int = solution.astype(int)
    >>>     ## If the designed solution is permutation, then need an extra step here
    >>>     ## .... Do it here and then return the valid solution
    >>>     return solution_int
    >>>
    >>> problem_dict2 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-100, ] * 30,
    >>>     "ub": [100, ] * 30,
    >>>     "minmax": "min",
    >>>     "log_to": "file",
    >>>     "log_file": "records.log",
    >>>     "amend_position": amend_position
    >>> }
    >>> model2 = BasePSO(problem_dict2, epoch=1000, pop_size=50)
    >>> best_position, best_fitness = model2.solve()
    >>> print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
    """

    def __init__(self, **kwargs):
        self.minmax = "min"
        self.log_to, self.log_file = "console", None
        self.n_objs = 1
        self.obj_weights = None
        self.multi_objs = False
        self.obj_is_list = False
        self.n_dims, self.lb, self.ub = None, None, None
        self.save_population = True
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.__check_problem(kwargs)

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __check_problem(self, kwargs):
        if ("fit_func" in kwargs) and ("lb" in kwargs) and ("ub" in kwargs):
            self.__set_problem(kwargs)
        elif ("problem" in kwargs) and (type(kwargs["problem"]) is dict):
            if ("fit_func" in kwargs["problem"]) and ("lb" in kwargs["problem"]) and ("ub" in kwargs["problem"]):
                self.__set_keyword_arguments(kwargs["problem"])
                self.__set_problem(kwargs["problem"])
            else:
                self.logger.error("Defined a problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
                exit(0)
        else:
            self.logger.error("Defined a problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
            exit(0)

    def __set_problem(self, problem):
        lb, ub, fit_func = problem["lb"], problem["ub"], problem["fit_func"]
        self.__set_domain_range(lb, ub, problem)
        self.__set_fitness_function(fit_func, problem)

    def __set_domain_range(self, lb, ub, kwargs):
        if isinstance(lb, list) and isinstance(ub, list):
            if len(lb) == len(ub):
                if len(lb) > 1:
                    self.n_dims = len(lb)
                    self.lb = np.array(lb)
                    self.ub = np.array(ub)
                elif len(lb) == 1:
                    if "n_dims" in kwargs:
                        self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                        self.lb = lb[0] * np.ones(self.n_dims)
                        self.ub = ub[0] * np.ones(self.n_dims)
                    else:
                        self.logger.error("n_dims is required in defined problem when lb and ub are a list of 1 element.")
                        exit(0)
                else:
                    self.logger.error("Lower bound and upper bound need to be a list of values and same length.")
                    exit(0)
            else:
                self.logger.error("Lower bound and upper bound need to be same length.")
                exit(0)
        elif type(lb) in [int, float] and type(ub) in [int, float]:
            if "n_dims" in kwargs:
                self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                self.lb = lb * np.ones(self.n_dims)
                self.ub = ub * np.ones(self.n_dims)
            else:
                self.logger.error("Parameter n_dims is required in problem dictionary when lb and ub are a single value.")
                exit(0)
        else:
            self.logger.error("Lower bound and Upper bound need to be a list and same length.")
            exit(0)

    def __check_problem_size(self, n_dims):
        if type(n_dims) == int and n_dims > 1:
            return int(n_dims)
        else:
            self.logger.error("n_dims (problem size) must be an integer number and > 1.")
            exit(0)

    def __set_fitness_function(self, fit_func, kwargs):
        tested_solution = self.generate_position(self.lb, self.ub)
        if callable(fit_func):
            self.fit_func = fit_func
        else:
            self.logger.error("Please enter your 'fit_func' as a callable function, and it needs to return a value or list of values.")
            exit(0)
        if "amend_position" in kwargs:
            if callable(kwargs["amend_position"]):
                tested_solution = self.amend_position(tested_solution, self.lb, self.ub)
            else:
                self.logger.error("Please enter your 'amend_position' as a callable function, and it needs to return amended solution.")
                exit(0)
        result = self.fit_func(tested_solution)
        if isinstance(result, list) or isinstance(result, np.ndarray):
            self.n_objs = len(result)
            self.obj_is_list = True
            if self.n_objs > 1:
                self.multi_objs = True
                if "obj_weights" in kwargs:
                    self.obj_weights = kwargs["obj_weights"]
                    if isinstance(self.obj_weights, list) or isinstance(self.obj_weights, np.ndarray):
                        if self.n_objs != len(self.obj_weights):
                            self.logger.error(f"{self.n_objs}-objective problem, but N weights: {len(self.obj_weights)}.")
                            exit(0)
                        self.msg = f"N objs: {self.n_objs} with N weights: {self.obj_weights}"
                    else:
                        self.logger.error(f"{self.n_objs}-objective problem, obj_weights must be a list or numpy np.array with length: {self.n_objs}.")
                        exit(0)
                else:
                    self.obj_weights = np.ones(self.n_objs)
                    self.msg = f"{self.n_objs}-objective problem and default weights: {self.obj_weights}."
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weights = np.ones(1)
                self.msg = f"Solving single objective optimization problem."
            else:
                self.logger.error(f"Fitness function needs to return a single value or list of values.")
                exit(0)
        elif type(result) in (int, float) or isinstance(result, np.floating) or isinstance(result, np.integer):
            self.multi_objs = False
            self.obj_is_list = False
            self.obj_weights = np.ones(1)
            self.msg = f"Solving single objective optimization problem."
        else:
            self.logger.error("Fitness function needs to return a single value or a list of values.")
            exit(0)

    def generate_position(self, lb=None, ub=None):
        """
        Generate the position depends on the problem. For discrete problem such as permutation, this method can be override.

        Args:
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            np.array: the position (the solution for the problem)
        """
        return np.random.uniform(lb, ub)

    def amend_position(self, position=None, lb=None, ub=None):
        """
        + This is default function in most algorithms. Otherwise, there will be an overridden function
        in child of Optimizer class for this function.
        + Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        # return np.maximum(self.problem.lb, np.minimum(self.problem.ub, position))
        return np.clip(position, lb, ub)
