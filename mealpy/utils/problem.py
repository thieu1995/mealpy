# !/usr/bin/env python
# Created by "Thieu" at 17:28, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np


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
    + obj_weight: list weights for all your objectives (Optional, default = [1, 1, ...1])
    + problem (dict): dictionary of the problem (contains at least the parameter 1, 2, 3) (Optional)
    + amend_position(callable): Depend on your problem, may need to design an amend_position function (Optional for continuous domain, Required for discrete domain)

    Examples
    ~~~~~~~~
    >>> ## 1st way:
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
    >>>     "verbose": True,
    >>> }
    >>> model1 = BasePSO(problem_dict, epoch=1000, pop_size=50)
    >>>
    >>> ## 2nd way:
    >>> from mealpy.utils.problem import Problem
    >>>
    >>> problem_obj2 = Problem(fit_func=fitness_function, lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20], minmax="min", verbose=True)
    >>> model3 = BasePSO(problem_obj2, epoch=1000, pop_size=50)
    """

    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    DEFAULT_LB = -1
    DEFAULT_UB = 1

    def __init__(self, **kwargs):
        self.minmax = "min"
        self.verbose = True
        self.n_objs = 1
        self.obj_weight = None
        self.multi_objs = False
        self.obj_is_list = False
        self.n_dims, self.lb, self.ub = None, None, None

        self.__set_keyword_arguments(kwargs)
        self.__check_problem(kwargs)

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __check_problem(self, kwargs):
        if ("fit_func" not in kwargs) and ("lb" not in kwargs) and ("ub" not in kwargs):
            if "problem" not in kwargs:
                print("You need to set up the problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
                exit(0)
            else:
                problem = kwargs["problem"]
                if type(problem) is not dict:
                    print("You need to set up the problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
                    exit(0)
                else:
                    if ("fit_func" not in problem) or ("lb" not in problem) or ("ub" not in problem):
                        print("You need to set up the problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
                        exit(0)
                    else:
                        lb, ub, fit_func = problem["lb"], problem["ub"], problem["fit_func"]
                        self.__set_problem(lb, ub, fit_func, problem)
        elif ("fit_func" in kwargs) and ("lb" in kwargs) and ("ub" in kwargs):
            lb, ub, fit_func = kwargs["lb"], kwargs["ub"], kwargs["fit_func"]
            self.__set_problem(lb, ub, fit_func, kwargs)
        else:
            print("You need to set up the problem dictionary with at least 'fit_func', 'lb', and 'ub'.")
            exit(0)

    def __set_problem(self, lb, ub, fit_func, kwargs):
        self.__set_domain_range(lb, ub, kwargs)
        self.__set_fitness_function(fit_func, kwargs)

    def __set_domain_range(self, lb, ub, kwargs):
        if (lb is None) or (ub is None):
            if "n_dims" in kwargs:
                self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                self.lb = self.DEFAULT_LB * np.ones(self.n_dims)
                self.ub = self.DEFAULT_UB * np.ones(self.n_dims)
                print(f"Default lb={self.lb}, ub={self.ub}.")
            else:
                print("If lb and ub are undefined, then you must set 'n_dims' to be an integer.")
                exit(0)
        else:
            if isinstance(lb, list) and isinstance(ub, list):
                if len(lb) == len(ub):
                    if len(lb) == 0:
                        if "n_dims" in kwargs:
                            self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                            self.lb = self.DEFAULT_LB * np.ones(self.n_dims)
                            self.ub = self.DEFAULT_UB * np.ones(self.n_dims)
                            print(f"Default lb={self.lb}, ub={self.ub}.")
                        else:
                            print("Please set up your lower bound and upper bound.")
                            exit(0)
                    elif len(lb) == 1:
                        if "n_dims" in kwargs:
                            self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                            self.lb = lb[0] * np.ones(self.n_dims)
                            self.ub = ub[0] * np.ones(self.n_dims)
                    else:
                        self.n_dims = len(lb)
                        self.lb = np.array(lb)
                        self.ub = np.array(ub)
                else:
                    print("Lower bound and Upper bound need to be same length")
                    exit(0)
            elif type(lb) in [int, float] and type(ub) in [int, float]:
                self.n_dims = self.__check_problem_size(kwargs["n_dims"])
                self.lb = lb * np.ones(self.n_dims)
                self.ub = ub * np.ones(self.n_dims)
            else:
                print("Lower bound and Upper bound need to be a list.")
                exit(0)

    def __check_problem_size(self, n_dims):
        if n_dims is None or n_dims <= 0:
            print("Problem size must be an int number and > 0.")
            exit(0)
        return int(np.ceil(n_dims))

    def __set_fitness_function(self, fit_func, kwargs):
        if callable(fit_func):
            self.fit_func = fit_func
        else:
            print("Please check your fitness function. It needs to return single value or a list of values!")
            exit(0)

        tested_solution = np.random.uniform(self.lb, self.ub)
        if "amend_position" in kwargs:
            tested_solution = self.amend_position(tested_solution)
        result = None
        try:
            result = self.fit_func(tested_solution)
        except Exception as err:
            print(f"Error: {err}\n")
            print("Please check your fitness function. It needs to return single value or a list of values!")
            exit(0)
        if isinstance(result, list) or isinstance(result, np.ndarray):
            self.n_objs = len(result)
            self.obj_is_list = True
            if self.n_objs > 1:
                self.multi_objs = True
                if "obj_weight" in kwargs:
                    self.obj_weight = kwargs["obj_weight"]
                    if isinstance(self.obj_weight, list) or isinstance(self.obj_weight, np.ndarray):
                        if self.n_objs != len(self.obj_weight):
                            print(f"Please check your fitness function/weight. N objs = {self.n_objs}, but N weights = {len(self.obj_weight)}")
                            exit(0)
                        print(f"N objs = {self.n_objs} with weights = {self.obj_weight}")
                    else:
                        print(
                            f"Please check your fitness function/weight. N objs = {self.n_objs}, weights must be a list or numpy np.array with same length.")
                        exit(0)
                else:
                    self.obj_weight = np.ones(self.n_objs)
                    print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weight = np.ones(1)
                print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            else:
                print(f"Please check your fitness function. It returns nothing!")
                exit(0)
        else:
            if type(result) in (int, float) or isinstance(result, np.floating) or isinstance(result, np.integer):
                self.multi_objs = False
                self.obj_is_list = False
                self.obj_weight = np.ones(1)
            else:
                print("Please check your fitness function. It needs to return single value or a list of values!")
                exit(0)

    def amend_position(self, position=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.

        Returns:
            Amended position (make the position is in bound)
        """
        # return np.maximum(self.problem.lb, np.minimum(self.problem.ub, position))
        return np.clip(position, self.lb, self.ub)
