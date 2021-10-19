#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:28, 13/10/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np


class Problem:
    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    DEFAULT_BATCH_IDEA = False
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_LB = -1
    DEFAULT_UB = 1

    def __init__(self, problem: dict):
        """
        Args:
            problem (dict): Dict properties of your problem

        Examples:
             problem = {
                "obj_func": your objective function,
                "lb": list of value
                "ub": list of value
                "minmax": "min" or "max"
                "verbose": True,
                "n_dims": int (Optional)
                "batch_idea": True or False (Optional)
                "batch_size": int (Optional, smaller than population size)
                "obj_weight": list weights for all your objectives (Optional, default = [1, 1, ...1])
             }
        """
        self.minmax = "min"
        self.batch_size = 10
        self.batch_idea = False
        self.verbose = True
        self.n_objs = 1
        self.obj_weight = None
        self.multi_objs = False
        self.obj_is_list = False
        self.n_dims, self.lb, self.ub = None, None, None
        self.__set_parameters__(problem)
        self.__check_parameters__(problem)
        self.__check_optional_parameters__(problem)
        self.__check_objective_function__(problem)

    def __set_parameters__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __check_parameters__(self, kwargs):
        if "lb" in kwargs and "ub" in kwargs:
            lb, ub = kwargs["lb"], kwargs["ub"]
            if (lb is None) or (ub is None):
                if "n_dims" in kwargs:
                    print(f"Default lb={self.DEFAULT_LB}, ub={self.DEFAULT_UB}.")
                    self.n_dims = self.__check_problem_size__(kwargs["n_dims"])
                    self.lb = self.DEFAULT_LB * np.ones(self.n_dims)
                    self.ub = self.DEFAULT_UB * np.ones(self.n_dims)
                else:
                    print("If lb, ub are undefined, then you must set problem size to be an integer.")
                    exit(0)
            else:
                if isinstance(lb, list) and isinstance(ub, list):
                    if len(lb) == len(ub):
                        if len(lb) == 0:
                            if "n_dims" in kwargs:
                                print(f"Default lb={self.DEFAULT_LB}, ub={self.DEFAULT_UB}.")
                                self.n_dims = self.__check_problem_size__(kwargs["n_dims"])
                                self.lb = self.DEFAULT_LB * np.ones(self.n_dims)
                                self.ub = self.DEFAULT_UB * np.ones(self.n_dims)
                            else:
                                print("Wrong lower bound and upper bound parameters.")
                                exit(0)
                        elif len(lb) == 1:
                            if "n_dims" in kwargs:
                                self.n_dims = self.__check_problem_size__(kwargs["n_dims"])
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
                    self.n_dims = self.__check_problem_size__(kwargs["n_dims"])
                    self.lb = lb * np.ones(self.n_dims)
                    self.ub = ub * np.ones(self.n_dims)
                else:
                    print("Lower bound and Upper bound need to be a list.")
                    exit(0)
        else:
            print("Please define lb (lower bound) and ub (upper bound) values!")
            exit(0)

    def __check_problem_size__(self, n_dims):
        if n_dims is None:
            print("Problem size must be an int number")
            exit(0)
        elif n_dims <= 0:
            print("Problem size must > 0")
            exit(0)
        return int(np.ceil(n_dims))

    def __check_optional_parameters__(self, kwargs):
        if "batch_idea" in kwargs:
            batch_idea = kwargs["batch_idea"]
            if type(batch_idea) == bool:
                self.batch_idea = batch_idea
            else:
                self.batch_idea = self.DEFAULT_BATCH_IDEA
            if "batch_size" in kwargs:
                batch_size = kwargs["batch_size"]
                if type(batch_size) == int:
                    self.batch_size = batch_size
                else:
                    self.batch_size = self.DEFAULT_BATCH_SIZE
            else:
                self.batch_size = self.DEFAULT_BATCH_SIZE
        else:
            self.batch_idea = self.DEFAULT_BATCH_IDEA

    def __check_objective_function__(self, kwargs):
        if "obj_func" in kwargs:
            obj_func = kwargs["obj_func"]
            if callable(obj_func):
                self.obj_func = obj_func
            else:
                print("Please check your function. It needs to return value!")
                exit(0)
        tested_solution = np.random.uniform(self.lb, self.ub)
        try:
            result = self.obj_func(tested_solution)
        except Exception as err:
            print(f"Error: {err}\n")
            print("Please check your defined objective function!")
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
                            print(f"Please check your objective function/weight. N objs = {self.n_objs}, but N weights = {len(self.obj_weight)}")
                            exit(0)
                        print(f"N objs = {self.n_objs} with weights = {self.obj_weight}")
                    else:
                        print(
                            f"Please check your objective function/weight. N objs = {self.n_objs}, weights must be a list or numpy np.array with same length.")
                        exit(0)
                else:
                    self.obj_weight = np.ones(self.n_objs)
                    print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weight = np.ones(1)
                print(f"N objs = {self.n_objs} with default weights = {self.obj_weight}")
            else:
                print(f"Please check your objective function. It returns nothing!")
                exit(0)
        else:
            if isinstance(result, np.floating) or type(result) in (int, float):
                self.multi_objs = False
                self.obj_is_list = False
                self.obj_weight = np.ones(1)
            else:
                print("Please check your objective function. It needs to return value!")
                exit(0)
