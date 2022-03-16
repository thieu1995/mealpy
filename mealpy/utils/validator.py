#!/usr/bin/env python
# Created by "Thieu" at 21:32, 14/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import operator
from mealpy.utils.logger import Logger


class Validator:
    def __init__(self, **kwargs):
        self.log_to, self.log_file = None, None
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __is_in_bound(self, value, bound):
        ops = None
        if type(bound) is tuple:
            ops = operator.lt
        elif type(bound) is list:
            ops = operator.le
        if bound[0] == float("-inf") and bound[1] == float("inf"):
            return True
        elif bound[0] == float("-inf") and ops(value, bound[1]):
            return True
        elif ops(bound[0], value) and bound[1] == float("inf"):
            return True
        elif ops(bound[0], value) and ops(value,  bound[1]):
            return True
        return False

    def check_int(self, name:str, value:int, bound=None):
        if type(value) in [int, float]:
            if bound is None:
                return int(value)
            elif self.__is_in_bound(value, bound):
                return int(value)
        bound = "" if bound is None else f", and valid range is: {bound}"
        self.logger.error(f"'{name}' should be an integer{bound}.")
        exit(0)

    def check_float(self, name: str, value: int, bound=None):
        if type(value) in [int, float]:
            if bound is None:
                return float(value)
            elif self.__is_in_bound(value, bound):
                return float(value)
        bound = "" if bound is None else f", and valid range is: {bound}"
        self.logger.error(f"'{name}' should be a float{bound}.")
        exit(0)

    def check_tuple_int(self, name: str, values: tuple, bounds=None):
        if type(values) in [tuple, list] and len(values) > 1:
            value_flag = [type(item) == int for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [self.__is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f", and should be in range: {bounds}"
        self.logger.error(f"'{name}' are int values{bounds}.")
        exit(0)

    def check_tuple_float(self, name: str, values: tuple, bounds=None):
        if type(values) in [tuple, list] and len(values) > 1:
            value_flag = [type(item) in [int, float] for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [self.__is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f", and should be in range: {bounds}"
        self.logger.error(f"'{name}' are float values{bounds}.")
        exit(0)

    def __is_in_list(self, value, my_list):
        return True if value in my_list else False

    def check_str_in_list(self, value: str, my_list: list):
        if type(value) == str and my_list is not None:
            return self.__is_in_list(value, my_list)
        return False















