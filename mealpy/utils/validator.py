#!/usr/bin/env python
# Created by "Thieu" at 21:32, 14/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import operator
from mealpy.utils.logger import Logger


def is_in_bound(value, bound):
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
    elif ops(bound[0], value) and ops(value, bound[1]):
        return True
    return False


def is_str_in_list(value: str, my_list: list):
    if type(value) == str and my_list is not None:
        return True if value in my_list else False
    return False


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

    def check_int(self, name:str, value: int, bound=None):
        if type(value) in [int, float]:
            if bound is None:
                return int(value)
            elif is_in_bound(value, bound):
                return int(value)
        bound = "" if bound is None else f"and value should be in range: {bound}"
        raise ValueError(f"'{name}' is an integer {bound}.")

    def check_float(self, name: str, value: float, bound=None):
        if type(value) in [int, float]:
            if bound is None:
                return float(value)
            elif is_in_bound(value, bound):
                return float(value)
        bound = "" if bound is None else f"and value should be in range: {bound}"
        raise ValueError(f"'{name}' is a float {bound}.")

    def check_str(self, name: str, value: str, bound=None):
        if type(value) is str:
            if bound is None or is_str_in_list(value, bound):
                return value
        bound = "" if bound is None else f"and value should be one of this: {bound}"
        raise ValueError(f"'{name}' is a string {bound}.")

    def check_bool(self, name: str, value: bool, bound=(True, False)):
        if type(value) is bool:
            if value in bound:
                return value
        bound = "" if bound is None else f"and value should be one of this: {bound}"
        raise ValueError(f"'{name}' is a boolean {bound}.")

    def check_tuple_int(self, name: str, values: tuple, bounds=None):
        if type(values) in [tuple, list] and len(values) > 1:
            value_flag = [type(item) == int for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f"and values should be in range: {bounds}"
        raise ValueError(f"'{name}' are integer {bounds}.")

    def check_tuple_float(self, name: str, values: tuple, bounds=None):
        if type(values) in [tuple, list] and len(values) > 1:
            value_flag = [type(item) in [int, float] for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f"and values should be in range: {bounds}"
        raise ValueError(f"'{name}' are float {bounds}.")

    def check_list_tuple(self, name: str, value: any, data_type: str):
        if type(value) in (tuple, list) and len(value) >= 1:
            return list(value)
        raise ValueError(f"'{name}' should be a list or tuple of {data_type}, and length >= 1.")

    def check_is_instance(self, name: str, value: any, class_type: any):
        if isinstance(value, class_type):
            return value
        raise ValueError(f"'{name}' should be an instance of {class_type} class.")

    def check_is_int_and_float(self, name: str, value: any, bound_int=None, bound_float=None):
        if type(value) is int:
            if bound_int is None or is_in_bound(value, bound_int):
                return int(value)
        bound_int_str = "" if bound_int is None else f"and value in range: {bound_int}"
        if type(value) is float:
            if bound_float is None or is_in_bound(value, bound_float):
                return float(value)
        bound_float_str = "" if bound_float is None else f"and value in range: {bound_float}"
        raise ValueError(f"'{name}' can be int {bound_int_str}, or float {bound_float_str}.")
