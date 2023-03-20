#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils import validator
import pytest


@pytest.mark.parametrize("value, bound, output",
                         [
                             (-3.3, [-10, 10], True),
                             (1000, (2, float("inf")), True),
                             (0.5, [0.3, 2], True),
                             (0, (0, 1.0), False),
                             [2.1, [1, 2.1], True]
                         ])
def test_check_int(value, bound, output):
    value_new = validator.is_in_bound(value, bound)
    assert value_new == output


@pytest.mark.parametrize("value, my_list, output",
                         [
                             (-3.3, [-10, 10], False),
                             (1000, (2, float("inf")), False),
                             (0.5, [0.3, 2], False),
                             ("hello", ["hello", "world", "now"], True),
                             ("a", ("b", "e", "A", "f"), False),
                             ("a", ("abc", "aa", "dc"), False)
                         ])
def test_check_str_in_list(value, my_list, output):
    value_new = validator.is_str_in_list(value, my_list)
    assert value_new == output


@pytest.mark.parametrize("value, bound, output",
                         [
                             (-3.3, [-10, 10], -3),
                             (1000, (2, float("inf")), 1000),
                             (0.5, [0.3, 2], 0),
                         ])
def test_check_bound(value, bound, output):
    valid_model = validator.Validator()
    value_new = valid_model.check_int("value", value, bound)
    assert value_new == output


@pytest.mark.parametrize("value, bound, output",
                         [
                             (None, [-10, 10], 0),
                             ("hello", (2, float("inf")), 0),
                             (-0.22, [0.3, 2], 0),
                             ([3, 2], (2, float("inf")), 0),
                             ((4, 2), [0.3, 2], 0),
                         ])
def test_check_float(value, bound, output):
    with pytest.raises(ValueError) as e:
        valid_model = validator.Validator()
        valid_model.check_float("value", value, bound)
    assert e.type == ValueError
