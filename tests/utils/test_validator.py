#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils.validator import Validator
import pytest


@pytest.mark.parametrize("value, bound, output",
                         [
                             (-3.3, [-10, 10], -3),
                             (1000, (2, float("inf")), 1000),
                             (0.5, [0.3, 2], 0),
                         ])
def test_check_int(value, bound, output):
    validator = Validator()
    value_new = validator.check_int("value", value, bound)
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
    with pytest.raises(SystemExit) as e:
        validator = Validator()
        value_new = validator.check_float("value", value, bound)
    assert e.type == SystemExit
    assert e.value.code == output
