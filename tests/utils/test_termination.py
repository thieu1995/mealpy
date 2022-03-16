#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils.termination import Termination
import pytest


@pytest.mark.parametrize("mode, system_code",
                         [
                             ("FA", 0),
                             (3, 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_mode(mode, system_code):
    term = {
        "mode": mode,
        "quantity": 1000
    }
    with pytest.raises(SystemExit) as e:
        prob = Termination(termination=term)
    assert e.type == SystemExit
    assert e.value.code == system_code


@pytest.mark.parametrize("quantity, system_code",
                         [
                             ("FA", 0),
                             (-1, 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_quantity(quantity, system_code):
    term = {
        "mode": "MG",
        "quantity": quantity
    }
    with pytest.raises(SystemExit) as e:
        prob = Termination(termination=term)
    assert e.type == SystemExit
    assert e.value.code == system_code
