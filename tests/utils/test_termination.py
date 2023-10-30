#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import Termination
import pytest


@pytest.mark.parametrize("max_epoch, system_code",
                         [
                             ("FA", 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_max_epoch(max_epoch, system_code):
    term = {
        "max_epoch": max_epoch,
    }
    with pytest.raises(ValueError) as e:
        Termination(termination=term)
    assert e.type == ValueError


@pytest.mark.parametrize("max_time, system_code",
                         [
                             ("FA", 0),
                             (-1, 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_max_time(max_time, system_code):
    term = {
        "max_time": max_time
    }
    with pytest.raises(ValueError) as e:
        Termination(termination=term)
    assert e.type == ValueError


@pytest.mark.parametrize("max_fe, system_code",
                         [
                             ("FA", 0),
                             (-1, 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_max_fe(max_fe, system_code):
    term = {
        "max_fe": max_fe
    }
    with pytest.raises(ValueError) as e:
        Termination(termination=term)
    assert e.type == ValueError


@pytest.mark.parametrize("max_early_stop, system_code",
                         [
                             ("FA", 0),
                             (-1, 0),
                             ([1, 2], 0),
                             ((0.3, 2), 0),
                             (None, 0),
                         ])
def test_max_early_stop(max_early_stop, system_code):
    term = {
        "max_early_stop": max_early_stop
    }
    with pytest.raises(ValueError) as e:
        Termination(termination=term)
    assert e.type == ValueError
