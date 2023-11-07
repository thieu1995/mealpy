#!/usr/bin/env python
# Created by "Thieu" at 21:32, 05/11/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.special import erf


# ________________________V-shaped transfer functions______________________


def vstf_01(x):
    return np.abs(erf((np.sqrt(np.pi) / 2) * x))


def vstf_02(x):
    return np.abs(np.tanh(x))


def vstf_03(x):
    return np.abs(x / np.sqrt(1. + np.square(x)))


def vstf_04(x):
    return np.abs((2. / np.pi) * np.arctan((np.pi / 2) * x))


##______________________S-shaped transfer functions_______________________


def sstf_01(x):
    return 1 / (1 + np.exp(-2. * x))


def sstf_02(x):
    return 1. / (1 + np.exp(-x))


def sstf_03(x):
    return 1. / (1 + np.exp(-x / 3.))


def sstf_04(x):
    return 1. / (1 + np.exp(-x / 2.))
