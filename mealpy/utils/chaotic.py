#!/usr/bin/env python
# Created by "Thieu" at 23:18, 17/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np


class ChaoticMap:
    """
    Implementation of 10 chaotic maps
    """

    @staticmethod
    def bernoulli_map(x: float, a: float = 0.5) -> float:
        """Bernoulli map"""
        if 0 <= x <= a:
            return x / (1 - a)
        else:
            return (x - a) / a

    @staticmethod
    def logistic_map(x: float, a: float = 4.0) -> float:
        """Logistic map"""
        return a * x * (1 - x)

    @staticmethod
    def chebyshev_map(x: float, a: float = 4.0) -> float:
        """Chebyshev map"""
        return np.cos(a * np.arccos(x))

    @staticmethod
    def circle_map(x: float, a: float = 0.5, b: float = 0.2) -> float:
        """Circle map"""
        return (x + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x)) % 1

    @staticmethod
    def cubic_map(x: float, q: float = 2.59) -> float:
        """Cubic map"""
        return q * x * (1 - x * x)

    @staticmethod
    def icmic_map(x: float, a: float = 0.7) -> float:
        """Iterative chaotic map with infinite collapses"""
        if x == 0:
            return 0.1  # Avoid division by zero
        return np.abs(np.sin(a / x))

    @staticmethod
    def piecewise_map(x: float, a: float = 0.4) -> float:
        """Piecewise map"""
        if 0 <= x < a:
            return x / a
        elif a <= x < 0.5:
            return (x - a) / (0.5 - a)
        elif 0.5 <= x < 1 - a:
            return (1 - a - x) / (0.5 - a)
        else:
            return (1 - x) / a

    @staticmethod
    def singer_map(x: float, a: float = 1.07) -> float:
        """Singer map"""
        return a * (7.86 * x - 23.31 * x ** 2 + 28.75 * x ** 3 - 13.302875 * x ** 4)

    @staticmethod
    def sinusoidal_map(x: float, a: float = 2.3) -> float:
        """Sinusoidal map"""
        return a * x * x * np.sin(np.pi * x)

    @staticmethod
    def tent_map(x: float) -> float:
        """Tent map"""
        if x < 0.7:
            return x / 0.7
        else:
            return (10 / 3) * (1 - x)
