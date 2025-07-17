#!/usr/bin/env python
# Created by "Thieu" at 23:20, 17/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%


class FuzzySystem:
    """Fuzzy System for hierarchical pyramid weights"""

    def __init__(self, pyramid_type='increase'):
        """
        Args:
            pyramid_type: 'increase' or 'decrease'
        """
        self.pyramid_type = pyramid_type

    def triangular_membership(self, x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    def fuzzify_iterations(self, iteration_percent):
        """Fuzzify input iterations (0-1)"""
        low = self.triangular_membership(iteration_percent, 0, 0, 0.5)
        medium = self.triangular_membership(iteration_percent, 0, 0.5, 1)
        high = self.triangular_membership(iteration_percent, 0.5, 1, 1)
        return {'low': low, 'medium': medium, 'high': high}

    def defuzzify_centroid(self, membership_values):
        """Defuzzification using centroid method"""
        # Triangular membership functions for output (0-100)
        low_center = 25
        medium_center = 50
        high_center = 75

        numerator = (membership_values['low'] * low_center +
                     membership_values['medium'] * medium_center +
                     membership_values['high'] * high_center)
        denominator = sum(membership_values.values())

        if denominator == 0:
            return 50  # Default value

        return numerator / denominator

    def get_fuzzy_weights(self, current_iteration, max_iterations):
        """Get fuzzy weights for alpha, beta, delta"""
        iteration_percent = current_iteration / max_iterations
        input_membership = self.fuzzify_iterations(iteration_percent)

        if self.pyramid_type == 'increase':
            # Rules for increase pyramid
            alpha_membership = {
                'low': input_membership['low'],
                'medium': max(input_membership['medium'], input_membership['low']),
                'high': input_membership['high']
            }
            beta_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['low'], input_membership['medium'], input_membership['high']),
                'high': 0
            }
            delta_membership = {
                'low': max(input_membership['medium'], input_membership['high']),
                'medium': input_membership['low'],
                'high': 0
            }
        else:  # decrease
            # Rules for decrease pyramid
            alpha_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['medium'], input_membership['high']),
                'high': input_membership['low']
            }
            beta_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['low'], input_membership['medium'], input_membership['high']),
                'high': 0
            }
            delta_membership = {
                'low': max(input_membership['low'], input_membership['medium']),
                'medium': input_membership['high'],
                'high': 0
            }

        alpha_weight = self.defuzzify_centroid(alpha_membership)
        beta_weight = self.defuzzify_centroid(beta_membership)
        delta_weight = self.defuzzify_centroid(delta_membership)

        return alpha_weight, beta_weight, delta_weight
