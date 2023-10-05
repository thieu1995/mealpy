#!/usr/bin/env python
# Created by "Thieu" at 05:33, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from abc import ABC, abstractmethod


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    def set_y(self, y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)
        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.
        """
        self.unique_labels = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        return self

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.

        Returns:
        --------
        encoded_labels : list
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.label_to_index[label] for label in y]

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : list, tuple
            Target values.

        Returns
        -------
        y : list
            Encoded labels.
        """
        y = self.set_y(y)
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : list, tuple
            Encoded integer labels.

        Returns:
        --------
        original_labels : list
            Original labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y]


class BaseVar(ABC):
    SUPPORTED_ARRAY = [tuple, list, np.ndarray]

    def __init__(self, n_vars=1, name="variable"):
        self._n_vars = None
        self.set_n_vars(n_vars)
        self.name = name
        self.lb, self.ub = None, None

    @property
    def n_vars(self):
        return self._n_vars

    def set_n_vars(self, n_vars):
        if type(n_vars) is int and n_vars > 0:
            self._n_vars = n_vars
        else:
            raise ValueError(f"Invalid n_vars. It should be integer and > 0.")

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def correct(self, x):
        pass

    @abstractmethod
    def generate(self):
        pass


class IntegerVar(BaseVar):
    def __init__(self, n_vars=1, lb=-10, ub=10, name="integer"):
        super().__init__(n_vars, name)
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if self.n_vars == 1:
            if type(lb) is int and type(ub) is int:
                self.lb, self.ub = (lb, ), (ub, )
            else:
                raise ValueError(f"Invalid lb or ub. It should be int value and lb < ub")
        else:
            if type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
                if len(lb) == len(ub) == self.n_vars:
                    self.lb, self.ub = np.array(lb, dtype=int), np.array(ub, dtype=int)
                else:
                    raise ValueError(f"Invalid lb or ub. Length of lb, ub should equal to n_vars.")
            else:
                raise TypeError(f"Invalid lb or ub. It should be {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return np.random.randint(self.lb, self.ub)


class FloatVar(BaseVar):
    def __init__(self, n_vars=1, lb=-10., ub=10., name="float"):
        super().__init__(n_vars, name)
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if self.n_vars == 1:
            if type(lb) in (int, float) and type(ub) in (int, float):
                self.lb, self.ub = (float(lb), ), (float(ub), )
            else:
                raise ValueError(f"Invalid lb or ub. It should be int value and lb < ub")
        else:
            if type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
                if len(lb) == len(ub) == self.n_vars:
                    self.lb, self.ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
                else:
                    raise ValueError(f"Invalid lb or ub. Length of lb, ub should equal to n_vars.")
            else:
                raise TypeError(f"Invalid lb or ub. It should be {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=float)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return np.random.uniform(self.lb, self.ub)


class PermutationVar(BaseVar):
    def __init__(self, n_vars=2, valid_set=(1, 2), name="permutation"):
        super().__init__(n_vars, name)
        self.eps = 1e-4
        self._set_bounds(valid_set)

    def _set_bounds(self, valid_set):
        if self.n_vars == 1:
            raise ValueError(f"Invalid n_vars. Permutation type needs at least 2 variables.")
        if type(valid_set) in self.SUPPORTED_ARRAY:
            if len(valid_set) == self.n_vars:
                self.valid_set = np.array(valid_set)
                self.le = LabelEncoder().fit(valid_set)
                self.lb = np.zeros(self.n_vars)
                self.ub = (self.n_vars - self.eps) * np.ones(self.n_vars)
            else:
                raise ValueError(f"Invalid valid_set. Length of valid_set should equal to n_vars.")
        else:
            raise TypeError(f"Invalid valid_set. It should be {self.SUPPORTED_ARRAY}")

    @staticmethod
    def min_max_scale(data, min_val, max_val):
        """
        Scale a NumPy array of values to a specified range [min_val, max_val].

        Parameters:
        - data (numpy.ndarray): The NumPy array of values to be scaled.
        - min_val (float): The minimum value in the desired range.
        - max_val (float): The maximum value in the desired range.

        Returns:
        - scaled_data (numpy.ndarray): The scaled NumPy array of values.
        """
        data_min, data_max = np.min(data), np.max(data)
        scaled_data = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        return scaled_data

    def encode(self, x):
        return np.array(self.le.transform(x), dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return self.le.inverse_transform(x)

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return np.random.permutation(self.valid_set)


class StringVar(BaseVar):
    def __init__(self, n_vars=1, valid_sets=(("",),), name="string"):
        super().__init__(n_vars, name)
        self.eps = 1e-4
        self._set_bounds(valid_sets)

    def _set_bounds(self, valid_sets):
        if self.n_vars == 1:
            if type(valid_sets) in self.SUPPORTED_ARRAY:
                if len(valid_sets) > 1:
                    self.valid_sets = (tuple(valid_sets), )
                    le = LabelEncoder().fit(valid_sets)
                    self.list_le = (le, )
                    self.lb = np.array([0, ])
                    self.ub = np.array([len(valid_sets), ])
                else:
                    raise ValueError(f"Invalid valid_sets. It should contains at least 2 values")
            else:
                raise ValueError(f"Invalid valid_sets. It should be {self.SUPPORTED_ARRAY} an contains at least 2 values")
        else:
            if type(valid_sets) in self.SUPPORTED_ARRAY:
                if len(valid_sets) == self.n_vars:
                    if all(len(item) > 1 for item in valid_sets):
                        self.valid_sets = valid_sets
                        self.list_le = []
                        ub = []
                        for vl_set in valid_sets:
                            le = LabelEncoder().fit(vl_set)
                            self.list_le.append(le)
                            ub.append(len(vl_set) - self.eps)
                        self.lb = np.zeros(self.n_vars)
                        self.ub = np.array(ub)
                    else:
                        raise ValueError(f"Invalid valid_sets. All string variables need to have at least 2 values.")
                else:
                    raise ValueError(f"Invalid valid_sets. Length of valid_sets should equal to n_vars.")
            else:
                raise TypeError(f"Invalid valid_sets. It should be {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array([le.transform(val)[0] for (le, val) in zip(self.list_le, x)], dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return [le.inverse_transform(val)[0] for (le, val) in zip(self.list_le, x)]

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return [np.random.choice(np.array(vl_set, dtype=object)) for vl_set in self.valid_sets]


class BinaryVar(BaseVar):
    def __init__(self, n_vars=1, name="binary"):
        super().__init__(n_vars, name)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return np.random.randint(0, 2, self.n_vars)


class BoolVar(BaseVar):
    def __init__(self, n_vars=1, name="boolean"):
        super().__init__(n_vars, name)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=int)

    def decode(self, x):
        x = self.correct(x)
        return x == 1

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return np.random.choice([True, False], self.n_vars, replace=True)