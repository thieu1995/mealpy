#!/usr/bin/env python
# Created by "Thieu" at 05:33, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import numbers as nb
from abc import ABC, abstractmethod
from mealpy.utils import transfer


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

    def __init__(self, name="variable"):
        self.name = name
        self.n_vars = None
        self.lb, self.ub = None, None
        self._seed = None
        self.generator = np.random.default_rng()

    def set_n_vars(self, n_vars):
        if type(n_vars) is int and n_vars > 0:
            self.n_vars = n_vars
        else:
            raise ValueError(f"Invalid n_vars. It should be integer and > 0.")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value
        self.generator = np.random.default_rng(self._seed)

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

    @staticmethod
    def round(x):
        frac = x - np.floor(x)
        t1 = np.floor(x)
        t2 = np.ceil(x)
        return np.where(frac < 0.5, t1, t2)


class FloatVar(BaseVar):
    def __init__(self, lb=-10., ub=10., name="float"):
        super().__init__(name)
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            self.lb, self.ub = np.array((lb, ), dtype=float), np.array((ub,), dtype=float)
            self.n_vars = 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                self.lb, self.ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
                self.n_vars = len(lb)
            else:
                raise ValueError(f"Invalid lb or ub. Length of lb should equal to length of ub.")
        else:
            raise TypeError(f"Invalid lb or ub. It should be one of following: {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=float)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.uniform(self.lb, self.ub)


class IntegerVar(BaseVar):
    def __init__(self, lb=-10, ub=10, name="integer"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            lb, ub = int(lb) - 0.5, int(ub) + 0.5 - self.eps
            self.lb, self.ub = np.array((lb, ), dtype=float), np.array((ub, ), dtype=float)
            self.n_vars = 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                self.lb, self.ub = np.array(lb, dtype=float) - 0.5, np.array(ub, dtype=float) + (0.5 - self.eps)
                self.n_vars = len(lb)
            else:
                raise ValueError(f"Invalid lb or ub. Length of lb should equal to length of ub.")
        else:
            raise TypeError(f"Invalid lb or ub. It should be one of following: {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = self.round(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(self.lb+0.5, self.ub+0.5+self.eps)


class PermutationVar(BaseVar):
    def __init__(self, valid_set=(1, 2), name="permutation"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(valid_set)

    def _set_bounds(self, valid_set):
        if type(valid_set) in self.SUPPORTED_ARRAY and len(valid_set) > 1:
            self.valid_set = np.array(valid_set)
            self.n_vars = len(valid_set)
            self.le = LabelEncoder().fit(valid_set)
            self.lb = np.zeros(self.n_vars)
            self.ub = (self.n_vars - self.eps) * np.ones(self.n_vars)
        else:
            raise TypeError(f"Invalid valid_set. It should be {self.SUPPORTED_ARRAY} and contains at least 2 variables")

    def encode(self, x):
        return np.array(self.le.transform(x), dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return self.le.inverse_transform(x)

    def correct(self, x):
        return np.argsort(x)

    def generate(self):
        return self.generator.permutation(self.valid_set)


class StringVar(BaseVar):
    def __init__(self, valid_sets=(("",),), name="string"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(valid_sets)

    def _set_bounds(self, valid_sets):
        if type(valid_sets) in self.SUPPORTED_ARRAY:
            if type(valid_sets[0]) not in self.SUPPORTED_ARRAY:
                self.n_vars = 1
                self.valid_sets = (tuple(valid_sets),)
                le = LabelEncoder().fit(valid_sets)
                self.list_le = (le,)
                self.lb = np.array([0, ])
                self.ub = np.array([len(valid_sets) - self.eps, ])
            else:
                self.n_vars = len(valid_sets)
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
                    raise ValueError(f"Invalid valid_sets. All variables need to have at least 2 values.")
        else:
            raise TypeError(f"Invalid valid_sets. It should be {self.SUPPORTED_ARRAY}.")

    def encode(self, x):
        return np.array([le.transform(val)[0] for (le, val) in zip(self.list_le, x)], dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return [le.inverse_transform(val)[0] for (le, val) in zip(self.list_le, x)]

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return [self.generator.choice(np.array(vl_set, dtype=str)) for vl_set in self.valid_sets]


class MixedSetVar(StringVar):
    def __init__(self, valid_sets=(("",),), name="mixed-set-var"):
        super().__init__(valid_sets, name)
        self.eps = 1e-4
        self._set_bounds(valid_sets)

    def generate(self):
        return [self.generator.choice(np.array(vl_set, dtype=object)) for vl_set in self.valid_sets]


class BinaryVar(BaseVar):
    def __init__(self, n_vars=1, name="binary"):
        super().__init__(name)
        self.set_n_vars(n_vars)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(0, 2, self.n_vars)


class TransferBinaryVar(BinaryVar):

    SUPPORTED_TF_FUNCS = ["vstf_01", "vstf_02", "vstf_03", "vstf_04", "sstf_01", "sstf_02", "sstf_03", "sstf_04"]

    def __init__(self, n_vars=1, name="tf-binary", tf_func="vstf_01", lb=-8., ub=8., all_zeros=True):
        super().__init__(n_vars, name)
        if tf_func in self.SUPPORTED_TF_FUNCS:
            self.tf_name = tf_func
            self.tf_func = getattr(transfer, tf_func)
        else:
            raise ValueError(f"Invalid transfer function! The supported TF funcs are: {self.SUPPORTED_TF_FUNCS}")
        self.lb = lb * np.ones(self.n_vars)
        self.ub = ub * np.ones(self.n_vars)
        self.all_zeros = all_zeros

    def _get_correct_x(self, x):
        if self.all_zeros:
            return x
        else:
            if np.sum(x) == 0:
                x[self.generator.integers(0, len(x))] = 1
            return x

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        x = self.tf_func(x)
        cons = self.generator.random(len(x))
        x = np.where(cons < x, 1, 0)
        return self._get_correct_x(x)

    def generate(self):
        x = self.generator.integers(0, 2, self.n_vars)
        return self._get_correct_x(x)


class BoolVar(BaseVar):
    def __init__(self, n_vars=1, name="boolean"):
        super().__init__(name)
        self.set_n_vars(n_vars)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = np.array(x, dtype=int)
        return x == 1

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.choice([True, False], self.n_vars, replace=True)


class TransferBoolVar(BoolVar):

    SUPPORTED_TF_FUNCS = ["vstf_01", "vstf_02", "vstf_03", "vstf_04", "sstf_01", "sstf_02", "sstf_03", "sstf_04"]

    def __init__(self, n_vars=1, name="boolean", tf_func="vstf_01", lb=-8., ub=8.):
        super().__init__(n_vars, name)
        if tf_func in self.SUPPORTED_TF_FUNCS:
            self.tf_name = tf_func
            self.tf_func = getattr(transfer, tf_func)
        else:
            raise ValueError(f"Invalid transfer function! The supported TF funcs are: {self.SUPPORTED_TF_FUNCS}")
        self.lb = lb * np.ones(self.n_vars)
        self.ub = ub * np.ones(self.n_vars)

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        x = self.tf_func(x)
        cons = self.generator.random(len(x))
        return np.where(cons < x, 1, 0)
