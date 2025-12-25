#!/usr/bin/env python
# Created by "Thieu" at 11:40, 20/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
import numpy as np

try:
    import opfunu
    from opfunu.name_based import a_func, b_func, c_func, d_func, e_func, g_func, k_func, q_func, s_func, x_func
    OPFUNU_AVAILABLE = True
except Exception:
    OPFUNU_AVAILABLE = False
    opfunu = None
    a_func = b_func = c_func = d_func = e_func = g_func = k_func = q_func = s_func = x_func = None


@dataclass
class BWOFunction:
    name: str
    func: Callable[[np.ndarray], float]
    lb: np.ndarray
    ub: np.ndarray
    n_dims: int


def _penalty_u(x, a, k, m):
    x = np.asarray(x)
    return np.where(x > a, k * (x - a) ** m, np.where(x < -a, k * (-x - a) ** m, 0.0))


def _katsuura(x):
    x = np.asarray(x)
    n = x.size
    j = np.arange(1, 33)
    terms = []
    for i in range(n):
        frac = np.abs(2 ** j * x[i] - np.round(2 ** j * x[i])) / (2 ** j)
        terms.append(1 + (i + 1) * np.sum(frac))
    scale = 10.0 / (n ** 2)
    return scale * np.prod(np.array(terms) ** (10 / (n ** 1.2))) - scale


def _ackley(x):
    x = np.asarray(x)
    n = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def _weierstrass(x, a=0.5, b=3.0, k_max=20):
    x = np.asarray(x)
    n = x.size
    k = np.arange(0, k_max + 1)
    left = np.sum([np.sum(a ** k * np.cos(2 * np.pi * b ** k * (x_i + 0.5))) for x_i in x])
    right = n * np.sum(a ** k * np.cos(2 * np.pi * b ** k * 0.5))
    return left - right


def _griewank(x):
    x = np.asarray(x)
    idx = np.arange(1, x.size + 1)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(idx))) + 1


def _rastrigin(x):
    x = np.asarray(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def _sphere(x):
    x = np.asarray(x)
    return np.sum(x ** 2)


def _rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def _scaffer(x1, x2):
    sum_sq = x1 ** 2 + x2 ** 2
    return 0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5) / (1 + 0.001 * sum_sq) ** 2


def _expanded_scaffer(x):
    x = np.asarray(x)
    total = 0.0
    for i in range(x.size - 1):
        total += _scaffer(x[i], x[i + 1])
    total += _scaffer(x[-1], x[0])
    return total


def _griewank_rosenbrock(x):
    z = np.asarray(x)
    tmp = 100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
    f = np.sum(tmp ** 2 / 4000 - np.cos(tmp) + 1)
    tmp = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1.0) ** 2
    f += tmp ** 2 / 4000 - np.cos(tmp) + 1
    return f


def _modified_schwefel(x):
    x = np.asarray(x)
    n = x.size
    y = x + 420.9687462275036
    z = np.zeros_like(y)
    mask_high = y > 500
    mask_low = y < -500
    mask_mid = ~(mask_high | mask_low)
    if np.any(mask_high):
        y_high = y[mask_high]
        z[mask_high] = (500 - np.mod(y_high, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(y_high, 500)))) - (
            (y_high - 500) ** 2) / (10000 * n)
    if np.any(mask_low):
        y_low = y[mask_low]
        z[mask_low] = (np.mod(np.abs(y_low), 500) - 500) * np.sin(
            np.sqrt(np.abs(np.mod(np.abs(y_low), 500) - 500))) - ((y_low + 500) ** 2) / (10000 * n)
    if np.any(mask_mid):
        z[mask_mid] = y[mask_mid] * np.sin(np.sqrt(np.abs(y[mask_mid])))
    return 418.9829 * n - np.sum(z)


def _penalized_1(x):
    x = np.asarray(x)
    n = x.size
    y = 1 + (x + 1) / 4
    term = np.sin(np.pi * y[0]) ** 2
    term += np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
    term += (y[-1] - 1) ** 2
    return (np.pi / n) * term + np.sum(_penalty_u(x, 10, 100, 4))


def _penalized_2(x):
    x = np.asarray(x)
    term = np.sin(3 * np.pi * x[0]) ** 2
    term += np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
    term += (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    return 0.1 * term + np.sum(_penalty_u(x, 5, 100, 4))


def _quartic(x):
    x = np.asarray(x)
    i = np.arange(1, x.size + 1)
    return np.sum(i * x ** 4) + np.random.rand()


def _schwefel_12(x):
    x = np.asarray(x)
    return np.sum([np.sum(x[:idx + 1]) ** 2 for idx in range(x.size)])


def _schwefel_21(x):
    x = np.asarray(x)
    return np.max(np.abs(x))


def _schwefel_22(x):
    x = np.asarray(x)
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def _step_2(x):
    x = np.asarray(x)
    return np.sum((x + 0.5) ** 2)


def _alpine_1(x):
    x = np.asarray(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


def _csendes(x):
    x = np.asarray(x)
    safe = np.where(x == 0, 1.0, x)
    return np.sum(x ** 6 * (2 + np.sin(1 / safe)))


def _rotated_ellipse_1(x):
    x1, x2 = x[0], x[1]
    return 7 * x1 ** 2 - 6 * np.sqrt(3) * x1 * x2 + 13 * x2 ** 2


def _rotated_ellipse_2(x):
    x1, x2 = x[0], x[1]
    return x1 ** 2 - x1 * x2 + x2 ** 2


def _schwefel_24(x):
    x = np.asarray(x)
    x1 = x[0]
    return np.sum((x - 1) ** 2 + (x1 - x ** 2) ** 2)


def _sum_squares(x):
    x = np.asarray(x)
    i = np.arange(1, x.size + 1)
    return np.sum(i * x ** 2)


def _step(x):
    x = np.asarray(x)
    return np.sum(np.floor(np.abs(x)))


def _schwefel_26(x):
    x = np.asarray(x)
    return np.sum(418.9829 - x * np.sin(np.sqrt(np.abs(x))))


def _schaffer(x):
    x1, x2 = x[0], x[1]
    num = np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5
    den = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return 0.5 + num / den


def _sum_abs(x):
    x = np.asarray(x)
    return np.sum(np.abs(x))


def _sum_floor(x):
    x = np.asarray(x)
    return np.sum(np.floor(x))


def _adjiman(x):
    x1, x2 = x[0], x[1]
    return np.cos(x1) * np.sin(x2) - x1 / (x2 ** 2 + 1)


def _bartels_conn(x):
    x1, x2 = x[0], x[1]
    return np.abs(x1 ** 2 + x2 ** 2 + x1 * x2) + np.abs(np.sin(x1)) + np.abs(np.cos(x2))


def _ackley_02(x):
    x1, x2 = x[0], x[1]
    return -200 * np.exp(-0.02 * np.sqrt(x1 ** 2 + x2 ** 2))


def _eggcrate(x):
    x1, x2 = x[0], x[1]
    return x1 ** 2 + x2 ** 2 + 25 * (np.sin(x1) ** 2 + np.cos(x2) ** 2)


def _f40(x):
    x1, x2 = x[0], x[1]
    return x1 * np.sin(4 * x1) + 1.1 * x2 * np.sin(2 * x2)


def _powell_singular_2(x):
    x = np.asarray(x)
    total = 0.0
    for i in range(0, x.size - 3):
        total += (x[i] + 10 * x[i + 1]) ** 2
        total += 5 * (x[i + 2] - x[i + 3]) ** 2
        total += (x[i + 1] - 2 * x[i + 2]) ** 4
        total += 10 * (x[i] - x[i + 3]) ** 4
    return total


def _quintic(x):
    x = np.asarray(x)
    return np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4))


def _qing(x):
    x = np.asarray(x)
    i = np.arange(1, x.size + 1)
    return np.sum((x ** 2 - i) ** 2)


def _salomon(x):
    x = np.asarray(x)
    r = np.sqrt(np.sum(x ** 2))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r


def _dixon_price(x):
    x = np.asarray(x)
    term = (x[0] - 1) ** 2
    i = np.arange(2, x.size + 1)
    term += np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2)
    return term


def _make_bounds(n_dims, low, high):
    lb = np.full(n_dims, low, dtype=float)
    ub = np.full(n_dims, high, dtype=float)
    return lb, ub


def _load_cec2005_shifts(n_funcs, n_dims, data_name="data_hybrid_func1"):
    if not OPFUNU_AVAILABLE:
        return np.zeros((n_funcs, n_dims), dtype=float)
    try:
        data_path = Path(opfunu.__file__).resolve().parent / "cec_based" / "data_2005" / f"{data_name}.txt"
        data = np.loadtxt(data_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] < n_funcs:
            return np.zeros((n_funcs, n_dims), dtype=float)
        return data[:n_funcs, :n_dims]
    except Exception:
        return np.zeros((n_funcs, n_dims), dtype=float)


def _compose_cec2005(x, funcs, shifts, deltas, lambdas, c=2000.0, bias=None):
    x = np.asarray(x, dtype=float)
    n_dims = x.size
    n_funcs = len(funcs)
    shifts = np.asarray(shifts, dtype=float)
    if shifts.shape[0] < n_funcs or shifts.shape[1] < n_dims:
        shifts = np.zeros((n_funcs, n_dims), dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if bias is None:
        bias = np.zeros(n_funcs, dtype=float)
    else:
        bias = np.asarray(bias, dtype=float)

    y = 5.0 * np.ones(n_dims, dtype=float)
    weights = np.zeros(n_funcs, dtype=float)
    fits = np.zeros(n_funcs, dtype=float)
    for idx in range(n_funcs):
        diff = x - shifts[idx, :n_dims]
        denom = 2.0 * n_dims * (deltas[idx] ** 2)
        weights[idx] = np.exp(-np.sum(diff ** 2) / denom) if denom > 0 else 1.0
        z = diff / lambdas[idx]
        fit = funcs[idx](z)
        f_max = funcs[idx](y / lambdas[idx])
        if f_max == 0:
            f_max = 1.0
        fits[idx] = c * fit / f_max

    maxw = np.max(weights)
    weights = np.where(weights != maxw, weights * (1 - maxw ** 10), weights)
    w_sum = np.sum(weights)
    if w_sum == 0:
        weights = np.full(n_funcs, 1.0 / n_funcs)
    else:
        weights = weights / w_sum
    return float(np.sum(weights * (fits + bias)))


def _make_composite_func(base_funcs, deltas, lambdas, shifts):
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    shifts = np.asarray(shifts, dtype=float)

    def _func(x):
        return _compose_cec2005(x, base_funcs, shifts, deltas, lambdas)

    return _func


def _opfunu_instance(cls, n_dims):
    if not OPFUNU_AVAILABLE or cls is None:
        return None
    try:
        return cls() if n_dims is None else cls(ndim=n_dims)
    except Exception:
        return None


def get_bwo_functions(n_dims: int = 30, include_composites: bool = False) -> List[BWOFunction]:
    funcs: List[BWOFunction] = []

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F1: Powell Sum", lambda x: np.sum(np.abs(x) ** (np.arange(1, x.size + 1) + 1)), lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    op = _opfunu_instance(getattr(c_func, "Cigar", None), n_dims)
    funcs.append(BWOFunction("F2: Cigar", op.evaluate if op else lambda x: x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2), lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F3: Discus", lambda x: 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2), lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -30, 30)
    funcs.append(BWOFunction("F4: Rosenbrock", _rosenbrock, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -35, 35)
    op = _opfunu_instance(getattr(a_func, "Ackley01", None), n_dims)
    funcs.append(BWOFunction("F5: Ackley", op.evaluate if op else _ackley, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    funcs.append(BWOFunction("F6: Weierstrass", _weierstrass, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    op = _opfunu_instance(getattr(g_func, "Griewank", None), n_dims)
    funcs.append(BWOFunction("F7: Griewank", op.evaluate if op else _griewank, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F8: Rastrigin", _rastrigin, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    funcs.append(BWOFunction("F9: Modified Schwefel", _modified_schwefel, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, 0, 10)
    funcs.append(BWOFunction("F10: Katsuura", _katsuura, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F11: HappyCat", lambda x: np.abs(np.sum(x ** 2) - x.size) ** 0.25 +
                              (0.5 * np.sum(x ** 2) + np.sum(x)) / x.size + 0.5, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F12: HGBat", lambda x: np.abs(np.sum(x ** 2) ** 2 - np.sum(x) ** 2) ** 0.5 +
                              (0.5 * np.sum(x ** 2) + np.sum(x)) / x.size + 0.5, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F13: Expanded Griewank plus Rosenbrock", _griewank_rosenbrock, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F14: Expanded Scaffer F6", _expanded_scaffer, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -np.pi, np.pi)
    funcs.append(BWOFunction("F15: Some of different powers", lambda x, k=1.0: 1 - (1 / x.size) *
                              np.sum(np.cos(k * x) * np.exp(-(x ** 2) / 2)), lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F16: Sphere", _sphere, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -50, 50)
    funcs.append(BWOFunction("F17: Penalized", _penalized_1, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -50, 50)
    funcs.append(BWOFunction("F18: Penalized2", _penalized_2, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -1.28, 1.28)
    op = _opfunu_instance(getattr(q_func, "Quartic", None), n_dims)
    funcs.append(BWOFunction("F19: Quartic", op.evaluate if op else _quartic, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    funcs.append(BWOFunction("F20: Schwefel 1.2", _schwefel_12, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    funcs.append(BWOFunction("F21: Schwefel 2.21", _schwefel_21, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    funcs.append(BWOFunction("F22: Schwefel 2.22", _schwefel_22, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -200, 200)
    funcs.append(BWOFunction("F23: Step 2", _step_2, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    op = _opfunu_instance(getattr(c_func, "Alpine01", None), n_dims)
    funcs.append(BWOFunction("F24: Alpine1", op.evaluate if op else _alpine_1, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -1, 1)
    op = _opfunu_instance(getattr(c_func, "Csendes", None), n_dims)
    funcs.append(BWOFunction("F25: Csendes", op.evaluate if op else _csendes, lb, ub, n_dims))

    lb, ub = _make_bounds(2, -500, 500)
    funcs.append(BWOFunction("F26: Rotated Ellipse", _rotated_ellipse_1, lb, ub, 2))

    lb, ub = _make_bounds(2, -500, 500)
    funcs.append(BWOFunction("F27: Rotated Ellipse2", _rotated_ellipse_2, lb, ub, 2))

    lb, ub = _make_bounds(n_dims, 0, 10)
    funcs.append(BWOFunction("F28: Schwefel 2.4", _schwefel_24, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    funcs.append(BWOFunction("F29: Sum Squares", _sum_squares, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    funcs.append(BWOFunction("F30: Step", _step, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -500, 500)
    funcs.append(BWOFunction("F31: Schwefel", _schwefel_26, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5, 5)
    op = _opfunu_instance(getattr(x_func, "XinSheYang01", None), n_dims)
    funcs.append(BWOFunction("F32: Xin-She Yang1", op.evaluate if op else lambda x: np.sum(np.random.rand(x.size) * np.abs(x) ** np.arange(1, x.size + 1)), lb, ub, n_dims))

    lb, ub = _make_bounds(2, -100, 100)
    funcs.append(BWOFunction("F33: Schaffer", _schaffer, lb, ub, 2))

    lb, ub = _make_bounds(n_dims, -1.0, 2.0)
    funcs.append(BWOFunction("F34: Absolute Value", _sum_abs, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -5.12, 5.12)
    funcs.append(BWOFunction("F35: Floor Sum", _sum_floor, lb, ub, n_dims))

    lb = np.array([-1, -1], dtype=float)
    ub = np.array([2, 1], dtype=float)
    op = _opfunu_instance(getattr(a_func, "Adjiman", None), None)
    funcs.append(BWOFunction("F36: Adjiman", op.evaluate if op else _adjiman, lb, ub, 2))

    lb, ub = _make_bounds(2, -500, 500)
    op = _opfunu_instance(getattr(b_func, "BartelsConn", None), None)
    funcs.append(BWOFunction("F37: Bartels Conn", op.evaluate if op else _bartels_conn, lb, ub, 2))

    lb, ub = _make_bounds(2, -500, 500)
    funcs.append(BWOFunction("F38: Ackley 2", _ackley_02, lb, ub, 2))

    lb, ub = _make_bounds(2, -2 * np.pi, 2 * np.pi)
    funcs.append(BWOFunction("F39: Eggcrate", _eggcrate, lb, ub, 2))

    lb, ub = _make_bounds(2, 0, 10)
    funcs.append(BWOFunction("F40: XSinY", _f40, lb, ub, 2))

    lb, ub = _make_bounds(n_dims, -4, 5)
    funcs.append(BWOFunction("F41: Powell Singular 2", _powell_singular_2, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    op = _opfunu_instance(getattr(q_func, "Quintic", None), n_dims)
    funcs.append(BWOFunction("F42: Quintic", op.evaluate if op else _quintic, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -500, 500)
    op = _opfunu_instance(getattr(q_func, "Qing", None), n_dims)
    funcs.append(BWOFunction("F43: Qing", op.evaluate if op else _qing, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -100, 100)
    op = _opfunu_instance(getattr(s_func, "Salomon", None), n_dims)
    funcs.append(BWOFunction("F44: Salomon", op.evaluate if op else _salomon, lb, ub, n_dims))

    lb, ub = _make_bounds(n_dims, -10, 10)
    op = _opfunu_instance(getattr(d_func, "DixonPrice", None), n_dims)
    funcs.append(BWOFunction("F45: Dixon & Price", op.evaluate if op else _dixon_price, lb, ub, n_dims))

    if include_composites:
        c_dims = 10
        c_lb, c_ub = _make_bounds(c_dims, -5, 5)
        shifts = _load_cec2005_shifts(10, c_dims)

        base_cf1 = [_sphere] * 10
        deltas_cf1 = [1] * 10
        lambdas_cf1 = [5 / 100] * 10
        funcs.append(BWOFunction("F46: Composite CF1", _make_composite_func(base_cf1, deltas_cf1, lambdas_cf1, shifts),
                                 c_lb, c_ub, c_dims))

        base_cf2 = [_griewank] * 10
        deltas_cf2 = [1] * 10
        lambdas_cf2 = [5 / 100] * 10
        funcs.append(BWOFunction("F47: Composite CF2", _make_composite_func(base_cf2, deltas_cf2, lambdas_cf2, shifts),
                                 c_lb, c_ub, c_dims))

        base_cf3 = [_griewank] * 10
        deltas_cf3 = [1] * 10
        lambdas_cf3 = [1] * 10
        funcs.append(BWOFunction("F48: Composite CF3", _make_composite_func(base_cf3, deltas_cf3, lambdas_cf3, shifts),
                                 c_lb, c_ub, c_dims))

        base_cf4 = [_ackley, _ackley, _rastrigin, _rastrigin, _weierstrass, _weierstrass,
                    _griewank, _griewank, _sphere, _sphere]
        deltas_cf4 = [1] * 10
        lambdas_cf4 = [5 / 32, 5 / 32, 1, 1, 5 / 0.5, 5 / 0.5, 5 / 100, 5 / 100, 5 / 100, 5 / 100]
        funcs.append(BWOFunction("F49: Composite CF4", _make_composite_func(base_cf4, deltas_cf4, lambdas_cf4, shifts),
                                 c_lb, c_ub, c_dims))

        base_cf5 = [_rastrigin, _rastrigin, _weierstrass, _weierstrass, _griewank, _griewank,
                    _ackley, _ackley, _sphere, _sphere]
        deltas_cf5 = [1] * 10
        lambdas_cf5 = [1 / 5, 1 / 5, 5 / 0.5, 5 / 0.5, 5 / 100, 5 / 100, 5 / 32, 5 / 32, 5 / 100, 5 / 100]
        funcs.append(BWOFunction("F50: Composite CF5", _make_composite_func(base_cf5, deltas_cf5, lambdas_cf5, shifts),
                                 c_lb, c_ub, c_dims))

        base_cf6 = base_cf5
        deltas_cf6 = [0.1 * (i + 1) for i in range(10)]
        lambdas_cf6 = [0.1 * 1 / 5, 0.2 * 1 / 5, 0.3 * 5 / 0.5, 0.4 * 5 / 0.5, 0.5 * 5 / 100,
                       0.6 * 5 / 100, 0.7 * 5 / 32, 0.8 * 5 / 32, 0.9 * 5 / 100, 1 * 5 / 100]
        funcs.append(BWOFunction("F51: Composite CF6", _make_composite_func(base_cf6, deltas_cf6, lambdas_cf6, shifts),
                                 c_lb, c_ub, c_dims))

    return funcs
