
from tkinter import W
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.swarm_based.GOA import OriginalGOA
from mealpy.swarm_based.GWO import OriginalGWO
import numpy as np

"""
refenrence article:
https://www.tandfonline.com/doi/abs/10.1080/0305215X.2018.1509963
"""


def gear_box(x):
    """_summary_
    
    b = x[0]
    d1 = x[1]
    d2 = x[2]
    Z = x[3]
    m = x[4]
    
    Args:
        x (list): _description_ list of variable of gear box problem
    """
    # input parameters
    b = x[0]
    d_1 = x[1]
    d_2 = x[2]
    Z1 = x[3]
    m = x[4]
    D_r = m * (i * Z1 * 2.5)
    l_w = 2.5
    D_i = D_r - 2 * l_w
    b_w = 3.4
    d_0 = d_2 + 25
    d_p = 0.25 * (D_i - d_0)
    D_1 = m * Z1
    D_2 = i * m * Z1
    N_2 = N_1 / i
    Z_2 = Z1 * D_2 / D_1
    v = np.pi * D_1 * N_1 / 60000
    b_1 = 1000 * P / v
    b_3 = 4.97 * pow(10, 6) * P
    F_s = np.pi * K_v * K_w * sigma * b * m * y
    F_p = 2 * K_v * K_w * D_1 * b * Z_2 / (Z1 + Z_2)

    # constraint function
    def g1(x):
        return -F_s + b_1

    def g2(x):
        return -(F_s / F_p) + b_2

    def g3(x):
        return -pow(d_1, 3) + b_3

    def g4(x):
        return pow(d_2, 3)

    def g5(x):
        return (((1 + i) * m * Z1) / 2)

    l = 1

    def violate(value):
        return 0 if value <= 0 else value

    fx = (np.pi / 4) * (rho / 1000) * (
                b * pow(m, 2) * pow(Z1, 2) * (pow(i, 2) + 1) - (pow(D_i, 2) - pow(d_0, 2)) * (l - b_w) - (n * pow(d_p, 2) * b_w) - (d_1 - d_2) * b)

    fx += violate(g1(x)) + violate(g2(x)) + violate(g3(x))
    return fx


if __name__ == "__main__":
    i = 4
    rho = 7.5
    n = 6
    sigma = 294.3
    y = 0.102
    b_2 = 0.193
    K_v = 0.389
    K_w = 0.8
    N_1 = 1500
    P = 7.5

    problem_dict1 = {
        "fit_func": gear_box,
        "lb": [20, 10, 30, 18, 2.75],
        "ub": [32, 30, 40, 25, 4],
        "minmax": "min",
    }
    model1 = OriginalWOA(epoch=1000, pop_size=500)
    best_position, best_fitness = model1.solve(problem_dict1)
    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
