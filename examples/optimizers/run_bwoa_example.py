#!/usr/bin/env python
# Created by "Thieu" at 18:37, 27/10/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import (FloatVar, BWOA, APO, GRSA, KLA, MGOA, AAA, NWOA, OSA, DandelionO, RFO, CrayfishOA, SPBO,
                    CCO, AHO, MSA, TSeedA, SBOA, ChameleonSA, WSO, FFA)


def objective_function(solution):
    return np.sum(solution ** 2)


problem = {
    "bounds": FloatVar(lb=(-10.0,) * 30, ub=(10.0,) * 30, name="x"),
    "minmax": "min",
    "obj_func": objective_function,
    "name": "Sphere",
}

model = BWOA.OriginalBWOA(epoch=100, pop_size=50, pp=0.6, cr=0.44, pm=0.4)
model = APO.OriginalAPO(epoch=100, pop_size=50, pf_max=0.1, n_pairs=2)
model = GRSA.OriginalGRSA(epoch=100, pop_size=50, n_geometry=5, w_max=0.9, w_min=0.4, g_max=0.5, g_min=0.1)
model = KLA.OriginalKLA(epoch=100, pop_size=50)
model = MGOA.OriginalMGOA(epoch=200, pop_size=50, attract_dim_rate=0.2)
model = AAA.OriginalAAA(epoch=1000, pop_size=50, s_force=2.0, e_loss=0.3, ap=0.5)
model = NWOA.OriginalNWOA(epoch=1000, pop_size=50, amplitude=1.0, delta_decay=0.01, lamda_decay=0.001)
model = NWOA.OriginalNO(epoch=1000, pop_size=50, alpha=2.0, sigma0=2.0)
model = OSA.OriginalOSA(epoch=1000, pop_size=50, alpha_max = 0.5, beta_max = 1.9)
model = DandelionO.OriginalDandelionO(epoch=1000, pop_size=50)
model = DandelionO.DevDandelionO(epoch=1000, pop_size=50)
model = RFO.OriginalRFO(epoch=1000, pop_size=50, phi_0=0.7, theta=0.5)
model = CrayfishOA.OriginalCrayfishOA(epoch=1000, pop_size=50)
model = SPBO.OriginalSPBO(epoch=1000, pop_size=50)
model = CCO.OriginalCCO(epoch=1000, pop_size=50, alpha=0.5, beta=1.0)
model = AHO.OriginalAHO(epoch=100000, pop_size=50, theta=0.26, omega=0.01)
model = MSA.OriginalMSA(epoch=1000, pop_size=50, n_best = 5, partition = 0.5, max_step_size = 1.0)
model = TSeedA.OriginalTSeedA(epoch=1000, pop_size=50, st=0.1)
model = SBOA.OriginalSBOA(epoch=1000, pop_size=50)
model = ChameleonSA.OriginalChameleonSA(epoch=1000, pop_size=50, pp=0.2, p1=0.5, p2=2.5, c1=1.4, c2=1.6, gama=1.0, alpha=5.0, rho=1.5)
model = ChameleonSA.IChameleonSA(epoch=1000, pop_size=50, r_chaos=0.5, k_spiral=10., p1=5.0, p2=3.0)
model = WSO.OriginalWSO(epoch=1000, pop_size=50, tau=4.2, p_min=0.5, p_max=2.0, f_min=0.1, f_max=0.8, a0=6, a1=100, a2=0.001)
model = FFA.MLFA_GD(epoch=1000, pop_size=50, m_females=3, beta0=1.0, gama=1.0, alpha=0.2, k_rw=10)


g_best = model.solve(problem, seed=10)
print(f"Best fitness: {g_best.target.fitness}")
