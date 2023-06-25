#!/usr/bin/env python
# Created by "Thieu" at 16:53, 20/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec2017 import F292017
from mealpy.swarm_based import WOA
from mealpy.bio_based.BMO import OriginalBMO
from mealpy.bio_based.TPO import OriginalTPO
from mealpy.swarm_based.EHO import OriginalEHO
from mealpy.swarm_based.ESOA import OriginalESOA

from mealpy import BBO as T1
from mealpy.bio_based import BBO as T2
from mealpy.bio_based.BBO import BaseBBO
from mealpy.swarm_based.ARO import LARO, OriginalARO
from mealpy.swarm_based.AGTO import MGTO
from mealpy import EOA, SBO, SMA, SOA, MA, BRO, BSO, CHIO, FBIO, HBO, QSA, SARO, TLO
from mealpy import PSS, ASO, EO, FLA, BFO, GJO, GTO, HHO, MPA, SeaHO, SRSR, AVOA, SA, BSO
from mealpy import GWO, SCSO, TS

# from mealpy.utils.problem import Problem
from mealpy import Problem
from mealpy import get_all_optimizers, get_optimizer_by_name


ndim = 30
f18 = F292017(ndim, f_bias=0)

def fitness(solution):
    # time.sleep(5)
    fit = f18.evaluate(solution)
    return fit

# print(type(fitness))

problem_dict1 = {
    "fit_func": fitness,
    "lb": f18.lb.tolist(),
    "ub": f18.ub.tolist(),
    "minmax": "min",
}

term_dict1 = {
    "max_epoch": 1000,
    "max_fe": 180000,  # 100000 number of function evaluation
    "max_time": 1000,  # 10 seconds to run the program
    "max_early_stop": 150  # 15 epochs if the best fitness is not getting better we stop the program
}

epoch = 1000
pop_size = 50

class Squared(Problem):
    def __init__(self, lb, ub, minmax, name="Squared", **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def fit_func(self, solution):
        return np.sum(solution ** 2)


P1 = Squared(lb=[-10, ] * 100, ub=[10, ] * 100, minmax="min")

if __name__ == "__main__":
    # model = WOA.OriginalWOA(epoch, pop_size)
    # model = OriginalBMO(epoch, pop_size)
    # model = OriginalTPO(epoch, pop_size)
    # model = OriginalEHO(epoch, pop_size)
    # model = OriginalESOA(epoch, pop_size)
    # model = T1.BaseBBO(epoch, pop_size)
    # model = T2.OriginalBBO(epoch, pop_size)
    # model = BaseBBO(epoch, pop_size)
    # model = LARO(epoch, pop_size)
    # model = OriginalARO(epoch, pop_size)
    # model = MGTO(epoch, pop_size)
    # model = EOA.OriginalEOA(epoch, pop_size)
    # model = SBO.OriginalSBO(epoch, pop_size)
    # model = SMA.OriginalSMA(epoch, pop_size)
    # model = SOA.DevSOA(epoch, pop_size)
    # model = MA.OriginalMA(epoch, pop_size)
    # model = BRO.BaseBRO(epoch, pop_size)
    # model = BSO.ImprovedBSO(epoch, pop_size)
    # model = CHIO.BaseCHIO(epoch, pop_size)
    # model = FBIO.OriginalFBIO(epoch, pop_size)
    # model = HBO.OriginalHBO(epoch, pop_size)
    # model = QSA.BaseQSA(epoch, pop_size)
    # model = QSA.OriginalQSA(epoch, pop_size)
    # model = QSA.OppoQSA(epoch, pop_size)
    # model = QSA.ImprovedQSA(epoch, pop_size)
    # model = SARO.BaseSARO(epoch, pop_size)
    # model = SARO.OriginalSARO(epoch, pop_size)
    # model = TLO.BaseTLO(epoch, pop_size)
    # model = TLO.ImprovedTLO(epoch, pop_size)
    # model = TLO.OriginalTLO(epoch, pop_size)
    # model = PSS.OriginalPSS(epoch, pop_size)
    # model = ASO.OriginalASO(epoch, pop_size)
    # model = EO.ModifiedEO(epoch, pop_size)
    # model = EO.AdaptiveEO(epoch, pop_size)
    # model = EO.OriginalEO(epoch, pop_size)
    # model = FLA.OriginalFLA(epoch, pop_size)
    # model = BFO.OriginalBFO(epoch, pop_size)
    # model = BFO.ABFO(epoch, pop_size)
    # model = GJO.OriginalGJO(epoch, pop_size)
    # model = GTO.Matlab102GTO(epoch, pop_size)
    # model = HHO.OriginalHHO(epoch, pop_size)
    # model = MPA.OriginalMPA(epoch, pop_size)
    # model = SeaHO.OriginalSeaHO(epoch, pop_size)
    # model = SRSR.OriginalSRSR(epoch, pop_size)
    # model = AVOA.OriginalAVOA(epoch, pop_size)
    # model = SA.OriginalSA(epoch, pop_size)
    # model = BSO.OriginalBSO(epoch, pop_size)
    # model = BSO.ImprovedBSO(epoch, pop_size)
    # model = SCSO.OriginalSCSO(epoch, pop_size)
    # model = TS.OriginalTS(epoch, pop_size=2, tabu_size=5, neighbour_size=20, perturbation_scale=0.05)
    # model = GWO.OriginalGWO(epoch, pop_size)
    # model = GWO.GWO_WOA(epoch, pop_size)
    # model = GWO.RW_GWO(epoch, pop_size)

    ## 1st way
    # model = GWO.IGWO(epoch, pop_size, a_min=0.02, a_max=1.6)

    # for opt_name, opt_class in get_all_optimizers().items():
    #     print(f"{opt_name}: {opt_class}")

    ## 2nd way
    model = get_optimizer_by_name("IGWO")(epoch, pop_size, a_min=0.02, a_max=1.6)
    model = get_optimizer_by_name("OriginalHC")(epoch, pop_size=2)
    model = get_optimizer_by_name("GaussianSA")(epoch, pop_size=50, temp_init=100)
    model = get_optimizer_by_name("SwarmSA")(epoch, pop_size=50, temp_init=100)
    model = get_optimizer_by_name("OriginalSA")(epoch, pop_size=50, temp_init=100)
    best_position, best_fitness = model.solve(P1, mode="thread", n_workers=4, termination=term_dict1)

    print(best_position)
    print(model.get_parameters())
    print(model.get_attributes()["epoch"])
