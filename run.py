#!/usr/bin/env python
# Created by "Thieu" at 16:53, 20/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec2017 import F292017

from mealpy import BBO, PSO, GA, ALO, AO, ARO, AVOA, BA, BBOA, BMO, EOA, IWO
from mealpy import SBO, SMA, SOA, SOS, TPO, TSA, VCS, WHO, AOA, CEM, CGO, CircleSA, GBO, HC, INFO, PSS, RUN, SCA
from mealpy import SHIO, TS, HS, AEO, GCO, WCA, CRO, DE, EP, ES, FPA, MA, SHADE, BRO, BSO, CA, CHIO, FBIO, GSKA, HBO
from mealpy import HCO, ICA, LCO, WarSO, TOA, TLO, SSDO, SPBO, SARO, QSA, ArchOA, ASO, CDO, EFO, EO, EVO, FLA
from mealpy import HGSO, MVO, NRO, RIME, SA, WDO, TWO, ABC, ACOR, AGTO, BeesA, BES, BFO, ZOA, WOA, WaOA, TSO
from mealpy import TDO, STO, SSpiderO, SSpiderA, SSO, SSA, SRSR, SLO, SHO, SFO, ServalOA, SeaHO, SCSO, POA
from mealpy import PFA, OOA, NGO, NMRA, MSA, MRFO, MPA, MGO, MFO, JA, HHO, HGS, HBA, GWO, GTO, GOA
from mealpy import GJO, FOX, FOA, FFO, FFA, FA, ESOA, EHO, DO, DMOA, CSO, CSA, CoatiOA, COA, BSA
from mealpy import StringVar, FloatVar, BoolVar, PermutationVar, MixedSetVar, IntegerVar, BinaryVar
from mealpy import Tuner, Multitask, Problem, Optimizer, Termination, ParameterGrid
from mealpy import get_all_optimizers, get_optimizer_by_name


ndim = 30
f18 = F292017(ndim, f_bias=0)

def objective_function(solution):
    # time.sleep(5)
    fit = f18.evaluate(solution)
    return fit

# print(type(objective_function))

problem_dict1 = {
    "obj_func": f18.evaluate,
    "bounds": FloatVar(lb=f18.lb, ub=f18.ub),
    "minmax": "min",
}

term_dict1 = {
    "max_epoch": 1000,
    "max_fe": 180000,  # 100000 number of function evaluation
    "max_time": 1000,  # 10 seconds to run the program
    "max_early_stop": 150  # 15 epochs if the best objective_function is not getting better we stop the program
}

epoch = 1000
pop_size = 50

class SquaredProblem(Problem):
    def __init__(self, bounds, minmax, name="Squared", **kwargs):
        self.name = name
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        return np.sum(solution ** 2)

bounds = FloatVar(lb=(-15., )*100, ub=(20., )*100, name="variable")
P1 = SquaredProblem(bounds=bounds, minmax="min")

if __name__ == "__main__":
    model = BBO.OriginalBBO(epoch=10, pop_size=30)
    model = PSO.OriginalPSO(epoch=100, pop_size=50)
    model = PSO.LDW_PSO(epoch=100, pop_size=50)
    model = PSO.AIW_PSO(epoch=100, pop_size=50)
    model = PSO.P_PSO(epoch=100, pop_size=50)
    model = PSO.HPSO_TVAC(epoch=100, pop_size=50)
    model = PSO.C_PSO(epoch=100, pop_size=50)
    model = PSO.CL_PSO(epoch=100, pop_size=50)
    model = GA.BaseGA(epoch=100, pop_size=50)
    model = GA.SingleGA(epoch=100, pop_size=50)
    model = GA.MultiGA(epoch=100, pop_size=50)
    model = GA.EliteSingleGA(epoch=100, pop_size=50)
    model = GA.EliteMultiGA(epoch=100, pop_size=50)
    model = ABC.OriginalABC(epoch=1000, pop_size=50)
    model = ACOR.OriginalACOR(epoch=1000, pop_size=50)
    model = AGTO.OriginalAGTO(epoch=1000, pop_size=50)
    model = AGTO.MGTO(epoch=1000, pop_size=50)
    model = ALO.OriginalALO(epoch=100, pop_size=50)
    model = ALO.DevALO(epoch=100, pop_size=50)
    model = AO.OriginalAO(epoch=100, pop_size=50)
    model = AO.OriginalAO(epoch=100, pop_size=50)
    model = ARO.OriginalARO(epoch=100, pop_size=50)
    model = ARO.LARO(epoch=100, pop_size=50)
    model = ARO.IARO(epoch=100, pop_size=50)
    model = AVOA.OriginalAVOA(epoch=100, pop_size=50)
    model = BA.OriginalBA(epoch=100, pop_size=50)
    model = BA.AdaptiveBA(epoch=100, pop_size=50)
    model = BA.DevBA(epoch=100, pop_size=50)
    model = BBOA.OriginalBBOA(epoch=100, pop_size=50)
    model = BMO.OriginalBMO(epoch=100, pop_size=50)
    model = EOA.OriginalEOA(epoch=100, pop_size=50)
    model = IWO.OriginalIWO(epoch=100, pop_size=50)
    model = SBO.DevSBO(epoch=100, pop_size=50)
    model = SBO.OriginalSBO(epoch=100, pop_size=50)
    model = SMA.OriginalSMA(epoch=100, pop_size=50)
    model = SMA.DevSMA(epoch=100, pop_size=50)
    model = SOA.OriginalSOA(epoch=100, pop_size=50)
    model = SOA.DevSOA(epoch=100, pop_size=50)
    model = SOS.OriginalSOS(epoch=100, pop_size=50)
    model = TPO.DevTPO(epoch=100, pop_size=50)
    model = TSA.OriginalTSA(epoch=100, pop_size=50)
    model = VCS.OriginalVCS(epoch=100, pop_size=50)
    model = VCS.DevVCS(epoch=100, pop_size=50)
    model = WHO.OriginalWHO(epoch=100, pop_size=50)
    model = AOA.OriginalAOA(epoch=100, pop_size=50)
    model = CEM.OriginalCEM(epoch=100, pop_size=50)
    model = CGO.OriginalCGO(epoch=100, pop_size=50)
    model = CircleSA.OriginalCircleSA(epoch=100, pop_size=50)
    model = GBO.OriginalGBO(epoch=100, pop_size=50)
    model = HC.OriginalHC(epoch=100, pop_size=50)
    model = HC.SwarmHC(epoch=100, pop_size=50)
    model = INFO.OriginalINFO(epoch=100, pop_size=50)
    model = PSS.OriginalPSS(epoch=100, pop_size=50)
    model = RUN.OriginalRUN(epoch=100, pop_size=50)
    model = SCA.OriginalSCA(epoch=100, pop_size=50)
    model = SCA.DevSCA(epoch=100, pop_size=50)
    model = SCA.QleSCA(epoch=100, pop_size=50)
    model = SHIO.OriginalSHIO(epoch=100, pop_size=50)
    model = TS.OriginalTS(epoch=100, pop_size=50)
    model = HS.OriginalHS(epoch=100, pop_size=50)
    model = HS.DevHS(epoch=100, pop_size=50)
    model = AEO.OriginalAEO(epoch=100, pop_size=50)
    model = AEO.EnhancedAEO(epoch=100, pop_size=50)
    model = AEO.ModifiedAEO(epoch=100, pop_size=50)
    model = AEO.ImprovedAEO(epoch=100, pop_size=50)
    model = AEO.AugmentedAEO(epoch=100, pop_size=50)
    model = GCO.OriginalGCO(epoch=100, pop_size=50)
    model = GCO.DevGCO(epoch=100, pop_size=50)
    model = WCA.OriginalWCA(epoch=100, pop_size=50)
    model = CRO.OriginalCRO(epoch=100, pop_size=50)
    model = CRO.OCRO(epoch=100, pop_size=50)
    model = DE.OriginalDE(epoch=100, pop_size=50, wf=0.1)
    model = DE.JADE(epoch=100, pop_size=50)
    model = DE.SADE(epoch=100, pop_size=50)
    model = DE.SAP_DE(epoch=100, pop_size=50)
    model = EP.OriginalEP(epoch=100, pop_size=50)
    model = EP.LevyEP(epoch=100, pop_size=50)
    model = ES.OriginalES(epoch=100, pop_size=50)
    model = ES.LevyES(epoch=100, pop_size=50)
    model = ES.CMA_ES(epoch=100, pop_size=50)
    model = ES.Simple_CMA_ES(epoch=100, pop_size=50)
    model = FPA.OriginalFPA(epoch=100, pop_size=50)
    model = MA.OriginalMA(epoch=100, pop_size=50)
    model = SHADE.OriginalSHADE(epoch=100, pop_size=50)
    model = SHADE.L_SHADE(epoch=100, pop_size=50)
    model = BRO.OriginalBRO(epoch=100, pop_size=50)
    model = BRO.DevBRO(epoch=100, pop_size=50)
    model = BSO.OriginalBSO(epoch=100, pop_size=50)
    model = BSO.ImprovedBSO(epoch=100, pop_size=50)
    model = CA.OriginalCA(epoch=100, pop_size=50)
    model = CHIO.OriginalCHIO(epoch=100, pop_size=50)
    model = CHIO.DevCHIO(epoch=100, pop_size=50)
    model = FBIO.OriginalFBIO(epoch=100, pop_size=50)
    model = FBIO.DevFBIO(epoch=100, pop_size=50)
    model = GSKA.OriginalGSKA(epoch=100, pop_size=50)
    model = GSKA.DevGSKA(epoch=100, pop_size=50)
    model = HBO.OriginalHBO(epoch=100, pop_size=50)
    model = HCO.OriginalHCO(epoch=100, pop_size=50)
    model = ICA.OriginalICA(epoch=100, pop_size=50)
    model = LCO.OriginalLCO(epoch=100, pop_size=50)
    model = LCO.ImprovedLCO(epoch=100, pop_size=50)
    model = LCO.DevLCO(epoch=100, pop_size=50)
    model = WarSO.OriginalWarSO(epoch=100, pop_size=50)
    model = TOA.OriginalTOA(epoch=100, pop_size=50)
    model = TLO.OriginalTLO(epoch=100, pop_size=50)
    model = TLO.ImprovedTLO(epoch=100, pop_size=50)
    model = TLO.DevTLO(epoch=100, pop_size=50)
    model = SSDO.OriginalSSDO(epoch=100, pop_size=50)
    model = SPBO.OriginalSPBO(epoch=100, pop_size=50)
    model = SPBO.DevSPBO(epoch=100, pop_size=50)
    model = SARO.OriginalSARO(epoch=100, pop_size=50)
    model = SARO.DevSARO(epoch=100, pop_size=50)
    model = QSA.OriginalQSA(epoch=100, pop_size=50)
    model = QSA.DevQSA(epoch=100, pop_size=50)
    model = QSA.OppoQSA(epoch=100, pop_size=50)
    model = QSA.LevyQSA(epoch=100, pop_size=50)
    model = QSA.ImprovedQSA(epoch=100, pop_size=50)
    model = ArchOA.OriginalArchOA(epoch=100, pop_size=50)
    model = ASO.OriginalASO(epoch=100, pop_size=50)
    model = CDO.OriginalCDO(epoch=100, pop_size=50)
    model = EFO.OriginalEFO(epoch=100, pop_size=50)
    model = EFO.DevEFO(epoch=100, pop_size=50)
    model = EO.OriginalEO(epoch=100, pop_size=50)
    model = EO.AdaptiveEO(epoch=100, pop_size=50)
    model = EO.ModifiedEO(epoch=100, pop_size=50)
    model = EVO.OriginalEVO(epoch=100, pop_size=50)
    model = FLA.OriginalFLA(epoch=100, pop_size=50)
    model = HGSO.OriginalHGSO(epoch=100, pop_size=50)
    model = MVO.OriginalMVO(epoch=100, pop_size=50)
    model = MVO.DevMVO(epoch=100, pop_size=50)
    model = NRO.OriginalNRO(epoch=100, pop_size=50)
    model = RIME.OriginalRIME(epoch=100, pop_size=50)
    model = SA.OriginalSA(epoch=100, pop_size=50)
    model = SA.GaussianSA(epoch=100, pop_size=50)
    model = SA.SwarmSA(epoch=100, pop_size=50)
    model = WDO.OriginalWDO(epoch=100, pop_size=50)
    model = TWO.OriginalTWO(epoch=100, pop_size=50)
    model = TWO.EnhancedTWO(epoch=100, pop_size=50)
    model = TWO.OppoTWO(epoch=100, pop_size=50)
    model = TWO.LevyTWO(epoch=100, pop_size=50)
    model = ABC.OriginalABC(epoch=100, pop_size=50)
    model = ACOR.OriginalACOR(epoch=100, pop_size=50)
    model = AGTO.OriginalAGTO(epoch=100, pop_size=50)
    model = AGTO.MGTO(epoch=100, pop_size=50)
    model = BeesA.OriginalBeesA(epoch=100, pop_size=50)
    model = BeesA.CleverBookBeesA(epoch=100, pop_size=50)
    model = BeesA.ProbBeesA(epoch=100, pop_size=50)
    model = BES.OriginalBES(epoch=100, pop_size=50)
    model = BFO.OriginalBFO(epoch=100, pop_size=50)
    model = BFO.ABFO(epoch=100, pop_size=50)
    model = ZOA.OriginalZOA(epoch=100, pop_size=50)
    model = WOA.OriginalWOA(epoch=100, pop_size=50)
    model = WOA.HI_WOA(epoch=100, pop_size=50)
    model = WaOA.OriginalWaOA(epoch=100, pop_size=50)
    model = TSO.OriginalTSO(epoch=100, pop_size=50)
    model = TDO.OriginalTDO(epoch=100, pop_size=50)
    model = STO.OriginalSTO(epoch=100, pop_size=50)
    model = SSpiderO.OriginalSSpiderO(epoch=100, pop_size=50)
    model = SSpiderA.OriginalSSpiderA(epoch=100, pop_size=50)
    model = SSO.OriginalSSO(epoch=100, pop_size=50)
    model = SSA.OriginalSSA(epoch=100, pop_size=50)
    model = SSA.DevSSA(epoch=100, pop_size=50)
    model = SRSR.OriginalSRSR(epoch=100, pop_size=50)
    model = SLO.OriginalSLO(epoch=100, pop_size=50)
    model = SLO.ModifiedSLO(epoch=100, pop_size=50)
    model = SLO.ImprovedSLO(epoch=100, pop_size=50)
    model = SHO.OriginalSHO(epoch=100, pop_size=50)
    model = SFO.OriginalSFO(epoch=100, pop_size=50)
    model = SFO.ImprovedSFO(epoch=100, pop_size=50)
    model = ServalOA.OriginalServalOA(epoch=100, pop_size=50)
    model = SeaHO.OriginalSeaHO(epoch=100, pop_size=50)
    model = SCSO.OriginalSCSO(epoch=100, pop_size=50)
    model = POA.OriginalPOA(epoch=100, pop_size=50)
    model = PFA.OriginalPFA(epoch=100, pop_size=50)
    model = OOA.OriginalOOA(epoch=100, pop_size=50)
    model = NGO.OriginalNGO(epoch=100, pop_size=50)
    model = NMRA.OriginalNMRA(epoch=100, pop_size=50)
    model = NMRA.ImprovedNMRA(epoch=100, pop_size=50)
    model = MSA.OriginalMSA(epoch=100, pop_size=50)
    model = MRFO.OriginalMRFO(epoch=100, pop_size=50)
    model = MRFO.WMQIMRFO(epoch=100, pop_size=50)
    model = MPA.OriginalMPA(epoch=100, pop_size=50)
    model = MGO.OriginalMGO(epoch=100, pop_size=50)
    model = MFO.OriginalMFO(epoch=100, pop_size=50)
    model = JA.OriginalJA(epoch=100, pop_size=50)
    model = JA.LevyJA(epoch=100, pop_size=50)
    model = JA.DevJA(epoch=100, pop_size=50)
    model = HHO.OriginalHHO(epoch=100, pop_size=50)
    model = HGS.OriginalHGS(epoch=100, pop_size=50)
    model = HBA.OriginalHBA(epoch=100, pop_size=50)
    model = GWO.OriginalGWO(epoch=100, pop_size=50)
    model = GWO.GWO_WOA(epoch=100, pop_size=50)
    model = GWO.RW_GWO(epoch=100, pop_size=50)
    model = GTO.OriginalGTO(epoch=100, pop_size=50)
    model = GTO.Matlab101GTO(epoch=100, pop_size=50)
    model = GTO.Matlab102GTO(epoch=100, pop_size=50)
    model = GOA.OriginalGOA(epoch=100, pop_size=50)
    model = GJO.OriginalGJO(epoch=100, pop_size=50)
    model = FOX.OriginalFOX(epoch=100, pop_size=50)
    model = FOA.OriginalFOA(epoch=100, pop_size=50)
    model = FOA.WhaleFOA(epoch=100, pop_size=50)
    model = FOA.DevFOA(epoch=100, pop_size=50)
    model = FFO.OriginalFFO(epoch=100, pop_size=50)
    model = FFA.OriginalFFA(epoch=100, pop_size=50)
    model = FA.OriginalFA(epoch=100, pop_size=50)
    model = ESOA.OriginalESOA(epoch=100, pop_size=50)
    model = EHO.OriginalEHO(epoch=100, pop_size=50)
    model = DO.OriginalDO(epoch=100, pop_size=50)
    model = DMOA.OriginalDMOA(epoch=100, pop_size=50)
    model = DMOA.DevDMOA(epoch=100, pop_size=50)
    model = CSO.OriginalCSO(epoch=100, pop_size=50)
    model = CSA.OriginalCSA(epoch=100, pop_size=50)
    model = CoatiOA.OriginalCoatiOA(epoch=100, pop_size=50)
    model = COA.OriginalCOA(epoch=100, pop_size=50)
    model = BSA.OriginalBSA(epoch=100, pop_size=50)

    model.solve(P1)
    print(model.problem.bounds)
    print(model.history)
    print(model.g_best.solution)
    print(model.g_best.target)
    print(model.g_best.target.fitness)
    print(model.g_best.target.objectives)


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
    best_position, best_objective_function = model.solve(P1, mode="thread", n_workers=4, termination=term_dict1)
    # print(model.nfe)

    model = GWO.RW_GWO(epoch, pop_size)
    model.solve(P1)
    print(model.nfe_counter)

    print(best_position)
    print(model.get_parameters())
    print(model.get_attributes()["epoch"])
