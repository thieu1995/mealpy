#!/usr/bin/env python
# Created by "Thieu" at 16:53, 20/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec2017 import F292017

from mealpy import BBO, PSO, GA, ALO, AO, ARO, AVOA, BA, BBOA, BMO, EOA, IWO
from mealpy import FloatVar
from mealpy import GJO, FOX, FOA, FFO, FFA, FA, ESOA, EHO, DO, DMOA, CSO, CSA, CoatiOA, COA, BSA
from mealpy import HCO, ICA, LCO, WarSO, TOA, TLO, SSDO, SPBO, SARO, QSA, ArchOA, ASO, CDO, EFO, EO, EVO, FLA
from mealpy import HGSO, MVO, NRO, RIME, SA, WDO, TWO, ABC, ACOR, AGTO, BeesA, BES, BFO, ZOA, WOA, WaOA, TSO
from mealpy import PFA, OOA, NGO, NMRA, MSA, MRFO, MPA, MGO, MFO, JA, HHO, HGS, HBA, GWO, GTO, GOA
from mealpy import Problem
from mealpy import SBO, SMA, SOA, SOS, TPO, TSA, VCS, WHO, AOA, CEM, CGO, CircleSA, GBO, HC, INFO, PSS, RUN, SCA
from mealpy import SHIO, TS, HS, AEO, GCO, WCA, CRO, DE, EP, ES, FPA, MA, SHADE, BRO, BSO, CA, CHIO, FBIO, GSKA, HBO
from mealpy import TDO, STO, SSpiderO, SSpiderA, SSO, SSA, SRSR, SLO, SHO, SFO, ServalOA, SeaHO, SCSO, POA
from mealpy import (StringVar, FloatVar, BoolVar, PermutationVar, MixedSetVar, IntegerVar, BinaryVar,
                    TransferBinaryVar, TransferBoolVar)
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
    def __init__(self, bounds, minmax, **kwargs):
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        x = self.decode_solution(solution)["variable"]
        return np.sum(x ** 2)



bounds = FloatVar(lb=(-15.,) * 100, ub=(20.,) * 100, name="variable")
P1 = SquaredProblem(bounds=bounds, minmax="min")

model = WarSO.OriginalWarSO(epoch=100, pop_size=50)
model = VCS.OriginalVCS(epoch=100, pop_size=50)
model = EP.OriginalEP(epoch=100, pop_size=50)
model.solve(P1)




# if __name__ == "__main__":
#     model = BBO.OriginalBBO(epoch=10, pop_size=30, p_m=0.01, n_elites=2)
#     model = PSO.OriginalPSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, w=0.4)
#     model = PSO.LDW_PSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
#     model = PSO.AIW_PSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, alpha=0.4)
#     model = PSO.P_PSO(epoch=100, pop_size=50)
#     model = PSO.HPSO_TVAC(epoch=100, pop_size=50, ci=0.5, cf=0.1)
#     model = PSO.C_PSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
#     model = PSO.CL_PSO(epoch=100, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7)
#     model = GA.BaseGA(epoch=100, pop_size=50, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points", mutation="swap")
#     model = GA.SingleGA(epoch=100, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points", mutation="swap")
#     model = GA.MultiGA(epoch=100, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points", mutation="swap")
#     model = GA.EliteSingleGA(epoch=100, pop_size=50, pc=0.95, pm=0.8, selection="roulette", crossover="uniform", mutation="swap", k_way=0.2, elite_best=0.1,
#                              elite_worst=0.3, strategy=0)
#     model = GA.EliteMultiGA(epoch=100, pop_size=50, pc=0.95, pm=0.8, selection="roulette", crossover="uniform", mutation="swap", k_way=0.2, elite_best=0.1,
#                             elite_worst=0.3, strategy=0)
#     model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits=50)
#     model = ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count=25, intent_factor=0.5, zeta=1.0)
#     model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
#     model = AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03)
#     model = ALO.OriginalALO(epoch=100, pop_size=50)
#     model = ALO.DevALO(epoch=100, pop_size=50)
#     model = AO.OriginalAO(epoch=100, pop_size=50)
#     model = ARO.OriginalARO(epoch=100, pop_size=50)
#     model = ARO.LARO(epoch=100, pop_size=50)
#     model = ARO.IARO(epoch=100, pop_size=50)
#     model = AVOA.OriginalAVOA(epoch=100, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
#     model = BA.OriginalBA(epoch=100, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0)
#     model = BA.AdaptiveBA(epoch=100, pop_size=50, loudness_min=1.0, loudness_max=2.0, pr_min=-2.5, pr_max=0.85, pf_min=0.1, pf_max=10.)
#     model = BA.DevBA(epoch=100, pop_size=50, pulse_rate=0.95, pf_min=0., pf_max=10.)
#     model = BBOA.OriginalBBOA(epoch=100, pop_size=50)
#     model = BMO.OriginalBMO(epoch=100, pop_size=50, pl=4)
#     model = EOA.OriginalEOA(epoch=100, pop_size=50, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=0.9, gama=0.9)
#     model = IWO.OriginalIWO(epoch=100, pop_size=50, seed_min=3, seed_max=9, exponent=3, sigma_start=0.6, sigma_end=0.01)
#     model = SBO.DevSBO(epoch=100, pop_size=50, alpha=0.9, p_m=0.05, psw=0.02)
#     model = SBO.OriginalSBO(epoch=100, pop_size=50, alpha=0.9, p_m=0.05, psw=0.02)
#     model = SMA.OriginalSMA(epoch=100, pop_size=50, p_t=0.03)
#     model = SMA.DevSMA(epoch=100, pop_size=50, p_t=0.03)
#     model = SOA.OriginalSOA(epoch=100, pop_size=50, fc=2)
#     model = SOA.DevSOA(epoch=100, pop_size=50, fc=2)
#     model = SOS.OriginalSOS(epoch=100, pop_size=50)
#     model = TPO.DevTPO(epoch=100, pop_size=50, alpha=0.3, beta=50., theta=0.9)
#     model = TSA.OriginalTSA(epoch=100, pop_size=50)
#     model = VCS.OriginalVCS(epoch=100, pop_size=50, lamda=0.5, sigma=0.3)
#     model = VCS.DevVCS(epoch=100, pop_size=50, lamda=0.5, sigma=0.3)
#     model = WHO.OriginalWHO(epoch=100, pop_size=50, n_explore_step=3, n_exploit_step=3, eta=0.15, p_hi=0.9, local_alpha=0.9, local_beta=0.3, global_alpha=0.2,
#                             global_beta=0.8, delta_w=2.0, delta_c=2.0)
#     model = AOA.OriginalAOA(epoch=100, pop_size=50, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9)
#     model = CEM.OriginalCEM(epoch=100, pop_size=50, n_best=20, alpha=0.7)
#     model = CGO.OriginalCGO(epoch=100, pop_size=50)
#     model = CircleSA.OriginalCircleSA(epoch=100, pop_size=50, c_factor=0.8)
#     model = GBO.OriginalGBO(epoch=100, pop_size=50, pr=0.5, beta_min=0.2, beta_max=1.2)
#     model = HC.OriginalHC(epoch=100, pop_size=50, neighbour_size=50)
#     model = HC.SwarmHC(epoch=100, pop_size=50, neighbour_size=10)
#     model = INFO.OriginalINFO(epoch=100, pop_size=50)
#     model = PSS.OriginalPSS(epoch=100, pop_size=50, acceptance_rate=0.8, sampling_method="LHS")
#     model = RUN.OriginalRUN(epoch=100, pop_size=50)
#     model = SCA.OriginalSCA(epoch=100, pop_size=50)
#     model = SCA.DevSCA(epoch=100, pop_size=50)
#     model = SCA.QleSCA(epoch=100, pop_size=50, alpha=0.1, gama=0.9)
#     model = SHIO.OriginalSHIO(epoch=100, pop_size=50)
#     model = TS.OriginalTS(epoch=100, pop_size=50, tabu_size=5, neighbour_size=20, perturbation_scale=0.05)
#     model = HS.OriginalHS(epoch=100, pop_size=50, c_r=0.95, pa_r=0.05)
#     model = HS.DevHS(epoch=100, pop_size=50, c_r=0.95, pa_r=0.05)
#     model = AEO.OriginalAEO(epoch=100, pop_size=50)
#     model = AEO.EnhancedAEO(epoch=100, pop_size=50)
#     model = AEO.ModifiedAEO(epoch=100, pop_size=50)
#     model = AEO.ImprovedAEO(epoch=100, pop_size=50)
#     model = AEO.AugmentedAEO(epoch=100, pop_size=50)
#     model = GCO.OriginalGCO(epoch=100, pop_size=50, cr=0.7, wf=1.25)
#     model = GCO.DevGCO(epoch=100, pop_size=50, cr=0.7, wf=1.25)
#     model = WCA.OriginalWCA(epoch=100, pop_size=50, nsr=4, wc=2.0, dmax=1e-6)
#     model = CRO.OriginalCRO(epoch=100, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, gamma_min=0.02, gamma_max=0.2, n_trials=5)
#     model = CRO.OCRO(epoch=100, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, gamma_min=0.02, gamma_max=0.2, n_trials=5, restart_count=50)
#     model = DE.OriginalDE(epoch=100, pop_size=50, wf=0.7, cr=0.9, strategy=0)
#     model = DE.JADE(epoch=100, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
#     model = DE.SADE(epoch=100, pop_size=50)
#     model = DE.SAP_DE(epoch=100, pop_size=50, branch="ABS")
#     model = EP.OriginalEP(epoch=100, pop_size=50, bout_size=0.05)
#     model = EP.LevyEP(epoch=100, pop_size=50, bout_size=0.05)
#     model = ES.OriginalES(epoch=100, pop_size=50, lamda=0.75)
#     model = ES.LevyES(epoch=100, pop_size=50, lamda=0.75)
#     model = ES.CMA_ES(epoch=100, pop_size=50)
#     model = ES.Simple_CMA_ES(epoch=100, pop_size=50)
#     model = FPA.OriginalFPA(epoch=100, pop_size=50, p_s=0.8, levy_multiplier=0.2)
#     model = MA.OriginalMA(epoch=100, pop_size=50, pc=0.85, pm=0.15, p_local=0.5, max_local_gens=10, bits_per_param=4)
#     model = SHADE.OriginalSHADE(epoch=100, pop_size=50, miu_f=0.5, miu_cr=0.5)
#     model = SHADE.L_SHADE(epoch=100, pop_size=50, miu_f=0.5, miu_cr=0.5)
#     model = BRO.OriginalBRO(epoch=100, pop_size=50, threshold=3)
#     model = BRO.DevBRO(epoch=100, pop_size=50, threshold=3)
#     model = BSO.OriginalBSO(epoch=100, pop_size=50, m_clusters=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, slope=20)
#     model = BSO.ImprovedBSO(epoch=100, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.6)
#     model = CA.OriginalCA(epoch=100, pop_size=50, accepted_rate=0.15)
#     model = CHIO.OriginalCHIO(epoch=100, pop_size=50, brr=0.15, max_age=10)
#     model = CHIO.DevCHIO(epoch=100, pop_size=50, brr=0.15, max_age=10)
#     model = FBIO.OriginalFBIO(epoch=100, pop_size=50)
#     model = FBIO.DevFBIO(epoch=100, pop_size=50)
#     model = GSKA.OriginalGSKA(epoch=100, pop_size=50, pb=0.1, kf=0.5, kr=0.9, kg=5)
#     model = GSKA.DevGSKA(epoch=100, pop_size=50, pb=0.1, kr=0.9)
#     model = HBO.OriginalHBO(epoch=100, pop_size=50, degree=3)
#     model = HCO.OriginalHCO(epoch=100, pop_size=50, wfp=0.65, wfv=0.05, c1=1.4, c2=1.4)
#     model = ICA.OriginalICA(epoch=100, pop_size=50, empire_count=5, assimilation_coeff=1.5, revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1,
#                             zeta=0.1)
#     model = LCO.OriginalLCO(epoch=100, pop_size=50, r1=2.35)
#     model = LCO.ImprovedLCO(epoch=100, pop_size=50)
#     model = LCO.DevLCO(epoch=100, pop_size=50, r1=2.35)
#     model = WarSO.OriginalWarSO(epoch=100, pop_size=50, rr=0.1)
#     model = TOA.OriginalTOA(epoch=100, pop_size=50)
#     model = TLO.OriginalTLO(epoch=100, pop_size=50)
#     model = TLO.ImprovedTLO(epoch=100, pop_size=50, n_teachers=5)
#     model = TLO.DevTLO(epoch=100, pop_size=50)
#     model = SSDO.OriginalSSDO(epoch=100, pop_size=50)
#     model = SPBO.OriginalSPBO(epoch=100, pop_size=50)
#     model = SPBO.DevSPBO(epoch=100, pop_size=50)
#     model = SARO.OriginalSARO(epoch=100, pop_size=50, se=0.5, mu=25)
#     model = SARO.DevSARO(epoch=100, pop_size=50, se=0.5, mu=25)
#     model = QSA.OriginalQSA(epoch=100, pop_size=50)
#     model = QSA.DevQSA(epoch=100, pop_size=50)
#     model = QSA.OppoQSA(epoch=100, pop_size=50)
#     model = QSA.LevyQSA(epoch=100, pop_size=50)
#     model = QSA.ImprovedQSA(epoch=100, pop_size=50)
#     model = ArchOA.OriginalArchOA(epoch=100, pop_size=50, c1=2, c2=5, c3=2, c4=0.5, acc_max=0.9, acc_min=0.1)
#     model = ASO.OriginalASO(epoch=100, pop_size=50, alpha=50, beta=0.2)
#     model = CDO.OriginalCDO(epoch=100, pop_size=50)
#     model = EFO.OriginalEFO(epoch=100, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
#     model = EFO.DevEFO(epoch=100, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
#     model = EO.OriginalEO(epoch=100, pop_size=50)
#     model = EO.AdaptiveEO(epoch=100, pop_size=50)
#     model = EO.ModifiedEO(epoch=100, pop_size=50)
#     model = EVO.OriginalEVO(epoch=100, pop_size=50)
#     model = FLA.OriginalFLA(epoch=100, pop_size=50, C1=0.5, C2=2.0, C3=0.1, C4=0.2, C5=2.0, DD=0.01)
#     model = HGSO.OriginalHGSO(epoch=100, pop_size=50, n_clusters=3)
#     model = MVO.OriginalMVO(epoch=100, pop_size=50, wep_min=0.2, wep_max=1.0)
#     model = MVO.DevMVO(epoch=100, pop_size=50, wep_min=0.2, wep_max=1.0)
#     model = NRO.OriginalNRO(epoch=100, pop_size=50)
#     model = RIME.OriginalRIME(epoch=100, pop_size=50, sr=5.0)
#     model = SA.OriginalSA(epoch=100, pop_size=50, temp_init=100, step_size=0.1)
#     model = SA.GaussianSA(epoch=100, pop_size=50, temp_init=100, cooling_rate=0.99, scale=0.1)
#     model = SA.SwarmSA(epoch=100, pop_size=50, max_sub_iter=5, t0=1000, t1=1, move_count=5, mutation_rate=0.1, mutation_step_size=0.1,
#                        mutation_step_size_damp=0.99)
#     model = WDO.OriginalWDO(epoch=100, pop_size=50, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3)
#     model = TWO.OriginalTWO(epoch=100, pop_size=50)
#     model = TWO.EnhancedTWO(epoch=100, pop_size=50)
#     model = TWO.OppoTWO(epoch=100, pop_size=50)
#     model = TWO.LevyTWO(epoch=100, pop_size=50)
#     model = ABC.OriginalABC(epoch=100, pop_size=50, n_limits=50)
#     model = ACOR.OriginalACOR(epoch=100, pop_size=50, sample_count=25, intent_factor=0.5, zeta=1.0)
#     model = AGTO.OriginalAGTO(epoch=100, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
#     model = AGTO.MGTO(epoch=100, pop_size=50, pp=0.03)
#     model = BeesA.OriginalBeesA(epoch=100, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4, selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0,
#                                 dance_radius=0.1, dance_reduction=0.99)
#     model = BeesA.CleverBookBeesA(epoch=100, pop_size=50, n_elites=16, n_others=4, patch_size=5.0, patch_reduction=0.985, n_sites=3, n_elite_sites=1)
#     model = BeesA.ProbBeesA(epoch=100, pop_size=50, recruited_bee_ratio=0.1, dance_radius=0.1, dance_reduction=0.99)
#     model = BES.OriginalBES(epoch=100, pop_size=50, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0)
#     model = BFO.OriginalBFO(epoch=100, pop_size=50, Ci=0.01, Ped=0.25, Nc=5, Ns=4, d_attract=0.1, w_attract=0.2, h_repels=0.1, w_repels=10)
#     model = BFO.ABFO(epoch=100, pop_size=50, C_s=0.1, C_e=0.001, Ped=0.01, Ns=4, N_adapt=2, N_split=40)
#     model = ZOA.OriginalZOA(epoch=100, pop_size=50)
#     model = WOA.OriginalWOA(epoch=100, pop_size=50)
#     model = WOA.HI_WOA(epoch=100, pop_size=50, feedback_max=10)
#     model = WaOA.OriginalWaOA(epoch=100, pop_size=50)
#     model = TSO.OriginalTSO(epoch=100, pop_size=50)
#     model = TDO.OriginalTDO(epoch=100, pop_size=50)
#     model = STO.OriginalSTO(epoch=100, pop_size=50)
#     model = SSpiderO.OriginalSSpiderO(epoch=100, pop_size=50, fp_min=0.65, fp_max=0.9)
#     model = SSpiderA.OriginalSSpiderA(epoch=100, pop_size=50, r_a=1.0, p_c=0.7, p_m=0.1)
#     model = SSO.OriginalSSO(epoch=100, pop_size=50)
#     model = SSA.OriginalSSA(epoch=100, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
#     model = SSA.DevSSA(epoch=100, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
#     model = SRSR.OriginalSRSR(epoch=100, pop_size=50)
#     model = SLO.OriginalSLO(epoch=100, pop_size=50)
#     model = SLO.ModifiedSLO(epoch=100, pop_size=50)
#     model = SLO.ImprovedSLO(epoch=100, pop_size=50, c1=1.2, c2=1.5)
#     model = SHO.OriginalSHO(epoch=100, pop_size=50, h_factor=5.0, n_trials=10)
#     model = SFO.OriginalSFO(epoch=100, pop_size=50, pp=0.1, AP=4.0, epsilon=0.0001)
#     model = SFO.ImprovedSFO(epoch=100, pop_size=50, pp=0.1)
#     model = ServalOA.OriginalServalOA(epoch=100, pop_size=50)
#     model = SeaHO.OriginalSeaHO(epoch=100, pop_size=50)
#     model = SCSO.OriginalSCSO(epoch=100, pop_size=50)
#     model = POA.OriginalPOA(epoch=100, pop_size=50)
#     model = PFA.OriginalPFA(epoch=100, pop_size=50)
#     model = OOA.OriginalOOA(epoch=100, pop_size=50)
#     model = NGO.OriginalNGO(epoch=100, pop_size=50)
#     model = NMRA.OriginalNMRA(epoch=100, pop_size=50, pb=0.75)
#     model = NMRA.ImprovedNMRA(epoch=100, pop_size=50, pb=0.75, pm=0.01)
#     model = MSA.OriginalMSA(epoch=100, pop_size=50, n_best=5, partition=0.5, max_step_size=1.0)
#     model = MRFO.OriginalMRFO(epoch=100, pop_size=50, somersault_range=2.0)
#     model = MRFO.WMQIMRFO(epoch=100, pop_size=50, somersault_range=2.0, pm=0.5)
#     model = MPA.OriginalMPA(epoch=100, pop_size=50)
#     model = MGO.OriginalMGO(epoch=100, pop_size=50)
#     model = MFO.OriginalMFO(epoch=100, pop_size=50)
#     model = JA.OriginalJA(epoch=100, pop_size=50)
#     model = JA.LevyJA(epoch=100, pop_size=50)
#     model = JA.DevJA(epoch=100, pop_size=50)
#     model = HHO.OriginalHHO(epoch=100, pop_size=50)
#     model = HGS.OriginalHGS(epoch=100, pop_size=50, PUP=0.08, LH=10000)
#     model = HBA.OriginalHBA(epoch=100, pop_size=50)
#     model = GWO.OriginalGWO(epoch=100, pop_size=50)
#     model = GWO.GWO_WOA(epoch=100, pop_size=50)
#     model = GWO.RW_GWO(epoch=100, pop_size=50)
#     model = GTO.OriginalGTO(epoch=100, pop_size=50, A=0.4, H=2.0)
#     model = GTO.Matlab101GTO(epoch=100, pop_size=50)
#     model = GTO.Matlab102GTO(epoch=100, pop_size=50)
#     model = GOA.OriginalGOA(epoch=100, pop_size=50, c_min=0.00004, c_max=1.0)
#     model = GJO.OriginalGJO(epoch=100, pop_size=50)
#     model = FOX.OriginalFOX(epoch=100, pop_size=50, c1=0.18, c2=0.82)
#     model = FOA.OriginalFOA(epoch=100, pop_size=50)
#     model = FOA.WhaleFOA(epoch=100, pop_size=50)
#     model = FOA.DevFOA(epoch=100, pop_size=50)
#     model = FFO.OriginalFFO(epoch=100, pop_size=50)
#     model = FFA.OriginalFFA(epoch=100, pop_size=50, gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2)
#     model = FA.OriginalFA(epoch=100, pop_size=50, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)
#     model = ESOA.OriginalESOA(epoch=100, pop_size=50)
#     model = EHO.OriginalEHO(epoch=100, pop_size=50, alpha=0.5, beta=0.5, n_clans=5)
#     model = DO.OriginalDO(epoch=100, pop_size=50)
#     model = DMOA.OriginalDMOA(epoch=100, pop_size=50, n_baby_sitter=3, peep=2)
#     model = DMOA.DevDMOA(epoch=100, pop_size=50, peep=2)
#     model = CSO.OriginalCSO(epoch=100, pop_size=50, mixture_ratio=0.15, smp=5, spc=False, cdc=0.8, srd=0.15, c1=0.4, w_min=0.4, w_max=0.9)
#     model = CSA.OriginalCSA(epoch=100, pop_size=50, p_a=0.3)
#     model = CoatiOA.OriginalCoatiOA(epoch=100, pop_size=50)
#     model = COA.OriginalCOA(epoch=100, pop_size=50, n_coyotes=5)
#     model = BSA.OriginalBSA(epoch=100, pop_size=50, ff=10, pff=0.8, c1=1.5, c2=1.5, a1=1.0, a2=1.0, fc=0.5)
#
#     model.solve(P1)
#     print(model.problem.bounds)
#     print(model.history)
#     print(model.g_best.solution)
#     print(model.g_best.target)
#     print(model.g_best.target.fitness)
#     print(model.g_best.target.objectives)
#     print(len(model.history.list_global_best))
#
#     ## 1st way
#     # model = GWO.IGWO(epoch, pop_size, a_min=0.02, a_max=1.6)
#
#     # for opt_name, opt_class in get_all_optimizers().items():
#     #     print(f"{opt_name}: {opt_class}")
#
#     ## 2nd way
#     model = get_optimizer_by_name("IGWO")(epoch, pop_size, a_min=0.02, a_max=1.6)
#     model = get_optimizer_by_name("OriginalHC")(epoch, pop_size=2)
#     model = get_optimizer_by_name("GaussianSA")(epoch, pop_size=50, temp_init=100)
#     model = get_optimizer_by_name("SwarmSA")(epoch, pop_size=50, temp_init=100)
#     model = get_optimizer_by_name("OriginalSA")(epoch, pop_size=50, temp_init=100)
#     gbest = model.solve(P1, mode="thread", n_workers=4, termination=term_dict1)
#     print(model.nfe_counter)
#
#     # model = GWO.RW_GWO(epoch, pop_size)
#     # model.solve(P1)
#     # print(model.nfe_counter)
#     #
#     # print(best_position)
#     # print(model.get_parameters())
#     # print(model.get_attributes()["epoch"])
#
#     term_dict2 = {
#         "max_epoch": 1000,
#         "max_time": 2.3,  # 10 seconds to run the program
#     }
#     model.solve(problem_dict1, termination=term_dict2)
#

