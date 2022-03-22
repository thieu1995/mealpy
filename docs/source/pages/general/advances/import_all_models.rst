Import All Models
=================

.. code-block:: python

	from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
	from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
	from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
	from mealpy.math_based import AOA, CGO, GBO, HC, SCA, PSS
	from mealpy.music_based import HS
	from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, SA, TWO, WDO
	from mealpy.probabilistic_based import CEM
	from mealpy.system_based import AEO, GCO, WCA
	from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
	from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA

	import numpy as np


	def fitness_function(solution):
	    return np.sum(solution ** 2)


	problem = {
	    "fit_func": fitness_function,
	    "lb": [-3],
	    "ub": [5],
	    "n_dims": 30,
	    "save_population": False,
	    "log_to": None,
	    "log_file": "results.log"
	}

	if __name__ == "__main__":
	    ## Run the algorithm
	    model = BBO.BaseBBO(problem_dict1, epoch=31, pop_size=10, p_m=0.01, elites=3, termination=term_dict, name="HGS", fit_name="F0")
	    model = EOA.BaseEOA(problem, epoch=10, pop_size=50, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=0.9, gamma=0.9)
	    model = IWO.OriginalIWO(problem, epoch=10, pop_size=50, seeds=(2, 10), exponent=2, sigmas=(0.1, 0.001))
	    model = SBO.BaseSBO(problem, epoch=10, pop_size=50, alpha=0.94, p_m=0.05, psw=0.02)
	    model = SBO.OriginalSBO(problem, epoch=10, pop_size=50, alpha=0.94, p_m=0.05, psw=0.02)
	    model = SMA.OriginalSMA(problem, epoch=10, pop_size=50, p_t=0.1, termination=term_dict)
	    model = SMA.BaseSMA(problem, epoch=10, pop_size=50, p_t=0.1, termination=term_dict)
	    model = VCS.OriginalVCS(problem, epoch=10, pop_size=50, lamda=0.99, xichma=0.2)
	    model = VCS.BaseVCS(problem, epoch=10, pop_size=50, lamda=0.99, xichma=2.2)
	    model = WHO.BaseWHO(problem, epoch=10, pop_size=50, n_s=3, n_e=3, eta=0.15, p_hi=0.9, local_move=(0.9, 0.3), global_move=(0.2, 0.8), delta=(2.0, 2.0))
	    model = CRO.BaseCRO(problem_dict1, epoch=20, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, G=(0.05, 0.21), n_trials=5)
	    model = CRO.OCRO(problem_dict1, epoch=20, pop_size=50, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1, G=(0.05, 0.21), n_trials=5, restart_count=10)
	    model = DE.BaseDE(problem_dict1, epoch=20, pop_size=50, wf=0.1, cr=0.9, strategy=5)
	    model = DE.JADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
	    model = DE.SADE(problem_dict1, epoch=20, pop_size=50)
	    model = DE.SHADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5)
	    model = DE.L_SHADE(problem_dict1, epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5)
	    model = DE.SAP_DE(problem_dict1, epoch=20, pop_size=50, branch="ABS")
	    model = EP.BaseEP(problem_dict1, epoch=20, pop_size=50, bout_size=0.2)
	    model = EP.LevyEP(problem_dict1, epoch=20, pop_size=50, bout_size=0.2)
	    model = ES.BaseES(problem_dict1, epoch=20, pop_size=50, lamda=0.7)
	    model = ES.LevyES(problem_dict1, epoch=20, pop_size=50, lamda=0.7)
	    model = FPA.BaseFPA(problem_dict1, epoch=20, pop_size=50, p_s=0.8, levy_multiplier=0.2)
	    model = GA.BaseGA(problem_dict1, epoch=10, pop_size=50, pc=0.95, pm=0.1, mutation_multipoints=True, mutation="flip")
	    model = MA.BaseMA(problem_dict1, epoch=10, pop_size=50, pc=0.85, pm=0.1, p_local=0.5, max_local_gens=25, bits_per_param=4)
	    model = BRO.OriginalBRO(problem_dict1, epoch=100, pop_size=50, threshold=1)
	    model = BRO.BaseBRO(problem_dict1, epoch=10, pop_size=50, threshold=1)
	    model = BSO.ImprovedBSO(problem_dict1, epoch=10, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5)
	    model = BSO.BaseBSO(problem_dict1, epoch=10, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, slope=30)
	    model = CA.OriginalCA(problem_dict1, epoch=10, pop_size=50, accepted_rate=0.15)
	    model = CHIO.OriginalCHIO(problem_dict1, epoch=10, pop_size=50, brr=0.15, max_age=2)
	    model = CHIO.BaseCHIO(problem_dict1, epoch=10, pop_size=50, brr=0.15, max_age=1)
	    model = FBIO.OriginalFBIO(problem_dict1, epoch=11, pop_size=50)
	    model = FBIO.BaseFBIO(problem_dict1, epoch=10, pop_size=50)
	    model = GSKA.OriginalGSKA(problem_dict1, epoch=10, pop_size=50, pb=0.1, kf=0.5, kr=0.9, kg=2)
	    model = GSKA.BaseGSKA(problem_dict1, epoch=10, pop_size=50, pb=0.1, kr=0.7)
	    model = ICA.BaseICA(problem_dict1, epoch=10, pop_size=50, empire_count=5, assimilation_coeff=1.5,
	                        revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1, zeta=0.1)
	    model = LCO.OriginalLCO(problem_dict1, epoch=10, pop_size=50, r1=2.35)
	    model = LCO.BaseLCO(problem_dict1, epoch=10, pop_size=50, r1=2.35)
	    model = LCO.ImprovedLCO(problem_dict1, epoch=10, pop_size=50)
	    model = QSA.BaseQSA(problem_dict1, epoch=10, pop_size=50)
	    model = QSA.OriginalQSA(problem_dict1, epoch=10, pop_size=50)
	    model = QSA.OppoQSA(problem_dict1, epoch=10, pop_size=50)
	    model = QSA.LevyQSA(problem_dict1, epoch=10, pop_size=50)
	    model = QSA.ImprovedQSA(problem_dict1, epoch=10, pop_size=50)
	    model = SARO.OriginalSARO(problem_dict1, epoch=10, pop_size=50, se=0.5, mu=5)
	    model = SARO.BaseSARO(problem_dict1, epoch=10, pop_size=50, se=0.5, mu=25)
	    model = SSDO.BaseSSDO(problem_dict1, epoch=10, pop_size=50)
	    model = TLO.BaseTLO(problem_dict1, epoch=10, pop_size=50)
	    model = TLO.OriginalTLO(problem_dict1, epoch=10, pop_size=50)
	    model = TLO.ITLO(problem_dict1, epoch=10, pop_size=50, n_teachers=5)
	    model = AOA.OriginalAOA(problem_dict1, epoch=30, pop_size=50, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9)
	    model = CGO.OriginalCGO(problem_dict1, epoch=10, pop_size=50)
	    model = GBO.OriginalGBO(problem_dict1, epoch=10, pop_size=50, pr=0.5, beta_minmax=(0.2,1.2))
	    model = HC.OriginalHC(problem_dict1, epoch=10, pop_size=50, neighbour_size=20)
	    model = HC.BaseHC(problem_dict1, epoch=10, pop_size=50, neighbour_size=20)
	    model = PSS.OriginalPSS(problem_dict1, epoch=10, pop_size=50, acceptance_rate=0.9, sampling_method="MC")
	    model = SCA.BaseSCA(problem_dict1, epoch=10, pop_size=50)
	    model = SCA.OriginalSCA(problem_dict1, epoch=10, pop_size=50)
	    model = HS.BaseHS(problem_dict1, epoch=10, pop_size=50)
	    model = HS.OriginalHS(problem_dict1, epoch=10, pop_size=50)
	    model = ArchOA.OriginalArchOA(problem_dict1, epoch=10, pop_size=50, c1=2, c2=6, c3=2, c4=0.5, acc_max=0.9, acc_min=0.1)
	    model = ArchOA.OriginalArchOA(problem_dict1, epoch=10, pop_size=50, alpha=50, beta=0.2)
	    model = EFO.OriginalEFO(problem_dict1, epoch=10, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
	    model = EFO.BaseEFO(problem_dict1, epoch=10, pop_size=50, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
	    model = EO.BaseEO(problem_dict1, epoch=10, pop_size=50)
	    model = EO.ModifiedEO(problem_dict1, epoch=10, pop_size=50)
	    model = EO.AdaptiveEO(problem_dict1, epoch=10, pop_size=50)
	    model = HGSO.BaseHGSO(problem_dict1, epoch=10, pop_size=50, n_clusters=3)
	    model = MVO.BaseMVO(problem_dict1, epoch=10, pop_size=50, wep_min=0.2, wep_max=1.0)
	    model = MVO.OriginalMVO(problem_dict1, epoch=10, pop_size=50, wep_min=0.2, wep_max=1.0)
	    model = NRO.BaseNRO(problem_dict1, epoch=10, pop_size=50)
	    model = SA.BaseSA(problem_dict1, epoch=10, pop_size=50, max_sub_iter=5, t0=1000, t1=1, move_count=5, mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99)
	    model = TWO.BaseTWO(problem_dict1, epoch=10, pop_size=50)
	    model = TWO.OppoTWO(problem_dict1, epoch=10, pop_size=50)
	    model = TWO.LevyTWO(problem_dict1, epoch=100, pop_size=50)
	    model = TWO.EnhancedTWO(problem_dict1, epoch=10, pop_size=50)
	    model = WDO.BaseWDO(problem, epoch=100, pop_size=50, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3)
	    model = CEM.BaseCEM(problem_dict1, epoch=10, pop_size=50, n_best=40, alpha=0.5)
	    model = AEO.OriginalAEO(problem_dict1, epoch=10, pop_size=50)
	    model = AEO.AdaptiveAEO(problem_dict1, epoch=10, pop_size=50)
	    model = AEO.ModifiedAEO(problem_dict1, epoch=10, pop_size=50)
	    model = AEO.EnhancedAEO(problem_dict1, epoch=10, pop_size=50)
	    model = AEO.IAEO(problem_dict1, epoch=10, pop_size=50)
	    model = GCO.BaseGCO(problem_dict1, epoch=10, pop_size=50, cr=0.7, wf=1.25)
	    model = GCO.OriginalGCO(problem_dict1, epoch=10, pop_size=50, cr=0.7, wf=1.25)
	    model = WCA.BaseWCA(problem_dict1, epoch=10, pop_size=50, nsr=4, wc=2.0, dmax=1e-6)
	    model = ABC.BaseABC(problem_dict1, epoch=10, pop_size=50, couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1))
	    model = ACOR.BaseACOR(problem, epoch=10, pop_size=50, sample_count=25, intent_factor=0.5, zeta=1.0)
	    model = ALO.OriginalALO(problem, epoch=10, pop_size=50)
	    model = ALO.BaseALO(problem, epoch=10, pop_size=50)
	    model = AO.OriginalAO(problem, epoch=10, pop_size=50)
	    model = BA.BaseBA(problem, epoch=10, pop_size=50, loudness=(1.0, 2.0), pulse_rate=(0.15, 0.85), pulse_frequency=(0, 10))
	    model = BA.OriginalBA(problem, epoch=10, pop_size=50, loudness=0.8, pulse_rate=0.95, pulse_frequency=(0, 10))
	    model = BA.ModifiedBA(problem, epoch=10, pop_size=50, pulse_rate=0.95, pulse_frequency=(0, 10))
	    model = BeesA.BaseBeesA(problem_dict1, epoch=10, pop_size=50, site_ratio=(0.5, 0.4), site_bee_ratio=(0.1, 2.0), dance_factor=(0.1, 0.99))
	    model = BeesA.BaseBeesA(problem_dict1, epoch=10, pop_size=50, recruited_bee_ratio=0.1, dance_factor=(0.1, 0.99))
	    model = BES.BaseBES(problem_dict1, epoch=10, pop_size=50, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0)
	    model = BFO.OriginalBFO(problem_dict1, epoch=10, pop_size=50, Ci=0.01, Ped=0.25, Nc=5, Ns=4, attract_repels=(0.1, 0.2, 0.1, 10))
	    model = BFO.ABFO(problem_dict1, epoch=10, pop_size=50, Ci=(0.1, 0.001), Ped=0.01, Ns=4, N_minmax=(2, 40))
	    model = BSA.BaseBSA(problem_dict1, epoch=10, pop_size=50, ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5)
	    model = BSA.BaseBSA(problem_dict1, epoch=10, pop_size=50, ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5)
	    model = COA.BaseCOA(problem_dict1, epoch=10, pop_size=50, n_coyotes=5)
	    model = CSA.BaseCSA(problem_dict1, epoch=10, pop_size=50, p_a=0.7)
	    model = CSO.BaseCSO(problem_dict1, epoch=10, pop_size=50, mixture_ratio=0.15, smp=5,
	                        spc=False, cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1)
	    model = DO.BaseDO(problem_dict1, epoch=10, pop_size=50)
	    model = EHO.BaseEHO(problem_dict1, epoch=10, pop_size=50, alpha=0.5, beta=0.5, n_clans=5)
	    model = FA.BaseFA(problem_dict1, epoch=10, pop_size=50, max_sparks=10, p_a=0.04, p_b=0.8, max_ea=30, m_sparks=5)
	    model = FFA.BaseFFA(problem_dict1, epoch=10, pop_size=50, gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2)
	    model = FOA.OriginalFOA(problem_dict1, epoch=10, pop_size=50)
	    model = FOA.BaseFOA(problem_dict1, epoch=10, pop_size=50)
	    model = FOA.WhaleFOA(problem_dict1, epoch=10, pop_size=50)
	    model = GOA.BaseGOA(problem_dict1, epoch=10, pop_size=50, c_minmax=(0.00004, 1))
	    model = GWO.BaseGWO(problem_dict1, epoch=10, pop_size=50)
	    model = GWO.RW_GWO(problem_dict1, epoch=10, pop_size=50)
	    model = HGS.OriginalHGS(problem_dict1, epoch=10, pop_size=50, PUP=0.08, LH=10000)
	    model = HHO.BaseHHO(problem_dict1, epoch=10, pop_size=50)
	    model = JA.BaseJA(problem_dict1, epoch=10, pop_size=50)
	    model = JA.OriginalJA(problem_dict1, epoch=10, pop_size=50)
	    model = JA.LevyJA(problem_dict1, epoch=10, pop_size=50)
	    model = MFO.OriginalMFO(problem_dict1, epoch=10, pop_size=50)
	    model = MFO.BaseMFO(problem_dict1, epoch=10, pop_size=50)
	    model = MRFO.BaseMRFO(problem_dict1, epoch=10, pop_size=50)
	    model = MSA.BaseMSA(problem_dict1, epoch=10, pop_size=50, n_best=5, partition=0.5, max_step_size=1.0)
	    model = NMRA.BaseNMRA(problem_dict1, epoch=10, pop_size=50)
	    model = NMRA.ImprovedNMRA(problem_dict1, epoch=10, pop_size=50, pb=0.75, pm=0.01)
	    model = PFA.BasePFA(problem_dict1, epoch=10, pop_size=50)
	    model = PSO.BasePSO(problem_dict1, epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
	    model = PSO.C_PSO(problem_dict1, epoch=10, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
	    model = PSO.CL_PSO(problem_dict1, epoch=10, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7)
	    model = PSO.PPSO(problem_dict1, epoch=10, pop_size=50)
	    model = PSO.HPSO_TVAC(problem_dict1, epoch=10, pop_size=50, ci=0.5, cf=0.2)
	    model = SFO.BaseSFO(problem_dict1, epoch=10, pop_size=50, pp=0.1, AP=4, epxilon=0.0001)
	    model = SFO.ImprovedSFO(problem_dict1, epoch=10, pop_size=50, pp=0.1)
	    model = SHO.BaseSHO(problem_dict1, epoch=10, pop_size=50, h_factor=1, rand_v=(0, 2), N_tried=30)
	    model = SLO.BaseSLO(problem_dict1, epoch=10, pop_size=50)
	    model = SLO.ModifiedSLO(problem_dict1, epoch=10, pop_size=50)
	    model = SLO.ISLO(problem_dict1, epoch=10, pop_size=50, c1=1.2, c2=1.2)
	    model = SRSR.BaseSRSR(problem_dict1, epoch=10, pop_size=50)
	    model = SSA.OriginalSSA(problem_dict1, epoch=10, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
	    model = SSA.BaseSSA(problem_dict1, epoch=10, pop_size=50, ST=0.8, PD=0.2, SD=0.1)
	    model = SSO.BaseSSO(problem, epoch=10, pop_size=50)
	    model = SSpiderA.BaseSSpiderA(problem_dict1, epoch=10, pop_size=50, r_a=1, p_c=0.7, p_m=0.1)
	    model = SSpiderO.BaseSSpiderO(problem_dict1, epoch=10, pop_size=50, fp=(0.65, 0.9))
	    model = WOA.BaseWOA(problem_dict1, epoch=10, pop_size=50)
	    model = WOA.HI_WOA(problem_dict1, epoch=10, pop_size=50, feedback_max=5)

	    best_position, best_fitness = model.solve()
	    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
