#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>>
# >>> from mealpy.swarm_based import PSO
# >>> import numpy as np
# >>>
# >>> def fitness_function(solution):
# >>>     return np.sum(solution ** 2)
# >>>
# >>> problem = {
# >>>    "fit_func": fitness_function,
# >>>    "lb": [-100, ] * 30,
# >>>    "ub": [100, ] * 30,
# >>>    "minmax": "min",
# >>>    "save_population": True,
# >>>    "log_to": "file",
# >>>    "log_file": "mealpy.log",
# >>>    "name": Square",
# >>> }
# >>>
# >>> ## Run the algorithm
# >>> model = PSO.C_PSO(epoch=5, pop_size=50, name="C-PSO")
# >>> best_position, best_fitness = model.solve(problem)
# >>> print(f"Best solution: {best_position}, Best fitness: {best_fitness}")


__version__ = "2.5.4-alpha.3"

from .bio_based import (BBO, BBOA, BMO, EOA, IWO, SBO, SMA, SOA, SOS, TPO, TSA, VCS, WHO)
from .evolutionary_based import (CRO, DE, EP, ES, FPA, GA, MA)
from .human_based import (BRO, BSO, CA, CHIO, FBIO, GSKA, HBO, HCO, ICA, LCO, QSA, SARO, SPBO, SSDO, TLO, TOA, WarSO)
from .math_based import (AOA, CEM, CGO, CircleSA, GBO, HC, INFO, PSS, RUN, SCA, SHIO)
from .physics_based import (ArchOA, ASO, CDO, EFO, EO, EVO, FLA, HGSO, MVO, NRO, RIME, SA, TWO, WDO)
from .swarm_based import (ABC, ACOR, AGTO, ALO, AO, ARO, AVOA, BA, BeesA, BES, BFO, BSA, COA, CoatiOA, CSA, CSO,
                          DMOA, DO, EHO, ESOA, FA, FFA, FFO, FOA, FOX, GJO, GOA, GTO, GWO, HBA, HGS, HHO, JA,
                          MFO, MGO, MPA, MRFO, MSA, NGO, NMRA, OOA, PFA, POA, PSO, SCSO, SeaHO, ServalOA, SFO,
                          SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, STO, TDO, TSO, WaOA, WOA, ZOA)
from .system_based import AEO, GCO, WCA
from .music_based import HS
from .utils.problem import Problem
from .utils.termination import Termination
from .tuner import Tuner
from .multitask import Multitask
