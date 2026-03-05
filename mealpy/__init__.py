#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>>
# >>> from mealpy import FloatVar, PSO
# >>> import numpy as np
# >>>
# >>> def objective_function(solution):
# >>>     return np.sum(solution ** 2)
# >>>
# >>> p1 = {
# >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
# >>>     "minmax": "min",
# >>>     "obj_func": objective_function,
# >>>     "save_population": True,  # To be able to draw the trajectory figure
# >>>     "log_to": "file",
# >>>     "log_file": "mealpy.log",
# >>>     "name": "Square"
# >>> }
# >>>
# >>> ## Run the algorithm
# >>> model = PSO.C_PSO(epoch=5, pop_size=50, name="C-PSO")
# >>> g_best = model.solve(problem)
# >>> print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")

__version__ = "3.0.3"

import sys
import inspect

# Bio‑based algorithms
from .bio_based import (
    BBO,
    BBOA,
    BMO,
    EOA,
    IWO,
    SBO,
    SMA,
    SOA,
    SOS,
    TPO,
    TSA,
    VCS,
    WHO,
    BCO,
    EAO,
    SFOA,
)

# Evolutionary algorithms (without BWO – moved to swarm_based)
from .evolutionary_based import (
    CRO,
    DE,
    EP,
    ES,
    FPA,
    GA,
    MA,
    SHADE,
)

# Human‑based algorithms
from .human_based import (
    BRO,
    BSO,
    CA,
    CHIO,
    FBIO,
    GSKA,
    HBO,
    HCO,
    ICA,
    LCO,
    QSA,
    SARO,
    SPBO,
    SSDO,
    TLO,
    TOA,
    WarSO,
    AFT,
    CDDO,
    DOA,
)

# Math‑based algorithms
from .math_based import (
    AOA,
    CEM,
    CGO,
    CircleSA,
    GBO,
    HC,
    INFO,
    PSS,
    RUN,
    SCA,
    SHIO,
    TS,
)

# Physics‑based algorithms (including MSO)
from .physics_based import (
    ArchOA,
    ASO,
    CDO,
    EFO,
    EO,
    EVO,
    FLA,
    HGSO,
    MVO,
    NRO,
    RIME,
    SA,
    TWO,
    WDO,
    ESO,
    SOO,
    MSO,
)

# Swarm‑based algorithms, including Beluga Whale Optimization (BWO)
from .swarm_based import (
    ABC,
    ACOR,
    AGTO,
    ALO,
    AO,
    ARO,
    AVOA,
    BA,
    BeesA,
    BES,
    BFO,
    BSA,
    BWO,
    COA,
    CoatiOA,
    CSA,
    CSO,
    DMOA,
    DO,
    EHO,
    ESOA,
    FA,
    FFA,
    FFO,
    FOA,
    FOX,
    GJO,
    GOA,
    GTO,
    GWO,
    HBA,
    HGS,
    HHO,
    JA,
    MFO,
    MGO,
    MPA,
    MRFO,
    MSA,
    MShOA,
    NGO,
    NMRA,
    OOA,
    PFA,
    POA,
    PSO,
    SCSO,
    SeaHO,
    ServalOA,
    SFO,
    SHO,
    SLO,
    SRSR,
    SSA,
    SSO,
    SSpiderA,
    SSpiderO,
    STO,
    TDO,
    TSO,
    WaOA,
    WOA,
    ZOA,
    EPC,
    SMO,
    SquirrelSA,
    FDO,
)

# System, music, game, and other categories
from .system_based import AEO, GCO, WCA
from .music_based import HS
from .game_based import THRO
from .sota_based import LSHADEcnEpSin, IMODE

from .utils.problem import Problem
from .utils.termination import Termination
from .tuner import Tuner, ParameterGrid
from .multitask import Multitask
from .optimizer import Optimizer
from .utils.space import (
    IntegerVar,
    FloatVar,
    StringVar,
    BinaryVar,
    BoolVar,
    CategoricalVar,
    SequenceVar,
    PermutationVar,
    TransferBinaryVar,
    TransferBoolVar,
)

__EXCLUDE_MODULES = ["__builtins__", "current_module", "inspect", "sys"]

def get_all_optimizers(verbose: bool = True):
    """
    Get all available optimizer classes in Mealpy library

    Args:
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.ismodule(obj) and (name not in __EXCLUDE_MODULES):
            for cls_name, cls_obj in inspect.getmembers(obj):
                if inspect.isclass(cls_obj) and issubclass(cls_obj, Optimizer):
                    cls[cls_name] = cls_obj
    # Remove base class
    cls.pop("Optimizer", None)
    if verbose:
        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer} - {optimizer()}")
    return cls

def get_optimizer_by_class(class_name: str, verbose: bool = False):
    """
    Get an optimizer class by its class name

    Args:
        class_name (str): the classname of the optimizer (e.g., C_PSO, OriginalGA);
            don't pass the module name (e.g., PSO, GA)
        verbose (bool): whether to print the optimizer information

    Returns:
        optimizer (Optimizer): the actual optimizer class or None if the classname is not supported
    """
    try:
        all_optimizers = get_all_optimizers(verbose=verbose)
        return all_optimizers[class_name]
    except KeyError:
        print(
            f"Mealpy doesn't support optimizer named: {class_name}.\n"
            f"Please see the supported Optimizer name from here: "
            f"https://mealpy.readthedocs.io/en/latest/pages/support.html#classification-table"
        )
        return None

def get_optimizer_by_name(name: str, verbose: bool = False):
    """
    Get an optimizer class by name

    Args:
        name (str): the classname of the optimizer (e.g., OriginalGA, OriginalWOA);
            don't pass the module name (e.g., ABC, WOA, GA)
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}
    flag = False
    for module_name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.ismodule(obj) and (module_name == name) and (name not in __EXCLUDE_MODULES):
            flag = True
            for cls_name, cls_obj in inspect.getmembers(obj):
                if inspect.isclass(cls_obj) and issubclass(cls_obj, Optimizer):
                    cls[cls_name] = cls_obj
    if verbose:
        if not flag:
            print(
                f"Mealpy doesn't support optimizer named: {name}.\n"
                f"Please see the supported Optimizer name from here: "
                f"https://mealpy.readthedocs.io/en/latest/pages/support.html#classification-table"
            )
            return None
        # Remove base class
        cls.pop("Optimizer", None)
        print("Found algorithm: {name}, the supported variants are:")
        for algo_name, optimizer in cls.items():
            print(f"Optimizer: {algo_name} - {optimizer} - {optimizer()}")
    return cls

# trigger workflow
