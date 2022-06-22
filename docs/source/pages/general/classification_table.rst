
* Meta-heuristic Categories: (Based on `this article`_)
    + Evolutionary-based: Idea from Darwin's law of natural selection, evolutionary computing
    + Swarm-based: Idea from movement, interaction of birds, organization of social ...
    + Physics-based: Idea from physics law such as Newton's law of universal gravitation, black hole, multiverse
    + Human-based: Idea from human interaction such as queuing search, teaching learning, ...
    + Biology-based: Idea from biology creature (or microorganism),...
    + System-based: Idea from eco-system, immune-system, network-system, ...
    + Math-based: Idea from mathematical form or mathematical law such as sin-cosin
    + Music-based: Idea from music instrument
    + Probabilistic-base: Probabilistic based algorithm
    + Dummy: Non-sense algorithms and Non-sense papers (code proofs)

.. _this article: https://doi.org/10.1016/j.procs.2020.09.075

* DBSP: Difference Between Sequential and Parallel training mode, the results of some algorithms may various due to the training mode.
	* significant: The results will be very different (because the selecting process - select a previous or the next solution to update current solution)
	* in-significant: The results will not much different (because the selecting process - select a random solution in population to update the current solution)

* Performance (Personal Opinion):
	* good: working good with benchmark functions (convergence good)
	* not good: not working good with benchmark functions (convergence not good, not balance the exploration and exploitation phase)

* Paras: The number of parameters in the algorithm (Not counting the fixed parameters in the original paper)
    + Almost algorithms have 2 paras (epoch, population_size) and plus some paras depend on each algorithm.
    + Some algorithms belong to "good" performance and have only 2 paras meaning the algorithms are outstanding

* Difficulty - Difficulty Level (Personal Opinion): Objective observation from author. Depend on the number of parameters, number of equations, the original ideas, time spend for coding, source lines of code (SLOC).
    + Easy: A few paras, few equations, SLOC very short
    + Medium: more equations than Easy level, SLOC longer than Easy level
    + Hard: Lots of equations, SLOC longer than Medium level, the paper hard to read.
    + Hard* - Very hard: Lots of equations, SLOC too long, the paper is very hard to read.

**For newbie, I recommend to read the paper of algorithms which difficulty is "easy" or "medium" difficulty level.**


 =============== ============================================ =========== ================= ======= ======== =============
  Group           Name                                         Module      Class             Year    Paras    Difficulty
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Evolutionary Programming                     EP          BaseEP            1964    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Evolution Strategies                         ES          BaseES            1971    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Memetic Algorithm                            MA          BaseMA            1989    7        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Genetic Algorithm                            GA          BaseGA            1992    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Differential Evolution                       DE          BaseDE            1997    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             JADE              2009    6        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             SADE              2005    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             SHADE             2013    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             L_SHADE           2014    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             SAP_DE            2006    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Flower Pollination Algorithm                 FPA         BaseFPA           2014    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary    Coral Reefs Optimization                     CRO         BaseCRO           2014    11       medium
 =============== ============================================ =========== ================= ======= ======== =============
  Evolutionary                                                             OCRO              2019    12       medium
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Particle Swarm Optimization                  PSO         BasePSO           1995    6        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    PPSO              2019    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    HPSO_TVAC         2017    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    C_PSO             2015    6        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    CL_PSO            2006    6        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Bacterial Foraging Optimization              BFO         OriginalBFO       2002    10       hard
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    ABFO              2019    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Bees Algorithm                               BeesA       BaseBeesA         2005    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    ProbBeesA         2015    5        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Cat Swarm Optimization                       CSO         BaseCSO           2006    11       hard
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Artificial Bee Colony                        ABC         BaseABC           2007    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Ant Colony Optimization                      ACO-R       BaseACOR          2008    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Cuckoo Search Algorithm                      CSA         BaseCSA           2009    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Firefly Algorithm                            FFA         BaseFFA           2009    8        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Fireworks Algorithm                          FA          BaseFA            2010    7        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Bat Algorithm                                BA          OriginalBA        2010    6        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Fruit-fly Optimization Algorithm             FOA         OriginalFOA       2012    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    WhaleFOA          2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Social Spider Optimization                   SSpiderO    BaseSSpiderO      2018    4        hard*
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Grey Wolf Optimizer                          GWO         BaseGWO           2014    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    RW_GWO            2019    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Social Spider Algorithm                      SSpiderA    BaseSSpiderA      2015    5        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Ant Lion Optimizer                           ALO         OriginalALO       2015    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Moth Flame Optimization                      MFO         OriginalMFO       2015    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Elephant Herding Optimization                EHO         BaseEHO           2015    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Jaya Algorithm                               JA          OriginalJA        2016    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    LevyJA            2021    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Whale Optimization Algorithm                 WOA         BaseWOA           2016    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm                                                                    HI_WOA            2019    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Dragonfly Optimization                       DO          BaseDO            2016    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Bird Swarm Algorithm                         BSA         BaseBSA           2016    9        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Spotted Hyena Optimizer                      SHO         BaseSHO           2017    6        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Salp Swarm Optimization                      SSO         BaseSSO           2017    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Swarm Robotics Search And Rescue             SRSR        BaseSRSR          2017    2        hard*
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Grasshopper Optimisation Algorithm           GOA         BaseGOA           2017    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Coyote Optimization Algorithm                COA         BaseCOA           2018    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Moth Search Algorithm                        MSA         BaseMSA           2018    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Sea Lion Optimization                        SLO         BaseSLO           2019    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Nake Mole-Rat Algorithm                      NMRA        BaseNMRA          2019    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Pathfinder Algorithm                         PFA         BasePFA           2019    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Sailfish Optimizer                           SFO         BaseSFO           2019    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Harris Hawks Optimization                    HHO         BaseHHO           2019    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Manta Ray Foraging Optimization              MRFO        BaseMRFO          2020    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Bald Eagle Search                            BES         BaseBES           2020    7        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Sparrow Search Algorithm                     SSA         OriginalSSA       2020    5        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Hunger Games Search                          HGS         OriginalHGS       2021    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Swarm           Aquila Optimizer                             AO          OriginalAO        2021    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Simulated Annealling                         SA          BaseSA            1987    9        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Wind Driven Optimization                     WDO         BaseWDO           2013    7        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Multi-Verse Optimizer                        MVO         OriginalMVO       2016    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Tug of War Optimization                      TWO         BaseTWO           2016    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Physics                                                                  EnhancedTWO       2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Electromagnetic Field Optimization           EFO         OriginalEFO       2016    6        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Nuclear Reaction Optimization                NRO         BaseNRO           2019    2        hard*
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Henry Gas Solubility Optimization            HGSO        BaseHGSO          2019    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Atom Search Optimization                     ASO         BaseASO           2019    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Equilibrium Optimizer                        EO          BaseEO            2019    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Physics                                                                  ModifiedEO        2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics                                                                  AdaptiveEO        2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Physics         Archimedes Optimization Algorithm            ArchOA      OriginalArchOA    2021    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Culture Algorithm                            CA          OriginalCA        1994    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Imperialist Competitive Algorithm            ICA         BaseICA           2007    8        hard*
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Teaching Learning-based Optimization         TLO         OriginalTLO       2011    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Human                                                                    ITLO              2013    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Brain Storm Optimization                     BSO         BaseBSO           2011    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Queuing Search Algorithm                     QSA         OriginalQSA       2019    2        hard
 =============== ============================================ =========== ================= ======= ======== =============
  Human                                                                    ImprovedQSA       2021    2        hard
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Search And Rescue Optimization               SARO        OriginalSARO      2019    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Life Choice-Based Optimization               LCO         OriginalLCO       2019    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Social Ski-Driver Optimization               SSDO        BaseSSDO          2019    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Gaining Sharing Knowledge-based Algorithm    GSKA        OriginalGSKA      2019    6        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Coronavirus Herd Immunity Optimization       CHIO        OriginalCHIO      2020    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Forensic-Based Investigation Optimization    FBIO        OriginalFBIO      2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Human           Battle Royale Optimization                   BRO         OriginalBRO       2020    3        medium
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Invasive Weed Optimization                   IWO         OriginalIWO       2006    7        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Biogeography-Based Optimization              BBO         OriginalBBO       2008    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Virus Colony Search                          VCS         OriginalVCS       2016    4        hard*
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Satin Bowerbird Optimizer                    SBO         OriginalSBO       2017    5        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Earthworm Optimisation Algorithm             EOA         BaseEOA           2018    8        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Wildebeest Herd Optimization                 WHO         BaseWHO           2019    12       medium
 =============== ============================================ =========== ================= ======= ======== =============
  Bio             Slime Mould Algorithm                        SMA         OriginalSMA       2020    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  System          Germinal Center Optimization                 GCO         OriginalGCO       2018    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  System          Water Cycle Algorithm                        WCA         BaseWCA           2012    5        medium
 =============== ============================================ =========== ================= ======= ======== =============
  System          Artificial Ecosystem-based Optimization      AEO         OriginalAEO       2019    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  System                                                                   EnhancedAEO       2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  System                                                                   ModifiedAEO       2020    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  System                                                                   IAEO              2021    2        medium
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Hill Climbing                                HC          OriginalHC        1993    3        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Cross-Entropy Method                         CEM         BaseCEM           1997    4        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Sine Cosine Algorithm                        SCA         OriginalSCA       2016    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Gradient-Based Optimizer                     GBO         OriginalGBO       2020    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Arithmetic Optimization Algorithm            AOA         OrginalAOA        2021    6        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Chaos Game Optimization                      CGO         OriginalCGO       2021    2        easy
 =============== ============================================ =========== ================= ======= ======== =============
  Math            Pareto-like Sequential Sampling              PSS         OriginalPSS       2021    4        medium
 =============== ============================================ =========== ================= ======= ======== =============
  0               0                                            0           0                 0       0        0
 =============== ============================================ =========== ================= ======= ======== =============
  Music           Harmony Search                               HS          OriginalHS        2001    4        easy
 =============== ============================================ =========== ================= ======= ======== =============


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

