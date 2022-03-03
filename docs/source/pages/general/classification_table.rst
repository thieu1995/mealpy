
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


 =============== ====== ============================================ =========== ======= ================= ============== ======== =============
  Group           STT    Name                                         Short       Year    DBSP              Performance    Paras    Difficulty
 =============== ====== ============================================ =========== ======= ================= ============== ======== =============
  Evolutionary    1      Evolutionary Programming                     EP          1964    no                not good       3        easy
  Evolutionary    2      Evolution Strategies                         ES          1971    no                not good       3        easy
  Evolutionary    3      Memetic Algorithm                            MA          1989    significant       not good       7        easy
  Evolutionary    3      Genetic Algorithm                            GA          1992    in-significant    good           4        easy
  Evolutionary    4      Differential Evolution                       DE          1997    in-significant    good           4        easy
  Evolutionary    5      Flower Pollination Algorithm                 FPA         2014    in-significant    good           3        easy
  Evolutionary    6      Coral Reefs Optimization                     CRO         2014    in-significant    good           7        medium
  0               7
  Swarm           1      Particle Swarm Optimization                  PSO         1995    in-significant    good           6        easy
  Swarm           2      Bacterial Foraging Optimization              BFO         2002    no                good           9        hard
  Swarm           3      Bees Algorithm                               BeesA       2005    no                not good       9        medium
  Swarm           4      Cat Swarm Optimization                       CSO         2006    significant       not good       9        hard
  Swarm           5      Ant Colony Optimization                      ACO         2006    in-significant    good           5        medium
  Swarm           6      Artificial Bee Colony                        ABC         2007    no                good           8        easy
  Swarm           7      Ant Colony Optimization                      ACO-R       2008    in-significant    good           5        medium
  Swarm           8      Cuckoo Search Algorithm                      CSA         2009    in-significant    good           3        easy
  Swarm           9      Firefly Algorithm                            FFA         2009    significant       good           8        medium
  Swarm           10     Fireworks Algorithm                          FA          2010    significant       good           7        medium
  Swarm           11     Bat Algorithm                                BA          2010    no                not good       5        easy
  Swarm           12     Fruit-fly Optimization Algorithm             FOA         2012    no                not good       2        easy
  Swarm           13     Social Spider Optimization                   SSpiderO    2013    no                not good       3        hard*
  Swarm           14     Grey Wolf Optimizer                          GWO         2014    no                good           2        easy
  Swarm           15     Social Spider Algorithm                      SSpiderA    2015    no                not good       5        easy
  Swarm           16     Ant Lion Optimizer                           ALO         2015    no                good           2        medium
  Swarm           17     Moth Flame Optimization                      MFO         2015    no                good           2        easy
  Swarm           18     Elephant Herding Optimization                EHO         2015    significant       good           5        easy
  Swarm           19     Jaya Algorithm                               JA          2016    no                good           2        easy
  Swarm           20     Whale Optimization Algorithm                 WOA         2016    no                good           2        easy
  Swarm           21     Dragonfly Optimization                       DO          2016    significant       good           2        medium
  Swarm           22     Bird Swarm Algorithm                         BSA         2016    in-significant    good           9        medium
  Swarm           23     Spotted Hyena Optimizer                      SHO         2017    no                good           6        medium
  Swarm           24     Salp Swarm Optimization                      SSO         2017    significant       good           2        easy
  Swarm           25     Swarm Robotics Search And Rescue             SRSR        2017    in-significant    good           2        hard*
  Swarm           26     Grasshopper Optimisation Algorithm           GOA         2017    no                not good       3        easy
  Swarm           27     Coyote Optimization Algorithm                COA         2018    no                good           3        medium
  Swarm           28     Moth Search Algorithm                        MSA         2018    no                good           5        easy
  Swarm           29     Sea Lion Optimization                        SLO         2019    no                good           2        medium
  Swarm           30     Nake Mole-Rat Algorithm                      NMRA        2019    in-significant    good           3        easy
  Swarm           31     Bald Eagle Search                            BES         2019    in-significant    good           7        medium
  Swarm           32     Pathfinder Algorithm                         PFA         2019    significant       good           2        easy
  Swarm           33     Sailfish Optimizer                           SFO         2019    no                good           5        medium
  Swarm           34     Harris Hawks Optimization                    HHO         2019    significant       good           2        medium
  Swarm           35     Manta Ray Foraging Optimization              MRFO        2020    no                good           3        easy
  Swarm           36     Sparrow Search Algorithm                     SSA         2020    no                good           5        medium
  Swarm           37     Hunger Games Search                          HGS         2021    no                good           4        medium
  Swarm           38     Aquila Optimizer                             AO          2021    no                good           2        easy
  0               39
  Physics         1      Simulated Annealling                         SA          1987    in-significant    not good       9        medium
  Physics         2      Wind Driven Optimization                     WDO         2013    in-significant    good           7        easy
  Physics         3      Multi-Verse Optimizer                        MVO         2016    in-significant    good           3        easy
  Physics         4      Tug of War Optimization                      TWO         2016    in-significant    not good       2        easy
  Physics         5      Electromagnetic Field Optimization           EFO         2016    significant       good           6        easy
  Physics         6      Nuclear Reaction Optimization                NRO         2019    in-significant    good           2        hard*
  Physics         7      Henry Gas Solubility Optimization            HGSO        2019    significant       good           3        medium
  Physics         8      Atom Search Optimization                     ASO         2019    no                good           4        medium
  Physics         9      Equilibrium Optimizer                        EO          2019    no                good           2        easy
  Physics         10     Archimedes Optimization Algorithm            ArchOA      2021    in-significant    good           6        medium
  0               11
  Human           1      Culture Algorithm                            CA          1994    no                not good       3        easy
  Human           2      Imperialist Competitive Algorithm            ICA         2007    significant       good           10       hard*
  Human           3      Teaching Learning-based Optimization         TLO         2011    in-significant    good           2        easy
  Human           4      Brain Storm Optimization                     BSO         2011    in-significant    not good       10       medium
  Human           5      Queuing Search Algorithm                     QSA         2019    in-significant    good           2        hard
  Human           6      Search And Rescue Optimization               SARO        2019    in-significant    good           4        medium
  Human           7      Life Choice-Based Optimization               LCO         2019    significant       good           2        easy
  Human           8      Social Ski-Driver Optimization               SSDO        2019    significant       good           2        easy
  Human           9      Gaining Sharing Knowledge-based Algorithm    GSKA        2019    significant       good           6        easy
  Human           10     Coronavirus Herd Immunity Optimization       CHIO        2020    significant       not good       4        medium
  Human           11     Forensic-Based Investigation Optimization    FBIO        2020    no                good           2        medium
  Human           12     Battle Royale Optimization                   BRO         2020    in-significant    not good       2        medium
  0               13
  Bio             1      Invasive Weed Optimization                   IWO         2006    no                good           5        easy
  Bio             2      Biogeography-Based Optimization              BBO         2008    in-significant    good           4        easy
  Bio             3      Virus Colony Search                          VCS         2016    significant       good           4        hard*
  Bio             4      Satin Bowerbird Optimizer                    SBO         2017    in-significant    good           5        easy
  Bio             5      Earthworm Optimisation Algorithm             EOA         2018    in-significant    good           8        medium
  Bio             6      Wildebeest Herd Optimization                 WHO         2019    no                good           12       medium
  Bio             7      Slime Mould Algorithm                        SMA         2020    in-significant    good           3        easy
  0               8
  System          1      Germinal Center Optimization                 GCO         2018    in-significant    good           4        medium
  System          2      Water Cycle Algorithm                        WCA         2012    in-significant    good           5        medium
  System          3      Artificial Ecosystem-based Optimization      AEO         2019    no                good           2        easy
  0               4
  Math            1      Hill Climbing                                HC          1993    no                not good       3        easy
  Math            2      Sine Cosine Algorithm                        SCA         2016    no                good           2        easy
  Math            3      Gradient-Based Optimizer                     GBO         2020    no                good           3        medium
  Math            4      Arithmetic Optimization Algorithm            AOA         2021    no                good           6        easy
  Math            5      Chaos Game Optimization                      CGO         2021    no                good           2        easy
  0               6
  Music           1      Harmony Search                               HS          2001    no                good           5        easy
  0               2
  Probabilistic   1      Cross-Entropy Method                         CEM         1997    in-significant    good           4        easy
  0               2
  Dummy           1      Pigeon-Inspired Optimization                 PIO         2014    good              2              medium
  Dummy           2      Artificial Algae Algorithm                   AAA         2015    not good          5              medium
  Dummy           3      Rhino Herd Optimization                      RHO         2018    not good          6              easy
  Dummy           4      Emperor Penguin Optimizer                    EPO         2018    good              2              easy
  Dummy           5      Butterfly Optimization Algorithm             BOA         2019    not good          6              medium
  Dummy           6      Blue Monkey Optimization                     BMO         2019    not good          3              medium
  Dummy           7      Sandpiper Optimization Algorithm             SOA         2020    not good          2              easy
  Dummy           8      Black Widow Optimization                     BWO         2020    good              5              medium
 =============== ====== ============================================ =========== ======= ================= ============== ======== =============




.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

