# A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon (mealpy)
[![PyPI version](https://badge.fury.io/py/mealpy.svg)](https://badge.fury.io/py/mealpy)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3711949.svg)](https://doi.org/10.5281/zenodo.3711948)

## mealpy
mealpy is a python module for the most of cutting-edge population meta-heuristic algorithms and is distributed
under MIT license.

## Installation

### Dependencies
* Python (>= 3.6)
* Numpy (>= 1.15.1)

### User installation
Install the [current PyPI release](https://pypi.python.org/pypi/mealpy):
```code 
    pip install mealpy
    pip install --upgrade mealpy 
```
Or install the development version from GitHub:
```bash
    pip install git+https://github.com/thieunguyen5991/mealpy
```

### Example
```code 
    python examples/simple_run.py
```
The documentation includes more detailed installation instructions.

### Changelog
* See the "ChangeLog.md" for a history of notable changes to mealpy.


### Important links

* Official source code repo: https://github.com/thieunguyen5991/mealpy
* Download releases: https://pypi.org/project/mealpy/
* Issue tracker: https://github.com/mealpy/mealpy/issues

* This project also related to my another projects which are "meta-heuristics" and "neural-network", check it here
    * https://github.com/thieunguyen5991/metaheuristics
    * https://github.com/chasebk
    

## Contributions 

### Citation
If you use mealpy in your project, I would appreciate citations: 

```code 
@software{thieu_nguyen_2020_3711949,
  author       = {Thieu Nguyen},
  title        = {A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon: Mealpy},
  month        = march,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3711948},
  url          = {https://doi.org/10.5281/zenodo.3711948}
}
```

* Nguyen, T., Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019). Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, 12(2), 1144-1161.

* Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

* Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.


### Documents
* Group: 
    + Evolu: Evolutionary-based
    + Swarm: Swarm-based
    + Physi: Physics-based
    + Human: Human-based
    + Bio: Biology-based
    
* Levy: Using levy-flight technique or not
* Version: 
    + original: Taken exactly from the paper
    + changed: I changed the flow or equation to make algorithm works
* Type:
    + weak: working fine with uni-modal and some multi-modal functions
    + strong: working good with uni-modal, multi-modal, some hybrid and some composite functions
    + best: working well with almost all kind of functions

* Some algorithm with original version and no levy techniques still belong to the best type such as:
    + Whale Optimization Algorithm
    + Bird Swarm Algorithm
    + Swarm Robotics Search And Rescue
    + Manta Ray Foraging Optimization
    + Henry Gas Solubility Optimization	
    + Atom Search Optimization
    + Equilibrium Optimizer

* Paras: The number of parameters in the algorithm (Don't count the fixed number in the original paper)
    + Almost algorithms has 2 (epoch, population_size) and plus some paras depend on each algorithm.
    + Some algorithms belong to "best" type and have only 2 paras meaning that algorithm is very good 
    
* Difficulty: Objective observation from author.
    + Depend on the number of parameters, number of equations, the original ideas, time spend for coding, source lines of code (SLOC)
    + Easy: A few paras, few equations, SLOC very short
    + Medium: more equations than Easy level, SLOC longer than Easy level
    + Hard: Lots of equations, SLOC longer than Medium level, the paper hard to read.
    + Very hard: Lots of equations, SLOC too long, the paper is very hard to read.

** For newbie, I recommend to read the paper of algorithms belong to "best or strong" type, "easy or medium" difficulty level.


| Group | STT | Name                               | Short | Year | Version   | Levy | Type   | Paras | Difficulty |
|-------|-----|------------------------------------|-------|------|-----------|------|--------|-------|------------|
| Evolu | 1   | Genetic Algorithm                  | GA    | 1992 | original  | no   | weak   | 4     | easy       |
|       | 2   | Differential Evolution             | DE    | 1997 | original  | no   | weak   | 4     | easy       |
|       | 3   |                                    |       |      |           |      |        |       |            |
| Swarm | 1   | Particle Swarm Optimization        | PSO   | 1995 | original  | no   | strong | 6     | easy       |
|       | 2   | Bacterial Foraging Optimization    | BFO   | 2002 | orginal   | no   | weak   | 11    | hard       |
|       | 3   | Cat Swarm Optimization             | CSO   | 2006 | original  | no   | weak   | 9     | hard       |
|       | 4   | Artificial Bee Colony              | ABC   | 2007 | changed   | no   | strong | 6     | easy       |
|       | 5   | Bat Algorithm                      | BA    | 2010 | original  | no   | weak   | 5     | easy       |
|       | 6   | Social Spider Optimization         | SSO   | 2013 | changed   | no   | weak   | 3     | very hard  |
|       | 7   | Grey Wolf Optimizer                | GWO   | 2014 | original  | no   | strong | 2     | easy       |
|       | 8   | Social Spider Algorithm            | SSA   | 2015 | original  | no   | strong | 5     | easy       |
|       | 9   | Ant Lion Optimizer                 | ALO   | 2015 | original  | no   | weak   | 2     | medium     |
|       | 10  | Moth Flame Optimization            | MFO   | 2015 | changed   | no   | strong | 2     | easy       |
|       | 11  | Whale Optimization Algorithm       | WOA   | 2016 | original  | no   | best   | 2     | easy       |
|       | 12  | Bird Swarm Algorithm               | BSA   | 2016 | original  | no   | best   | 9     | medium     |
|       | 13  | Swarm Robotics Search And Rescue   | SRSR  | 2017 | original  | no   | best   | 2     | very hard  |
|       | 14  | Grasshopper Optimisation Algorithm | GOA   | 2017 | original  | no   | weak   | 3     | easy       |
|       | 15  | Earthworm Optimisation Algorithm   | EOA   | 2018 | original  | no   | weak   | 8     | medium     |
|       | 16  | Moth Search Algorithm              | MSA   | 2018 | changed   | no   | weak   | 5     | easy       |
|       | 17  | Rhino Herd Optimization            | RHO   | 2018 | original  | no   | weak   | 6     | easy       |
|       | 18  | Emperor Penguin Optimizer          | EPO   | 2018 | changed   | no   | strong | 2     | easy       |
|       | 19  | Nake Mole\-rat Algorithm           | NMRA  | 2019 | original  | no   | strong | 3     | easy       |
|       | 20  | Bald Eagle Search                  | BES   | 2019 | changed   | no   | best   | 7     | medium     |
|       | 21  | Pathfinder Algorithm               | PFA   | 2019 | original  | no   | strong | 2     | easy       |
|       | 22  | Sailfish Optimizer                 | SFO   | 2019 | original  | no   | strong | 5     | medium     |
|       | 23  | Harris Hawks Optimization          | HHO   | 2019 | original  | yes  | best   | 2     | medium     |
|       | 24  | Manta Ray Foraging Optimization    | MRFO  | 2020 | original  | no   | best   | 3     | easy       |
|       | 25  |                                    |       |      |           |      |        |       |            |
| Physi | 1   | Wind Driven Optimization           | WDO   | 2013 | original  | no   | strong | 7     | easy       |
|       | 2   | Multi\-Verse Optimizer             | MVO   | 2016 | changed   | no   | strong | 3     | easy       |
|       | 3   | Tug of War Optimization            | TWO   | 2016 | original  | no   | strong | 2     | easy       |
|       | 4   | Electromagnetic Field Optimization | EFO   | 2016 | original  | no   | strong | 6     | easy       |
|       | 5   | Nuclear Reaction Optimization      | NRO   | 2019 | original  | yes  | best   | 2     | very hard  |
|       | 6   | Henry Gas Solubility Optimization  | HGSO  | 2019 | original  | no   | best   | 3     | medium     |
|       | 7   | Atom Search Optimization           | ASO   | 2019 | original  | no   | best   | 4     | medium     |
|       | 8   | Equilibrium Optimizer              | EO    | 2019 | original  | no   | best   | 2     | easy       |
|       | 9   |                                    |       |      |           |      |        |       |            |
| Human | 1   | Teaching Learning Optimization     | TLO   | 2011 | original  | no   | strong | 2     | easy       |
|       | 2   | Queuing Search Algorithm           | QSA   | 2019 | original  | no   | strong | 2     | hard       |
|       | 3   | Search And Rescue Optimization     | SARO  | 2019 | original  | no   | strong | 4     | medium     |
|       | 4   | Life Choice\-Based Optimization    | LCBO  | 2019 | original  | no   | strong | 2     | easy       |
|       | 5   | Social Ski\-Driver Optimization    | SSDO  | 2019 | changed   | no   | BEST   | 2     | easy       |
|       | 6   |                                    |       |      |           |      |        |       |            |
| Bio   | 1   | Invasive Weed Optimization         | IWO   | 2006 | original  | no   | strong | 5     | easy       |
|       | 2   | Biogeography\-Based Optimization   | BBO   | 2008 | original  | no   | strong | 4     | easy       |
|       | 3   |                                    |       |      |           |      |        |       |            |
