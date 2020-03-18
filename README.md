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




| Group        | STT | Name                               | Short | Year | Version   | Levy | Type   | Cite |
|--------------|-----|------------------------------------|-------|------|-----------|------|--------|------|
| Evolutionary | 1   | Genetic Algorithm                  | GA    | 1992 | original  | no   | weak   |      |
|              | 2   | Differential Evolution             | DE    | 1997 | original  | no   | weak   |      |
|              | 3   |                                    |       |      |           |      |        |      |
| Swarm        | 1   | Particle Swarm Optimization        | PSO   | 1995 | original  | no   | strong |      |
|              | 2   | Bacterial Foraging Optimization    | BFO   | 2002 | orginal   | no   | weak   |      |
|              | 3   | Cat Swarm Optimization             | CSO   | 2006 | original  | no   | weak   |      |
|              | 4   | Artificial Bee Colony              | ABC   | 2007 | changed   | no   | strong |      |
|              | 5   | Bat Algorithm                      | BA    | 2010 | original  | no   | weak   |      |
|              | 6   | Social Spider Optimization         | SSO   | 2013 | changed   | no   | weak   |      |
|              | 7   | Grey Wolf Optimizer                | GWO   | 2014 | original  | no   | strong |      |
|              | 8   | Social Spider Algorithm            | SSA   | 2015 | original  | no   | strong |      |
|              | 9   | Ant Lion Optimizer                 | ALO   | 2015 | original  | no   | weak   |      |
|              | 10  | Moth Flame Optimization            | MFO   | 2015 | changed   | no   | strong |      |
|              | 11  | Whale Optimization Algorithm       | WOA   | 2016 | original  | no   | best   |      |
|              | 12  | Bird Swarm Algorithm               | BSA   | 2016 | original  | no   | best   |      |
|              | 13  | Swarm Robotics Search And Rescue   | SRSR  | 2017 | original  | no   | best   |      |
|              | 14  | Grasshopper Optimisation Algorithm | GOA   | 2017 | original  | no   | weak   |      |
|              | 15  | Earthworm Optimisation Algorithm   | EOA   | 2018 | original  | no   | weak   |      |
|              | 16  | Moth Search Algorithm              | MSA   | 2018 | changed   | no   | weak   |      |
|              | 17  | Rhino Herd Optimization            | RHO   | 2018 | original  | no   | weak   |      |
|              | 18  | Emperor Penguin Optimizer          | EPO   | 2018 | changed   | no   | strong |      |
|              | 19  | Nake Mole\-rat Algorithm           | NMRA  | 2019 | original  | no   | strong |      |
|              | 20  | Bald Eagle Search                  | BES   | 2019 | changed   | no   | best   |      |
|              | 21  | Pathfinder Algorithm               | PFA   | 2019 | original  | no   | strong |      |
|              | 22  | Sailfish Optimizer                 | SFO   | 2019 | original  | no   | strong |      |
|              | 23  | Harris Hawks Optimization          | HHO   | 2019 | original  | yes  | best   |      |
|              | 24  | Manta Ray Foraging Optimization    | MRFO  | 2020 | original  | no   | best   |      |
|              | 25  |                                    |       |      |           |      |        |      |
| Physics      | 1   | Wind Driven Optimization           | WDO   | 2013 | original  | no   | strong |      |
|              | 2   | Multi\-Verse Optimizer             | MVO   | 2016 | changed   | no   | strong |      |
|              | 3   | Tug of War Optimization            | TWO   | 2016 | original  | no   | strong |      |
|              | 4   | Electromagnetic Field Optimization | EFO   | 2016 | original  | no   | strong |      |
|              | 5   | Nuclear Reaction Optimization      | NRO   | 2019 | original  | yes  | best   |      |
|              | 6   | Henry Gas Solubility Optimization  | HGSO  | 2019 | original  | no   | best   |      |
|              | 7   | Atom Search Optimization           | ASO   | 2019 | original  | no   | best   |      |
|              | 8   | Equilibrium Optimizer              | EO    | 2019 | original  | no   | best   |      |
