============
Introduction
============

**MEALPY** (MEta-heuristic ALgorithms in PYthon) is the most comprehensive Python library for cutting-edge, nature-inspired meta-heuristic algorithms.
It is released under the **MIT** license.

- **Current version:** 3.0.3
- **Total algorithms:** 233
  - 206 official (originals, hybrids, and variants)
  - 27 custom-developed by our developers

.. _All releases: https://pypi.org/project/mealpy/#history

**Important:**
Different versions of MEALPY introduce different ways to define and pass hyperparameters. Please check your version carefully:

- `< 1.0.5`
- `1.1.0 - 1.2.2`
- `2.0.0 - 2.1.2`
- `2.2.0`
- `2.3.0`
- `2.4.0 - 2.4.2`: Supports discrete problems
- `2.5.1 - 2.5.4`: Define once, solve multiple problems
- `>= 3.0.0`: Fully object-oriented design


Goals of this framework:

- To share knowledge of the meta-heuristic field with everyone at no cost.
- To help researchers in all fields access optimization algorithms as quickly as possible.
- To implement both classical and state-of-the-art meta-heuristics, covering the entire history of meta-heuristics.


What MEALPY offers:

- Analyze the parameters of algorithms.
- Perform qualitative and quantitative analyses of algorithms.
- Analyze the rate of convergence of algorithms.
- Test and analyze the scalability and robustness of algorithms.

Want to request a new algorithm?
Open an `Issue ticket <https://github.com/thieu1995/mealpy/issues>`_ or build your own using MEALPY’s modular components.

* And please give us credits if you use this library, check some of my `previous paper`_. ::

   @article{van2023mealpy,
      title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
      author={Van Thieu, Nguyen and Mirjalili, Seyedali},
      journal={Journal of Systems Architecture},
      year={2023},
      publisher={Elsevier},
      doi={10.1016/j.sysarc.2023.102871}
   }

.. _previous paper: https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa


------------
Optimization
------------

Optimization is the process of finding the best solution from a set of feasible solutions. In real-world problems,
objective functions can be nonlinear, noisy, non-differentiable, or highly constrained—making traditional
gradient-based methods ineffective.

Meta-heuristic algorithms offer a powerful alternative. They do not require gradient information and can handle
a wide range of optimization types: continuous, discrete, constrained, and multi-objective.

A general optimization problem can be formulated as:

.. image:: /_static/images/general_format.png

Where `x` is the vector of decision variables (real, integer, or categorical), `f(x)` are the objective functions,
and `g(x), h(x)` are constraints.

Classical methods like Newton-Raphson work well for smooth problems, but fail on complex landscapes.
Meta-heuristics provide a flexible, robust way to solve challenging problems when exact solutions are impractical or impossible.

-------------------------
Meta-heuristic Algorithms
-------------------------

Meta-heuristics are high-level search strategies that balance **exploration** (global search) and **exploitation**
(local refinement). They are inspired by natural processes like evolution, swarm behavior, physics, and chemistry.

Key features:
- Use randomness to escape local optima
- Perform well on complex, multimodal problems
- Require minimal assumptions about the problem

.. image:: /_static/images/bio_inspired.png


Meta-heuristics can be:
- **Population-based**: e.g., Genetic Algorithm, PSO
- **Trajectory-based**: e.g., Simulated Annealing

.. image:: /_static/images/history_metaheuristics.png

Despite their stochastic nature, well-designed meta-heuristics often produce high-quality solutions consistently.
MEALPY offers 200+ implementations of such algorithms, from classical to cutting-edge.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
