============
Introduction
============

.. toctree::
   :maxdepth: 3


**MEALPY** (MEta-heuristic ALgorithms in PYthon) is the most comprehensive Python library for cutting-edge, nature-inspired meta-heuristic algorithms. It is officially released under the **MIT** license.

.. note::
    **Current Project Status**

    * **Current version:** ``3.0.3``
    * **Total algorithms:** 233
        * 206 official implementations (originals, hybrids, and variants)
        * 27 custom-developed algorithms by our core development team

    *Check out all historical updates on the* `PyPI Releases Page <https://pypi.org/project/mealpy/#history>`_.

.. important::
    **Version Compatibility Warning**

    Different versions of MEALPY introduce significant architectural changes regarding how hyperparameters are defined and passed. Please verify your installed version and adhere to its specific paradigm:

    * ``< 1.0.5``: Legacy implementation.
    * ``1.1.0 - 1.2.2``: Stable legacy structures.
    * ``2.0.0 - 2.1.2``: Core framework refactoring.
    * ``2.2.0``: Enhanced parameter mapping.
    * ``2.3.0``: Addition of advanced utility features.
    * ``2.4.0 - 2.4.2``: Native support for discrete problems introduced.
    * ``2.5.1 - 2.5.4``: "Define once, solve multiple problems" architecture.
    * ``>= 3.0.0``: **Fully Object-Oriented Design (Current Standard).**

Goals of this Framework
-----------------------
* To share comprehensive knowledge of the meta-heuristic field with the global community at no cost.
* To help researchers across all domains access robust optimization algorithms as quickly as possible.
* To implement both classical and state-of-the-art meta-heuristics, effectively preserving the entire history of the field.

What MEALPY Offers
------------------
* Detailed analysis of algorithm hyperparameters.
* Robust frameworks for qualitative and quantitative performance analysis.
* Convergence rate evaluation across different algorithmic strategies.
* Scalability and robustness testing for varying problem complexities.

.. hint::
    **Contributing & Citations**

    Want to request a new algorithm? Open an `Issue ticket <https://github.com/thieu1995/mealpy/issues>`_ or build your own using MEALPY’s modular components.

    If you utilize MEALPY in your academic or professional research, please credit our work by citing our main paper:

.. code-block:: bibtex

   @article{van2023mealpy,
      title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
      author={Van Thieu, Nguyen and Mirjalili, Seyedali},
      journal={Journal of Systems Architecture},
      year={2023},
      publisher={Elsevier},
      doi={10.1016/j.sysarc.2023.102871}
   }

.. _previous paper: https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa


Optimization
------------

Optimization is the mathematical process of finding the absolute best solution from a set of feasible solutions. In real-world applications, objective functions are often nonlinear, noisy, non-differentiable, or highly constrained—making traditional gradient-based methods ineffective.

Meta-heuristic algorithms offer a powerful alternative. They do not require analytical gradient information and can seamlessly handle a wide array of optimization types: continuous, discrete, constrained, and multi-objective.

A general continuous optimization problem can be formulated as follows:

.. image:: /_static/images/general_format.png
   :align: center
   :alt: General Mathematical Formulation of an Optimization Problem

Where:
* :math:`x` is the vector of decision variables (real, integer, or categorical).
* :math:`f(x)` represents the objective functions to be minimized or maximized.
* :math:`g(x)` and :math:`h(x)` represent the inequality and equality constraints, respectively.

While classical methods (like Newton-Raphson) work exceptionally well for smooth, convex problems, they fail on complex fitness landscapes. Meta-heuristics provide a flexible, robust mechanism to solve challenging problems when deriving exact mathematical solutions is impractical or impossible.


Meta-heuristic Algorithms
-------------------------

Meta-heuristics are high-level algorithmic search strategies designed to strike a delicate balance between **exploration** (global search across the solution space) and **exploitation** (local refinement around promising areas). They draw heavy inspiration from natural processes such as evolution, swarm behavior, physics, and chemical reactions.

.. note::
    **Key Features of Meta-heuristics:**

    * Utilize stochastic (random) elements to escape local optima traps.
    * Perform exceptionally well on complex, multimodal, and non-differentiable problems.
    * Treat the objective function as a "black box," requiring minimal to no assumptions about the underlying mathematical problem.

.. image:: /_static/images/bio_inspired.png
   :align: center
   :alt: Bio-inspired Meta-heuristics Classification

Broadly, meta-heuristics can be categorized into two structural paradigms:

1. **Population-based Methods:** Evolve a set of solutions simultaneously, sharing information across the group (e.g., Genetic Algorithm, Particle Swarm Optimization).
2. **Trajectory-based Methods:** Trace a single solution path through the search space over time (e.g., Simulated Annealing).

.. image:: /_static/images/history_metaheuristics.png
   :align: center
   :alt: History and Timeline of Meta-heuristic Algorithms

Despite their inherent stochastic nature, well-designed meta-heuristics consistently converge to high-quality solutions. **MEALPY provides over 200 standardized implementations** of these algorithms, ranging from foundational classical models to the absolute cutting edge of current research.
