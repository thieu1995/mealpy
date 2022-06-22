============
Introduction
============


------
MEALPY
------

.. image:: https://img.shields.io/badge/release-2.4.2-yellow.svg
   :target: https://github.com/thieu1995/mealpy/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/mealpy

.. image:: https://badge.fury.io/py/mealpy.svg
   :target: https://badge.fury.io/py/mealpy

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3711948.svg
   :target: https://doi.org/10.5281/zenodo.3711948

.. image:: https://readthedocs.org/projects/mealpy/badge/?version=latest
   :target: https://mealpy.readthedocs.io/en/latest/?badge=latest

.. image:: https://pepy.tech/badge/mealpy
   :target: https://pepy.tech/project/mealpy

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


* MEALPY is a largest python module for the most of cutting-edge nature-inspired meta-heuristic
  algorithms and is distributed under GNU General Public License (GPL) V3 license.

* Current version: 2.4.2, Total algorithms: 84 original, 24 official variants, 38 developed variants, 9 dummies.

* Different versions of mealpy in term of passing hyper-parameters. So please careful check your version before
  using this library (Check `All releases`_)
   * mealpy < 1.0.5
   * 1.1.0 < mealpy < 1.2.2
   * 2.0.0 <= mealpy <= 2.1.2
   * mealpy == 2.2.0
   * mealpy == 2.3.0
   * 2.4.0 <= mealpy <= 2.4.2 (From this version, algorithms can solve discrete problem)

.. _All releases: https://pypi.org/project/mealpy/#history

* The goals of this framework are:
    * Sharing knowledge of meta-heuristic fields to everyone without a fee
    * Helping other researchers in all field access to optimization algorithms as quickly as possible
    * Implement the classical as well as the state-of-the-art meta-heuristics (The whole history of meta-heuristics)

* What you can do with this library:
    * Analyse parameters of algorithms.
    * Perform Qualitative Analysis of algorithms.
    * Perform Quantitative Analysis of algorithms.
    * Analyse rate of convergence of algorithms.
    * Test the scalability of algorithms.
    * Analyse the stability of algorithms.
    * Analyse the robustness of algorithms.


* If you guys want a new algorithm, please open an `Issues ticket`_

.. _Issues ticket: https://github.com/thieu1995/mealpy/issues


* If you are facing multiple/many objective optimization problems, you can use Mealpy with weighted-sum method to
  transform it into single-objective optimization problem. But you want to find Pareto front / Reference front, then
  I recommend to checkout the `PYMOO library`_.

.. _PYMOO library: https://pymoo.org/

* And please give me some credits if you use this library, check some of my `first-author paper`_.

.. _first-author paper: https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa

::

	@software{thieu_nguyen_2020_3711949,
	  author       = {Nguyen Van Thieu},
	  title        = {A collection of the state-of-the-art Meta-heuristics Algorithms in Python: Mealpy},
	  month        = march,
	  year         = 2020,
	  publisher    = {Zenodo},
	  doi          = {10.5281/zenodo.3711948},
	  url          = {https://doi.org/10.5281/zenodo.3711948}
	}


------------
Optimization
------------
A very short introduction to meta-heuristic algorithms and how to use them to solve optimization problems. This document also introduces some basic concepts and conventions.

Meta-heuristic algorithms are becoming increasingly popular in optimization and applications over the last three decades. There are many reasons for this popularity and success, and one of the main reasons is that these algorithms have been developed by mimicking the most successful processes in nature, including biological systems, and physical and chemical processes. For most algorithms, we know their fundamental components, how exactly they interact to achieve efficiency remains partly a mystery, which inspires more active studies. Convergence analysis of a few algorithms such as particle swarm optimization shows some insight, but in general mathematical analysis of metaheuristic algorithms remains unsolved and still an ongoing active research topic.


The solution to an optimization problem requires the choice and the correct use of the right algorithm. The choice of an algorithm largely depends on the type of optimization problem at hand. For large-scale nonlinear global optimization problems, there is no agreed guideline for how to choose and what to choose. We are not even sure whether an efficient algorithm exists, which is especially true for NP-hard problems, and the most real-world problems often are NP-hard indeed and in most applications, we can in general write an optimization problem as the following generic form:

.. image:: /_static/images/general_format.png

Here the components xi of x are called design or decision variables, and they can be real continuous, discrete, or a mix of these two. The functions fi(x)
wherei= 1,2, ..., Mare called the objective functions or simply cost functions, and in the case of M= 1, there is only a single objective. The space spanned by
the decision variables is called the design space or search space Rn, while the space formed by the objective function values is called the solution space or
response space. The equalities for hj and inequalities for gk are called constraints. It is worth pointing out that we can also write the inequalities in the
other way ≥ 0, and we can also formulate the objectives as a maximization problem.


The algorithms used for solving optimization problems can be very diverse, from conventional algorithms to modern meta-heuristics Most conventional or
classic algorithms are deterministic.  For example, the simplex method in linear programming is deterministic. Some deterministic
optimization algorithms used gradient information, they are called gradient-based algorithms. For example, the well-known Newton-Raphson algorithm is
gradient-based, as it uses the function values and their derivatives, and it works extremely well for smooth unimodal problems. However, if there is some
discontinuity in the objective function, it does not work well. In this case, a non-gradient algorithm is preferred. Non-gradient-based, or
gradient-free/derivative-free, algorithms do not use any derivative, but only the function values. Hooke-Jeeves pattern search and Nelder-Mead downhill simplex
are examples of gradient-free algorithms


For stochastic algorithms, in general, we have two types: heuristic and meta-heuristic, though their difference is small. Loosely speaking, heuristic means ‘to
find’ or ‘to discover by trial and error’. Quality solutions to a tough optimization problem can be found in a reasonable amount of time, but there is no
guarantee that optimal solutions are reached. It hopes that these algorithms work most of the time, but not necessarily all the time. This is good when we do
not necessarily want the best solutions but easily reachable and rather good solutions.




--------------------------
Meta-heuristics algorithms
--------------------------

In meta-heuristic algorithms,meta-means ‘beyond’ or ‘higher level’, and they generally perform better than simple heuristics. All meta-heuristic algorithms use
certain trade-offs of local search and global exploration. A variety of solutions are often realized via randomization. Despite the popularity of meta-heuristics,
there is no agreed definition of heuristics and meta-heuristics in the literature. Some researchers use heuristics and ‘meta-heuristics’ interchangeably.
However, the recent trend tends to name all stochastic algorithms with randomization and global exploration as meta-heuristic. In this review, we will also follow this convention.


.. image:: /_static/images/bio_inspired.png

Randomization provides a good way to move away from local search to the search on a global scale. Therefore, almost all meta-heuristic algorithms intend to
be suitable for global optimization. Meta-heuristics can be an efficient way to produce acceptable solutions by trial and error to a complex problem in a
reasonably practical time. The complexity of the problem of interest makes it impossible to search for every possible solution or combination, the aim is to find
a good feasible solution in an acceptable timescale. There is no guarantee that the best solutions can be found, and we even do not know whether an algorithm
will work and why it does work. The idea is to have an efficient but practical algorithm that will work most of the time and can produce good-quality
solutions. Among the found quality solutions, it is expected some of them are nearly optimal, though there is no guarantee for such optimality. The main
components of any metaheuristic algorithms are intensification and diversification, or exploitation and exploration (Blum and Roli 2003). Diversification
means generating diverse solutions to explore the search space on a global scale, while intensification means focusing on the search in a local
region by exploiting the information that a current good solution is found in this region. This is in combination with these lections of the best solutions.


The selection of the best ensures that the solutions will converge to optimality. On other hand, the diversification via randomization avoids the
solutions being trapped at local optima while increasing the diversity of the solutions. A good combination of these two major components will usually ensure
that global optimality is achievable.

Metaheuristic algorithms can be classified in many ways. One way is to classify them as population-based and trajectory-based. For example, genetic algorithms
are population-based as they use a set of strings, so is the particle swarm optimization (PSO) which uses multiple agents or particles (Kennedy and Eberhart
1995). On the other hand, simulated annealing uses a single agent or solution which moves through the design space or search space in a piecewise style
(Kirkpatrick et al. 1983).


.. image:: /_static/images/history_metaheuristics.png


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4